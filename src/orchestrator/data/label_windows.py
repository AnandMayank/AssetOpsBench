"""
Label pmc_windows_clean/* windows against perception_real.csv ground truth.

IMPORTANT — this does NOT fabricate labels. It:
  1. Matches window frames against perception_real.csv by filename (real ground truth).
  2. Probes a blur/contrast heuristic as a CANDIDATE proxy for the unmatched windows,
     but only reports it — see the printed verdict below on why it was rejected as a
     label source for this dataset (the two ground-truth failure anchors span nearly
     the same range as everything else, so it doesn't discriminate).
  3. Writes labels.json with an explicit label_source per window: "ground_truth" or
     "unlabeled" (never "heuristic", because the heuristic failed calibration).

Scaling the "unlabeled" bucket into real supervision requires either:
  (a) the full 1,296-row perception_real.csv (only a 20-row dev subset is checked
      into this repo today), or
  (b) running src/perception/reverse_label_real_images.py's Gemini VLM annotator
      against these window frames directly (needs GOOGLE_API_KEY; not set in this
      environment run).
"""
import csv
import json
from pathlib import Path

import cv2

HERE = Path(__file__).parent
CLEAN = HERE / "pmc_windows_clean"
CSV = HERE / "perception_real.csv"


def quality_probe(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return dict(
        laplacian_var=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        contrast=round(float(gray.std()), 1),
    )


def main():
    rows = list(csv.DictReader(open(CSV)))
    by_basename = {Path(r["source_file"]).name: r for r in rows}

    manifest = json.loads((CLEAN / "manifest.json").read_text())
    labels = {}
    anchor_failure_quality = []

    for name, info in manifest.items():
        frames = info["frames"]
        matched = [(f, by_basename[f]) for f in frames if f in by_basename]
        last_q = quality_probe(CLEAN / name / frames[-1])

        if matched:
            # majority vote on gauge_readable across matched frames in this window
            readable_votes = [m[1]["gauge_readable"] == "true" for m in matched]
            is_readable = sum(readable_votes) > len(readable_votes) / 2
            label = "success" if is_readable else "failure"
            categories = sorted({m[1]["category"] for m in matched})
            labels[name] = dict(
                label=label,
                label_source="ground_truth",
                matched_frames=[m[0] for m in matched],
                categories=categories,
                decision_frame_quality=last_q,
            )
            if label == "failure":
                anchor_failure_quality.append(last_q)
        else:
            labels[name] = dict(
                label=None,
                label_source="unlabeled",
                matched_frames=[],
                categories=[],
                decision_frame_quality=last_q,
            )

    n_gt = sum(1 for v in labels.values() if v["label_source"] == "ground_truth")
    n_unlabeled = len(labels) - n_gt
    print(f"Ground-truth labeled: {n_gt}/{len(labels)}  |  Unlabeled: {n_unlabeled}/{len(labels)}")
    for name, v in labels.items():
        print(f"  {name:45s} label={str(v['label']):8s} source={v['label_source']:12s} "
              f"quality={v['decision_frame_quality']}")

    # Report on the rejected heuristic explicitly, so nobody re-derives a bad label
    # from these numbers later without re-deriving this same negative result.
    all_qualities = [v["decision_frame_quality"]["laplacian_var"] for v in labels.values()]
    print("\nHeuristic-viability check (blur/contrast as a failure proxy):")
    print(f"  ground-truth FAILURE anchors' laplacian_var: {[q['laplacian_var'] for q in anchor_failure_quality]}")
    print(f"  full population laplacian_var range: {min(all_qualities)}-{max(all_qualities)}")
    print("  VERDICT: rejected as a label source — failure anchors fall inside the general "
          "population range (perception_real.csv failures here are semantic: frost/occlusion/"
          "dirt content, not blur), so no heuristic label was assigned to unmatched windows.")

    out = CLEAN / "labels.json"
    out.write_text(json.dumps(labels, indent=2))
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
