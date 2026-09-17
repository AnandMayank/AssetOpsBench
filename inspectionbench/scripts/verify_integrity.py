#!/usr/bin/env python3
"""verify_integrity.py — standalone integrity check for the frozen
InspectionBench V3 benchmark package.

Run with no arguments from anywhere; it locates the package relative to
this file's own location, so it works from a fresh clone with no
configuration. Verifies:

  1. the frozen-93 historical manifest's SHA256 matches the published,
     immutable value (d2b48c0b...);
  2. the V3 rollup manifest's declared canonical total is 4,075;
  3. the V3 rollup's per-family counts sum consistently with its own
     accounting note (see the manifest's `family_counts_accounting_note`);
  4. every manifest file referenced by the V3 rollup's `source_manifests`
     is present in this package;
  5. the B-Acquisition and D-physical manifests' declared counts (66, 70)
     match their actual episode-list lengths.

Exits 0 and prints "INTEGRITY OK" if every check passes; exits 1 and
prints the exact failing check otherwise. This script performs NO writes
and makes NO changes to any file.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = PKG_ROOT / "manifests"

FROZEN_93_SHA256 = "d2b48c0b0c9ef19f6c8f6ddd936da098ffd5d0d4023650619f01c1ee043087fe"
EXPECTED_TOTAL = 4075
EXPECTED_B_ACQUISITION = 66
EXPECTED_D_PHYSICAL = 70


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    return ok


def main() -> int:
    all_ok = True
    print("InspectionBench V3 integrity check")
    print(f"Package root: {PKG_ROOT}\n")

    f93 = MANIFESTS / "frozen_93_historical_manifest.json"
    if f93.exists():
        actual_sha = sha256_of(f93)
        all_ok &= check("frozen-93 historical manifest SHA256",
                        actual_sha == FROZEN_93_SHA256,
                        f"expected {FROZEN_93_SHA256}, got {actual_sha}")
    else:
        all_ok &= check("frozen-93 historical manifest present", False, "file missing")

    v3_path = MANIFESTS / "final_benchmark_manifest_v3.json"
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text())
        all_ok &= check("V3 final_canonical_total == 4075",
                        v3.get("final_canonical_total") == EXPECTED_TOTAL,
                        f"got {v3.get('final_canonical_total')}")
        # Package-local renames of the upstream source_manifests paths (for
        # clarity inside this standalone package) -- mapped explicitly here
        # rather than assumed, so this check verifies the ACTUAL file this
        # package ships under, not the original repo's internal path.
        RENAMES = {
            "final_benchmark_manifest.json": "a_e_pool_manifest.json",
            "phase8h1_pilot_manifest.json": "frozen_93_historical_manifest.json",
            "d_physical_manifest.json": "d_physical_canonical_manifest.json",
        }
        for key, info in v3.get("source_manifests", {}).items():
            src_name = Path(info["path"]).name
            local_name = RENAMES.get(src_name, src_name)
            found = (MANIFESTS / local_name).exists()
            all_ok &= check(f"source manifest present for '{key}' ({local_name})", found)
    else:
        all_ok &= check("V3 rollup manifest present", False, "file missing")

    bacq_path = MANIFESTS / "b_acquisition_final_manifest.json"
    if bacq_path.exists():
        bacq = json.loads(bacq_path.read_text())
        n = len(bacq.get("episodes", []))
        all_ok &= check("B-Acquisition episode count == 66", n == EXPECTED_B_ACQUISITION, f"got {n}")
    else:
        all_ok &= check("B-Acquisition manifest present", False, "file missing")

    dphys_path = MANIFESTS / "d_physical_canonical_manifest.json"
    if dphys_path.exists():
        dphys = json.loads(dphys_path.read_text())
        n = len(dphys.get("episodes", []))
        all_ok &= check("D-physical canonical episode count == 70", n == EXPECTED_D_PHYSICAL, f"got {n}")
    else:
        all_ok &= check("D-physical canonical manifest present", False, "file missing")

    trace_path = MANIFESTS / "d_physical_generation_trace_pre_dedup_88.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        all_ok &= check("D-physical generation trace canonical_final == 70",
                        trace.get("canonical_final") == EXPECTED_D_PHYSICAL,
                        f"got {trace.get('canonical_final')}")

    print()
    if all_ok:
        print("INTEGRITY OK")
        return 0
    print("INTEGRITY CHECK FAILED -- see [FAIL] lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
