# AOBv2-REAL-018 — decommissioned-instrument appendix example

**Image**: `AOBv2-REAL-018_out_of_service_gauge_IEA762.jpg` (this directory).
**Source**: `data_detection/images/test/IMG_20220106_105244.jpg` (the PMC real-image gauge track),
referenced as `scenario_id=AOBv2-REAL-018` in `src/orchestrator/data/perception_real.csv` and
`pairs_full.csv`.

## What the image shows
A 0–50°C compressor/pump temperature gauge carrying an orange **停用证** ("out-of-service
certificate") tag: `NO: IEA762`, decommission date `2021年7月15日` (15 July 2021), with a
handwritten confirming signature. A hand-drawn red line marks the 45–50°C band on the dial face
separately from the tag. The needle sits near 0.

## Why it's a good appendix case
This scenario is an already-recorded, real dual-labeling disagreement — not a hypothetical:

`reports/v7/label_adjudication_worksheet.csv` (status: `conflict`) shows two independent labelers
disagreeing on nearly every field for this image:
- `gauge_readable`: `true` vs `false`
- `gauge_value`: `48` vs `UNREADABLE`
- `asset`: `Compressor` vs `Industrial pump`
- `image_role`: `reference` vs `query`

`reports/v7/label_status.csv` confirms `gauge_readable`, `gauge_value`, `asset`, and `image_role`
are all `contested` for this scenario (dual-labeling disagreement).

Despite disagreeing on everything else, **both labelers' free-text descriptions independently
converged on the same root cause**: *"A robot inspector would face a challenge reading this gauge
due to the '停用证' (decommissioning certificate) label partially covering the dial face and
obscuring some of the numerical markings."*

The scenario's schema records `recommended_action: CLEAN_GAUGE` (i.e. the obstruction is treated
as a cleaning problem) — but the actual obstruction is an administrative decommission tag, not
dirt or glare. No field in the current annotation schema (`category`, `gauge_readable`,
`gauge_value`, `recommended_action`) can express "this instrument has been formally taken out of
service," which is the fact that should actually govern the correct downstream action, independent
of whether the dial happens to be numerically readable underneath the tag.

## Suggested appendix framing
Use this as a worked case in the cross-checking/human-verification discussion: dual-labeling
disagreement here is not labeler error — it is both labelers correctly noticing a real-world
condition (decommissioning) that the annotation schema has no field for, which is why they fell
back to disagreeing on `readable`/`value`/`asset` instead. This motivates adding an explicit
`instrument_status` (or similar) field distinct from `gauge_readable`, rather than treating this
disagreement as pure annotator noise.
