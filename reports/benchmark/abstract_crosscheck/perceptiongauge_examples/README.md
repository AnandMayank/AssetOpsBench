# PerceptionGauge transformation examples — real, freshly run this session

These are genuine outputs of the unmodified transform functions in
`src/perception/generate_perception_gauge_49.py` (`apply_gauge_degradation`, `apply_occlusion`,
`apply_scale_interpretation`, `apply_glare`, `apply_iot_overlay`), run this session on two real base
photographs from the `AssetOps Gauge` sample pool (the same real-world facility photo source already
used and cleared elsewhere in this project's PerceptionGauge/PMC work). **This is an illustrative
sample, not the full 49-row catalog run** — see `appendix_perceptiongauge_rendering.md` (one directory
up) for the repo action still required to freeze the complete pipeline output. Seed = 42, matching the
generator's own default.

## Files

Two source photographs, each with its reference and five category transforms:

| File | What it shows |
|---|---|
| `example{1,2}_reference.jpg` | The unmodified base photograph (a real pressure gauge on industrial piping) |
| `example{1,2}_gauge_degradation_soot.jpg` | `apply_gauge_degradation(img, "soot", rng)` — dark brownish overlay + Gaussian pixel noise, simulating soot/carbon fouling on the dial |
| `example{1,2}_occlusion.jpg` | `apply_occlusion(img, rng)` — an opaque structure blocking roughly a third of the frame (side chosen at random: left/right/top) |
| `example{1,2}_scale_interpretation.jpg` | `apply_scale_interpretation(img, rng)` — the gauge downscaled to 18–35% and re-composited onto a flat grey panel, simulating a distant standoff read |
| `example{1,2}_glare_lighting.jpg` | `apply_glare(img, rng)` — a centre-weighted blur blend simulating heat-haze/glare distortion over the dial |
| `example{1,2}_iot_contradiction.jpg` | `apply_iot_overlay(img, iot_value="245 bar", gauge_value="238 bar", rng)` — a rendered telemetry banner showing a contradicting IoT reading (illustrative values, not tied to any real sensor) |

## Suggested appendix caption

> Figure `[X]`: PerceptionGauge transformation examples. Each row shows one real base photograph
> (left) and its five category transforms (soot degradation, occlusion, scale/standoff simulation,
> glare, and IoT-contradiction overlay), produced by the unmodified transform functions described in
> Appendix `[X]` (rendering-mechanism detail). These are illustrative examples generated for this
> appendix, not scenes from the full seed-matrix run — see the frozen-manifest requirement in Appendix
> `[X]` before citing a total scene count.

## Provenance note

Source photographs are from the `AssetOps Gauge` real-world sample pool
(`external_datasets/real_world_samples/AssetOps Gauge/Gauge/Medium Gauge/`), the same source already
established as part of this project's real-photo data and previously cleared for publication in this
repository (see the AOBv2-REAL-018/AOBv2-REAL-008 appendix examples pushed earlier in this session,
drawn from the same PMC/real-world-samples data tier).
