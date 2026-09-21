# Appendix content — PerceptionGauge rendering pipeline and the 15 synthetic asset classes

Drop-in appendix text (and a paragraph to append after the "Procedurally rendered scenes" sentence in
Section 3.1). Every parameter below is quoted directly from
`src/perception/generate_perception_gauge_49.py`, re-opened this pass. This is the same source
already flagged in `section3_reproducibility_fixes.md` Item 3 — this document supplies the
*rendering-mechanism detail* and the *asset-class clarification* the paper currently omits, so the
two should be read together.

---

## Paragraph to append after "Procedurally rendered scenes..." (Section 3.1) — [MAIN]

> **Rendering mechanism.** Scenes are produced by 2D image compositing in Pillow (PIL), not a 3D or
> vector renderer. For each of the 49 seed rows, the pipeline starts from one base photograph (drawn
> from SyncG, an internally curated synthetic-gauge set, or a real facility photograph) and applies a
> category-specific transformation: gauge degradation (soot/frost/oil/salt/dust overlays via alpha
> compositing plus Gaussian blur and additive Gaussian pixel noise), occlusion (an opaque rectangle
> over a randomly chosen third of the frame, edge-softened with blur), scale interpretation
> (downscaling to 18–35% of source resolution and re-compositing onto a grey industrial-panel
> background, simulating a distant standoff view), glare (a radially weighted blend of the image with
> a heavily blurred copy of itself, strongest at frame centre), and IoT–physical contradiction
> (a rendered telemetry overlay banner showing the IoT and gauge values side by side with a
> contradiction flag). Each seed row additionally yields three lighting variants of its transformed
> image — under-lit (brightness ×0.55, contrast ×0.85), standard (×1.00/×1.00), and over-lit
> (×1.55/×1.20) — produced with PIL's `ImageEnhance`. This gives `[FILL AFTER FREEZE: N]` images per
> completed run (49 seeds × (1 reference + 1 transformed query + 3 lighting variants) = 245 images at
> minimum, before any additional per-category resampling); the specific 856/440/1,296 figures
> currently in the draft do not correspond to any output this pipeline has produced and should not be
> retained without a fresh, frozen run (see the release-manifest requirement below).
>
> **Parameter variation.** "Rendering parameters vary gauge scale and viewing distance" refers
> specifically to the scale-interpretation transform's resize factor (uniform-random 0.18–0.35× source
> resolution per instance) and to the choice of degradation sub-variant within each category (e.g.
> soot vs frost vs oil-film within `gauge_degradation`), both drawn from a seeded `random.Random`
> instance. One known reproducibility gap: the degradation transform's pixel-noise step
> (`apply_gauge_degradation`) additionally samples from `numpy`'s global RNG state, which is not
> seeded anywhere in the script — re-running with the same `--seed` therefore reproduces which
> transform and which resize factor were chosen, but not the exact noise pattern within the soot/fog
> overlays. This should be fixed (seed `numpy.random` explicitly) before the pipeline is treated as
> fully reproducible.

## Paragraph to append immediately after (or as a footnote to) the "15 asset classes" clause — [MAIN]

> **The 15 asset classes are synthetic coverage classes for the rendered-perception tier, not
> additional facility-grounded assets.** They label the *type of gauge/instrument depicted in the base
> photograph* used for compositing (bearings, gearboxes, induction motors, PMSM/synchronous motors,
> centrifugal pumps, reciprocating compressors, hydraulic systems, lithium-ion batteries, wind-turbine
> drivetrains, gas turbines, power transformers, HVAC chillers, HVAC AHUs, CNC machine tools, robotic
> manipulators), and exist to give the perception-robustness tier broad visual coverage independent of
> which physical asset a canonical episode is grounded in. They are **disjoint from the four canonical
> grounded assets** (`chiller_6`, `hydraulic_pump_1`, `metro_pump_1`, `motor_01`) that every scored A/B/
> C/D/E episode is actually built on: no PerceptionGauge scene is tied to a specific canonical
> `world_id`, `asset_id`, or scored episode, and no canonical episode's evidence delivery is sourced
> from a PerceptionGauge-rendered image. **Do not read "15 asset classes" as extending or
> re-characterising the benchmark's four-asset canonical scope** (the same confusion this audit already
> flagged for the "six industrial asset types" claim elsewhere in Section 3.1 — see
> `section3_reproducibility_fixes.md` Item C/F). If the paper wants to state the canonical-episode
> asset count anywhere near this sentence, use "four grounded assets" and cite Section 3.2, not this
> paragraph.

---

## [APPENDIX] — Full transformation reference table

For a reproducibility appendix table (one row per category, matching the six perception categories
already named in the paper):

| Category | Transform (function) | Mechanism | Key parameters |
|---|---|---|---|
| `gauge_degradation` | `apply_gauge_degradation` | Alpha-composited colour overlay (fog/soot/salt/dust/oil film, chosen by keyword match on the seed row's description) + Gaussian blur; soot variant additionally adds pixel-level Gaussian noise | Overlay alpha 110–210/255 depending on variant; blur radius 1.2–2.5 px; noise σ = `rng.gauss(0,12)` scaling a per-pixel `np.random.randn` draw (unseeded — see reproducibility note above) |
| `occlusion` | `apply_occlusion` | Opaque rectangle over a randomly chosen left/right/top third of the frame, blurred at the edge | Occluder fill (60,55,50); blur radius 1.0 px; side chosen uniformly from {left, right, top} |
| `scale_interpretation` | `apply_scale_interpretation` | Downscale (Lanczos) then re-composite onto a flat grey panel, simulating a distant standoff view | Resize factor `rng.uniform(0.18, 0.35)`; panel colour (130,132,128) |
| `glare_lighting` | `apply_glare` | Radially weighted blend of the image with a heavily blurred copy, strongest at frame centre | Blur radius 4.0 px; centre-weighted mask, blend intensity ×1.15 in the glare zone |
| `iot_contradiction` | `apply_iot_overlay` | Rendered text banner (IoT value, gauge value, contradiction flag) composited into the top-right corner | Fixed banner geometry (≤220×60 px box); IoT/gauge values taken directly from the seed row |
| `never_read` | `apply_never_read` | No visual transform — the base image is returned unchanged; the "never-read" condition is a systemic access failure, not a visual perception failure | — |
| (all categories) | `apply_lighting_variant` | Three post-hoc lighting variants per scene: under-lit, standard, over-lit | brightness/contrast = (0.55, 0.85) / (1.00, 1.00) / (1.55, 1.20) |

---

## Repo action still required before any of the counts above can be frozen

Per `section3_reproducibility_fixes.md` R1: commit `src/perception/` (currently untracked), seed
`numpy.random` explicitly inside `apply_gauge_degradation`, point `OUTPUT_ROOT` at a real, versioned
location, run the pipeline once end-to-end, and publish a frozen manifest (generator version, seed,
per-image SHA256, total counts by category and lighting variant). Only then should the paper state a
specific total-scene count in place of `[FILL AFTER FREEZE]` above.
