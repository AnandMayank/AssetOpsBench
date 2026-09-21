# Annotation-model discrepancy: the paper says gemini-2.5-flash; the code runs two different pipelines under two different models

**Claim in the paper (Section 3.1, "Model-assisted annotation"):** *"We annotate each burst of
inspection frames with `gemini-2.5-flash`. A request carries up to four evenly spaced frames..."*

**What the code actually contains — two separate, non-interchangeable pipelines:**

## 1. The pipeline that matches the paper's *description* (multi-frame burst) — defaults to GPT, not Gemini

`src/orchestrator/data/label_windows_vlm.py` is the script whose feature set matches the paper's
prose exactly: it samples up to four evenly-spaced frames from a burst, always including the first
and last (`:90-97`), and has a Gemini REST call path (`call_vlm(..., backend="gemini")`, `:173-175`).
**But its actual default backend, when run with no override, is `tokenrouter`**
(`VLM_BACKEND = os.environ.get("VLM_BACKEND", "tokenrouter")`, `:182`), which dispatches to
`tokenrouter_backend.py`'s `DEFAULT_MODEL = "openai/gpt-5.4-mini"` (`tokenrouter_backend.py:21`) —
not Gemini at all. Whoever ran this script's default invocation used GPT-5.4-mini, contradicting the
paper's specific "with gemini-2.5-flash" claim for burst annotation. Its output (`labels_vlm.json`)
does not currently exist on disk and is not read by any canonical manifest or evaluated-episode
artifact — it feeds only a video-rendering byproduct script.

## 2. The pipeline that actually feeds evaluated data — genuinely uses Gemini, but is single-image, not burst

`src/perception/reverse_label_real_images.py` is the pipeline whose *output* (`perception_real.csv`)
is actually read by the real evaluation path (`pmc_dataset.py`, `run_pmc_benchmark.py`,
`grader.py:grade_real_pmc`, `scripts/v7_label_adjudication.py`, `scripts/build_pairs.py`,
`scripts/build_splits.py`). It genuinely calls `gemini-2.5-flash`
(`call_gemini`, `reverse_label_real_images.py:143-147`) — but on **one image at a time**, with a
16-field extraction prompt, not the four-frame burst the paper describes. This is the real prompt
Appendix A should transcribe if it currently reproduces the other script's text:

```
EXTRACT_PROMPT (src/perception/reverse_label_real_images.py:101-133, verbatim):

You are an expert industrial asset inspector analysing a real photograph of an industrial gauge.

Extract EVERY field below directly from what you can observe in the image. Be specific and factual.
If you truly cannot determine a field from the image, give a best-educated guess based on context.

Respond ONLY with this exact JSON structure, no other text:
{
  "category": "<one of: gauge_degradation | occlusion | scale_interpretation | glare_lighting | iot_contradiction | never_read | clean>",
  "asset": "<industrial asset type the gauge is monitoring, e.g. Centrifugal pump, Induction motor, HVAC chiller, Hydraulic system, Compressor, Boiler, etc.>",
  "location": "<industrial environment, e.g. Oil & Gas refinery, Chemical plant, Power plant, Manufacturing floor, Data center, Offshore platform, etc.>",
  "failure_mode": "<physical fault or degradation mode visible, e.g. Cavitation, Bearing wear, Soot fouling, Pipe vibration, Refrigerant leak, etc. — or 'None visible' if clean>",
  "description": "<one detailed sentence describing exactly what visual challenge a robot inspector would face reading this gauge>",
  "gauge_readable": true | false,
  "gauge_value": "<numeric value shown by pointer/needle if readable, else 'UNREADABLE'>",
  "gauge_range_min": "<minimum value on dial scale, e.g. '0'>",
  "gauge_range_max": "<maximum value on dial scale, e.g. '10'>",
  "gauge_unit": "<unit of measurement on dial, e.g. 'bar', 'PSI', 'MPa', 'A', '°C', 'RPM', '%', etc.>",
  "iot_value": "<plausible nominal IoT sensor value for this asset type in normal operation>",
  "image_role": "<'query' if gauge shows degradation/obstruction/challenge, 'reference' if gauge is clean and clear>"
}

Category definitions:
- gauge_degradation: dial face contaminated (soot, oil film, frost, moisture, dust, corrosion)
- occlusion: gauge physically blocked by pipe, cable, structure, or object
- scale_interpretation: gauge too small/far/dim to read graduation marks reliably
- glare_lighting: strong glare, reflections, or thermal shimmer obscuring the dial
- iot_contradiction: two instruments visible showing contradicting values
- never_read: gauge exists but is inaccessible, behind enclosure, or off the inspection route
- clean: gauge is clear, unobstructed, and fully readable (reference candidate)
```

## 3. The model actually doing gauge "detection" in the evaluated PMC track — also GPT-5.4-mini, not Gemini

Separate from annotation entirely: the PMC track's live evaluation runs (`real_pmc_orchestrator.py`)
use the model supplied as `model_name` to answer `read_gauge` calls during real agent trajectories.
The two traces documented in this same folder (`README.md`, Figure 3 in the paper — scenario
AOBv2-REAL-008) were both run with `openai/gpt-5.4-mini` as this perceiver model, confirmed directly
in both stored trajectory files (`AOBv2-REAL-008__pmc_bcb631.json`, `AOBv2-REAL-008__pmc_6a6f42.json`).
So the model that is actually "detecting" gauge readings inside Figure 3's own example is GPT, not
Gemini — worth stating explicitly if the paper's Section 3.1 annotation paragraph is read as implying
Gemini is used throughout the pipeline.

## Recommended fix for the paper

1. **Section 3.1, "Model-assisted annotation":** either (a) correct the model name to match whichever
   script's output actually reaches the released data — Gemini 2.5-flash via
   `reverse_label_real_images.py`, single-image, 16-field prompt (reproduce the prompt above in
   Appendix A) — or (b) if the burst-annotation feature described in the prose is intentional and
   important to keep, disclose that its reference implementation (`label_windows_vlm.py`) defaults to
   GPT-5.4-mini via TokenRouter and that its output does not currently feed any canonical or evaluated
   artifact.
2. **Figure 3's caption:** name the model explicitly — "gpt-5.4-mini, acting as the gauge perceiver" —
   rather than leaving the reader to assume it is the same `gemini-2.5-flash` referenced in 3.1.
3. **Do not present this as one unified annotation pipeline.** There are three distinct model/script
   pairings in the current codebase (burst/GPT-default, single-image/Gemini, PMC-perceiver/GPT), each
   serving a different, non-overlapping purpose. The paper should name each one where it is actually
   used, rather than one blanket "gemini-2.5-flash" sentence covering all of them.

## How to list this alongside AOBv2-REAL-018 in the appendix

Both cases share the same lesson — a labeling/annotation process surfacing a real limitation rather
than being pure noise — so they read well as a matched pair:

- **AOBv2-REAL-018** (already pushed, `reports/paper_audit/figure_assets/`): a dual-labeling
  *disagreement* that turns out to reveal a missing schema field (`instrument_status`, for a
  decommissioned gauge) — an annotation-schema gap.
- **AOBv2-REAL-008** (this folder): the model actually performing gauge detection/annotation in a
  *worked, evaluated example* is different from the model the construction section names — a
  model-attribution gap.

Suggested appendix framing: *"Table X / Figure X: two annotation-pipeline limitations surfaced during
construction. (left) AOBv2-REAL-018 — a dual-label disagreement traced to a missing schema field for
decommissioned instruments. (right) AOBv2-REAL-008 — the evaluated PMC trajectory in Figure 3 was
perceived by gpt-5.4-mini, not the gemini-2.5-flash named in Section 3.1's annotation paragraph; we
disclose this explicitly rather than let the two be conflated."*
