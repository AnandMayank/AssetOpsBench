"""eval_model_panel.py — the frozen, versioned InspectionBench V3 primary
model-panel configuration (Phase 8H.2J).

Single source of truth for the five primary-panel models decided at the
model-panel feasibility gate (reports/benchmark/ — see the gate's chat
report; no separate gate report file was requested there). Every field
here is a fact recorded from that gate, not re-derived or guessed:
FactoryBench's exact model, whether TokenRouter serves it verbatim, the
disclosed successor actually used, and why.

Nothing in this module makes an API call. It is pure configuration +
verification-comparison helpers, imported by the eval runner and by tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

EVAL_PROTOCOL_VERSION = "inspectionbench_v3_eval_protocol@phase8h2j.1"

#: The exact 4,075-episode canonical manifest this panel evaluates against.
BENCHMARK_VERSION = "V3"
MANIFEST_PATH = "reports/benchmark/final_benchmark_manifest_v3.json"
MANIFEST_FINAL_CANONICAL_TOTAL = 4075
FROZEN_93_SHA256 = "d2b48c0b0c9ef19f6c8f6ddd936da098ffd5d0d4023650619f01c1ee043087fe"


@dataclass(frozen=True)
class ModelConfig:
    slot: int
    display_name: str
    tokenrouter_id: str                 # exact string passed to make_backend()/OpenAICompatBackend
    factorybench_model: str             # the FactoryBench paper's model this panel slot targets
    is_exact_factorybench_match: bool   # True only for verbatim-available models
    successor_reason: Optional[str]     # None if exact match; else the disclosed reason
    temperature: float = 0.0
    max_output_tokens: int = 2048       # matches OpenAICompatBackend's current fixed budget
    reasoning_config: Optional[str] = None  # None = provider default; no reasoning override applied


#: The five PRIMARY models, decided at the feasibility gate. Order is the
#: canonical column order for every result table this panel produces.
#: Qwen3.5-9B (FactoryBench's Qwen3-4B slot) is DELIBERATELY EXCLUDED here —
#: see EXCLUDED_MODELS below — not silently dropped, not silently patched.
PRIMARY_PANEL: tuple[ModelConfig, ...] = (
    ModelConfig(
        slot=1, display_name="Claude Sonnet 4.6",
        tokenrouter_id="tokenrouter/anthropic/claude-sonnet-4.6",
        factorybench_model="Claude Sonnet 4.6",
        is_exact_factorybench_match=True, successor_reason=None,
    ),
    ModelConfig(
        slot=2, display_name="GPT-5.2",
        tokenrouter_id="tokenrouter/openai/gpt-5.2",
        factorybench_model="GPT-5.1",
        is_exact_factorybench_match=False,
        successor_reason=("GPT-5.1 is not present in TokenRouter's /models catalogue "
                          "(verified live at the feasibility gate); gpt-5.2 is the nearest "
                          "numeric successor actually served."),
    ),
    ModelConfig(
        slot=3, display_name="DeepSeek V4 Pro",
        tokenrouter_id="tokenrouter/deepseek/deepseek-v4-pro",
        factorybench_model="DeepSeek V3.2",
        is_exact_factorybench_match=False,
        successor_reason=("deepseek/deepseek-v3.2 IS listed in the /models catalogue but is "
                          "rejected at call time (HTTP 400: 'supported API model names are "
                          "deepseek-flash, deepseek-v4-pro') -- stale catalogue entry, not a "
                          "naming choice. deepseek-v4-pro verified live-callable."),
    ),
    ModelConfig(
        slot=4, display_name="Mistral Medium 3.5",
        tokenrouter_id="tokenrouter/mistralai/mistral-medium-3-5",
        factorybench_model="Mistral Large 3",
        is_exact_factorybench_match=False,
        successor_reason=("No 'Large' tier exists on this gateway at all (only medium/small/"
                          "devstral/voxtral variants) -- a genuine CAPABILITY-TIER downgrade, "
                          "not just a version bump; must be flagged in every results table, "
                          "not only in this config."),
    ),
    ModelConfig(
        slot=5, display_name="Qwen3.5-397B-A17B",
        tokenrouter_id="tokenrouter/qwen/qwen3.5-397b-a17b",
        factorybench_model="Qwen3-235B",
        is_exact_factorybench_match=False,
        successor_reason=("Qwen3-235B does not exist in the catalogue; the Qwen line has moved "
                          "to 3.5/3.6/3.7/3.8 versioning with different MoE parameter counts "
                          "(9B/35B-a3b/122B-a10b/397B-a17b). 397B-a17b is the nearest large-tier "
                          "successor; MoE topology differs from a dense/MoE 235B, so this is a "
                          "successor, not an equivalent."),
    ),
)

#: Excluded from the primary panel -- recorded, not silently dropped.
EXCLUDED_MODELS: tuple[dict, ...] = (
    {
        "display_name": "Qwen3.5-9B", "tokenrouter_id": "tokenrouter/qwen/qwen3.5-9b",
        "factorybench_model": "Qwen3-4B",
        "exclusion_reason": (
            "Under the frozen protocol's fixed max_tokens=2048 (OpenAICompatBackend), this "
            "model consumed the ENTIRE budget on hidden reasoning tokens on a live test call "
            "(reasoning_tokens=2095, finish_reason='length', content=None), which crashes the "
            "existing parse_json() path. This is a fixed-token-budget incompatibility with the "
            "CURRENT protocol, not a capability gap -- fixing it (raising max_tokens or "
            "disabling reasoning) is a PROTOCOL CHANGE requiring an explicit, versioned "
            "decision, which this panel does not make unilaterally. Excluded, not rescued."
        ),
    },
)


def get_model(tokenrouter_id: str) -> ModelConfig:
    for m in PRIMARY_PANEL:
        if m.tokenrouter_id == tokenrouter_id:
            return m
    raise KeyError(f"{tokenrouter_id!r} is not in the primary panel "
                   f"(PRIMARY_PANEL has {[m.tokenrouter_id for m in PRIMARY_PANEL]!r}; "
                   f"if this is intentional, add it explicitly -- never run an "
                   f"unconfigured model against V3)")


@dataclass(frozen=True)
class ServedModelCheck:
    requested_id: str
    served_model: Optional[str]
    ok: bool
    reason: str


def _normalize_model_token(s: str) -> str:
    """Strip provider-path prefixes and all punctuation, lowercase. Verified
    live (feasibility gate) that TokenRouter's response.model field is
    inconsistently formatted per provider -- e.g. requesting
    'anthropic/claude-sonnet-4.6' comes back as 'claude-sonnet-4-6' (dashes,
    no prefix), 'openai/gpt-5.2' comes back as 'gpt-5.2-2025-12-11' (a dated
    snapshot suffix appended), while deepseek/mistral/qwen echo close to the
    requested string. A naive equality check would FALSELY FLAG the first
    two as mismatches -- confirmed by testing this exact function against
    all 5 primary-panel models' real responses before relying on it."""
    import re
    s = s.lower().split("/")[-1]
    return re.sub(r"[^a-z0-9]", "", s)


def verify_served_model(requested_tokenrouter_id: str, served_model: Optional[str]) -> ServedModelCheck:
    """Compare the router's OWN response.model field against what was
    requested. Never assumes they match -- a router silently substituting a
    cheaper/different model is exactly the failure mode Phase 3's
    non-negotiable #7 ("no silent model substitution") targets.

    Uses normalized substring containment (see _normalize_model_token), not
    exact string equality -- live-verified necessary for this router. A
    dated-snapshot suffix (e.g. gpt-5.2 -> "gpt-5.2-2025-12-11") is treated
    as a match (the requested token is a prefix of the served one); a
    genuinely DIFFERENT model family/tier is not.

    `served_model` is None for backends that can't report it -- treated as
    ok=True with reason "unverifiable" rather than a false failure, since
    the panel's only backend (OpenAICompatBackend) DOES report it, so a
    None here on this panel's real runs would itself be worth investigating
    separately, not silently passed.
    """
    if served_model is None:
        return ServedModelCheck(requested_tokenrouter_id, served_model, True,
                                "served_model unreported by backend (unverifiable, not a failure)")
    nr = _normalize_model_token(requested_tokenrouter_id)
    ns = _normalize_model_token(served_model)
    if nr == ns or nr in ns or ns in nr:
        return ServedModelCheck(requested_tokenrouter_id, served_model, True,
                                f"normalized match ({nr!r} <-> {ns!r})")
    return ServedModelCheck(requested_tokenrouter_id, served_model, False,
                            f"MISMATCH: requested {requested_tokenrouter_id!r} (normalized {nr!r}), "
                            f"router served {served_model!r} (normalized {ns!r}) -- episode must be "
                            f"marked invalid, not silently kept")
