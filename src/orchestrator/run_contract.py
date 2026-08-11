"""run_contract.py — One declaration of the evaluation contract (P0-5).

The plan requires that models be compared under identical task/tool contracts,
and that a model which cannot access the same modalities or tools be reported in
a separate track. Three runners currently make that unenforceable by disagreeing
silently:

===========================  ===========  ===========  ==================
runner                       temperature  max_tokens   channel
===========================  ===========  ===========  ==================
run_pmc_benchmark.py         0.4 / 1.0 / 0.2 (per provider)  4096  vision
run_frm_probe_eval.py        1            1024         text + vision
run_fm7_fm15_probe_eval.py   1            1024         text
===========================  ===========  ===========  ==================

Nothing detects that, and none of it reaches the result JSON, so two "identical"
runs can differ in decoding budget by 4x. This module makes the contract a
value: constructed once, hashed, asserted comparable across runs, and written
into every result file.

``assert_comparable`` is the load-bearing part. It reports *which* fields differ
rather than a bare boolean, so a genuine track separation ("this model cannot
take images") is distinguishable from an accidental drift ("someone changed
max_tokens"), which is exactly the distinction the plan asks for.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Fields that must match for two runs to be compared on the same leaderboard.
#: ``model`` is deliberately absent — comparing models is the point.
COMPARABLE_FIELDS = (
    "task", "temperature", "max_tokens", "system_prompt_id", "prompt_variant",
    "framing", "channels", "max_observations", "max_actions",
    "image_max_width", "tool_surface",
)

#: Fields that legitimately differ between tracks rather than indicating drift.
TRACK_FIELDS = ("channels", "tool_surface")

#: Evidence channels an agent may be given. V1 ablates these.
ALL_CHANNELS = ("rgb", "iot", "enterprise", "metadata")


class ContractViolation(AssertionError):
    """Raised when two runs are compared that were not run under one contract."""


@dataclass(frozen=True)
class RunContract:
    """Everything that must be frozen for a comparison to mean anything."""

    task: str                              # e.g. "pmc_l1", "frm_probe", "fm7_fm15"
    model: str                             # provider/model id, exact snapshot
    api: str = ""                          # tokenrouter | genai | openai | ollama | mock
    temperature: float = 0.0
    max_tokens: int = 4096                 # 512 truncated thinking models into empty output
    system_prompt_id: str = ""             # stable id, not the prose
    prompt_variant: str = "baseline"       # A13/E2: baseline | informed
    framing: str = "neutral"               # E3: neutral | deployment | benchmark | safety_audit
    channels: Tuple[str, ...] = ALL_CHANNELS
    max_observations: int = 3              # min_reads / observation budget
    max_actions: int = 25
    image_max_width: int = 800
    tool_surface: str = ""                 # id of the tool set offered
    seed: str = "pmc-eval"
    notes: str = ""

    # ---------------------------------------------------------------- identity

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    def fingerprint(self) -> str:
        """Hash of the comparable fields only.

        Two runs with the same fingerprint are on the same leaderboard;
        differing ``model`` or ``seed`` does not change it.
        """
        payload = {k: self.to_dict()[k] for k in COMPARABLE_FIELDS}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    def stamp(self) -> Dict[str, Any]:
        """Provenance block for a result file."""
        return {"run_contract": self.to_dict(), "contract_fingerprint": self.fingerprint()}

    # ------------------------------------------------------------ comparison

    def differences(self, other: "RunContract") -> Dict[str, Tuple[Any, Any]]:
        a, b = self.to_dict(), other.to_dict()
        return {k: (a[k], b[k]) for k in COMPARABLE_FIELDS if a[k] != b[k]}

    def assert_comparable(self, other: "RunContract") -> None:
        """Raise unless two runs may be placed on the same leaderboard.

        The message separates *track* differences (different modality or tool
        access — a legitimate reason to report separately) from *drift*
        (someone changed a decoding budget), because the remedies differ.
        """
        diffs = self.differences(other)
        if not diffs:
            return
        track = {k: v for k, v in diffs.items() if k in TRACK_FIELDS}
        drift = {k: v for k, v in diffs.items() if k not in TRACK_FIELDS}
        lines: List[str] = [
            f"{self.model} and {other.model} were not run under one contract:"
        ]
        for k, (x, y) in sorted(drift.items()):
            lines.append(f"  DRIFT {k}: {x!r} vs {y!r}")
        for k, (x, y) in sorted(track.items()):
            lines.append(f"  TRACK {k}: {x!r} vs {y!r}")
        if track and not drift:
            lines.append("  -> different capability track; report separately, "
                         "do not rank against each other")
        else:
            lines.append("  -> configuration drift; re-run under a single contract")
        raise ContractViolation("\n".join(lines))

    def same_track(self, other: "RunContract") -> bool:
        return all(self.to_dict()[k] == other.to_dict()[k] for k in TRACK_FIELDS)

    # -------------------------------------------------------------- variants

    def with_framing(self, framing: str) -> "RunContract":
        """E3: identical in every respect but the framing paragraph."""
        return replace(self, framing=framing)

    def with_channels(self, *channels: str) -> "RunContract":
        """V1: modality ablation."""
        unknown = set(channels) - set(ALL_CHANNELS)
        if unknown:
            raise ValueError(f"unknown channels: {sorted(unknown)}")
        return replace(self, channels=tuple(channels))

    def with_prompt_variant(self, variant: str) -> "RunContract":
        """E2/A13: baseline vs informed operational rules."""
        if variant not in ("baseline", "informed"):
            raise ValueError(f"unknown prompt_variant: {variant!r}")
        return replace(self, prompt_variant=variant)


def assert_all_comparable(contracts: Sequence[RunContract]) -> None:
    """Every contract in a leaderboard must agree with the first."""
    if len(contracts) < 2:
        return
    for c in contracts[1:]:
        contracts[0].assert_comparable(c)


def group_by_track(contracts: Sequence[RunContract]) -> Dict[str, List[RunContract]]:
    """Partition into comparable tracks so mixed model sets report honestly."""
    tracks: Dict[str, List[RunContract]] = {}
    for c in contracts:
        key = "|".join(f"{k}={c.to_dict()[k]}" for k in TRACK_FIELDS)
        tracks.setdefault(key, []).append(c)
    return tracks
