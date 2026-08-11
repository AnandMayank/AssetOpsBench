"""frozen_config.py — Pin the evaluation configuration and detect drift (P0-4).

Every threshold below changes scores. A3 measured the calibration gate's τ sweep
and found a *sharp cliff* rather than a smooth trade-off (V=0.00 through
τ=0.3–0.6, then V=1.00 by τ=0.7), so an unrecorded threshold edit silently
rewrites the leaderboard. The pinned values live in ``config/frozen_config.json``
and their SHA-256 is stamped into every episode trace, making any run
attributable to an exact configuration.

``verify()`` compares the live module constants against the frozen file and
reports drift; the CI test in ``tests/test_frozen_config.py`` fails on any
mismatch, so changing a threshold requires deliberately re-freezing.

**Determinism caveat — recorded, not papered over.** The Rev-2 plan assumed
``temperature=0`` throughout. The code does not do that: sampling temperatures
are 0.4 (Gemini / TokenRouter), 1.0 (Robotics-ER, which the SDK does not let us
pin lower for the agentic-vision path) and 0.2 (moondream). Two consequences,
both real:

* only the ``mock`` backend is bit-reproducible; live-backend runs are *not*,
  so paired comparisons need repeated seeds rather than assuming determinism;
* setting these to 0 now would break comparability with the 351 retained
  traces, which back the A11/A12 results.

The honest position is therefore to freeze the temperatures **at their actual
values**, record that runs are stochastic, and treat "move everything to
temperature 0" as a scored migration rather than a silent edit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_PATH = REPO_ROOT / "config" / "frozen_config.json"

#: The pinned evaluation configuration. Keys mirror the live constants; see
#: ``verify()`` for where each is read from.
FROZEN: Dict[str, Any] = {
    "schema": "assetops.frozen_config/1",
    "description": "Pinned evaluation configuration for AssetOpsBench v2 (P0-4).",
    "gate": {
        # real_pmc_orchestrator.py
        "TAU_COMMIT": 0.82,
        "TAU_ESCALATE": 0.65,
        # A3 knee (reports/ablations/a3_tau_sweep.csv); cliff at tau ~= 0.67
        "tau_entropy": 0.3,
        "weights": {"C": 0.35, "A": 0.35, "H": 0.30},
    },
    "reads": {
        "min_reads": 3,
        # A2: self-conditioning gain crosses 1 at N>=3; collapse rate reaches
        # 65% by N=10. Kept at 1 so no prior read is fed back into a later one.
        "temporal_window_N": 1,
    },
    "energy_J": {
        "NAV_J_PER_M": 45.0,
        "READ_J": 4.0,
        "REPOSITION_DISTANCE_M": 3.0,
    },
    "decoding": {
        # Actual values in the code, not aspirational ones. 4096 is the §4.4
        # fix: 512 truncated thinking-style models into empty responses.
        "max_tokens": 4096,
        "temperature": {
            "gemini": 0.4,
            "tokenrouter": 0.4,
            "gemini_er": 1.0,
            "moondream": 0.2,
            "mock": 0.0,
        },
        "deterministic": False,
        "determinism_note": (
            "Only the mock backend is bit-reproducible. Live backends sample; "
            "paired comparisons must use repeated seeds, not assumed determinism."
        ),
    },
    "grading": {
        "commit_tolerance_frac_of_span": 0.05,
        # P0-3: above this fraction of non-answering reads an episode is INVALID
        "apparatus_invalid_threshold": 0.5,
    },
    "dataset": {
        "perception_csv": "RobotInspection/shared/perception/perception.csv",
        "pairs_csv": "src/orchestrator/data/pairs_full.csv",
        "paired_query_scenarios": 751,
        "readable_gt_true": 311,
        "readable_gt_false": 440,
    },
}


def canonical_json(cfg: Dict[str, Any]) -> str:
    """Stable serialisation so the hash depends on values, not key order."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: Dict[str, Any] = None) -> str:
    """SHA-256 of the frozen configuration; stamped into every trace."""
    return hashlib.sha256(canonical_json(cfg or FROZEN).encode()).hexdigest()


def write(path: Path = FROZEN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(FROZEN)
    payload["config_hash"] = config_hash(FROZEN)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load(path: Path = FROZEN_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text())


def stamp() -> Dict[str, str]:
    """Provenance block to embed in an episode trace."""
    return {"frozen_config_hash": config_hash(), "frozen_config_schema": FROZEN["schema"]}


def verify() -> List[str]:
    """Compare live module constants against the frozen values.

    Returns a list of human-readable drift descriptions; empty means no drift.
    Import failures are reported rather than raised so this stays usable in
    environments where optional backends are absent.
    """
    drift: List[str] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            drift.append(f"{name}: live={actual!r} frozen={expected!r}")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import real_pmc_orchestrator as rpo

        check("gate.TAU_COMMIT", rpo.TAU_COMMIT, FROZEN["gate"]["TAU_COMMIT"])
        check("gate.TAU_ESCALATE", rpo.TAU_ESCALATE, FROZEN["gate"]["TAU_ESCALATE"])
        check("energy_J.NAV_J_PER_M", rpo.NAV_J_PER_M, FROZEN["energy_J"]["NAV_J_PER_M"])
        check("energy_J.REPOSITION_DISTANCE_M", rpo.REPOSITION_DISTANCE_M,
              FROZEN["energy_J"]["REPOSITION_DISTANCE_M"])
    except Exception as exc:  # pragma: no cover - environment dependent
        drift.append(f"could not import real_pmc_orchestrator: {exc}")

    try:
        from apparatus import APPARATUS_INVALID_THRESHOLD

        check("grading.apparatus_invalid_threshold", APPARATUS_INVALID_THRESHOLD,
              FROZEN["grading"]["apparatus_invalid_threshold"])
    except Exception as exc:  # pragma: no cover
        drift.append(f"could not import apparatus: {exc}")

    if FROZEN_PATH.exists():
        on_disk = load()
        if on_disk.get("config_hash") != config_hash(FROZEN):
            drift.append(
                f"config/frozen_config.json is stale: "
                f"on-disk hash {str(on_disk.get('config_hash'))[:12]}... != "
                f"module hash {config_hash(FROZEN)[:12]}... (re-run this module)"
            )
    else:
        drift.append(f"{FROZEN_PATH} missing (run: python -m frozen_config)")

    return drift


def main() -> int:
    path = write()
    print(f"wrote {path}")
    print(f"config_hash = {config_hash()}")
    drift = verify()
    if drift:
        print("\nDRIFT DETECTED:")
        for d in drift:
            print("  " + d)
        return 1
    print("no drift: live constants match the frozen configuration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
