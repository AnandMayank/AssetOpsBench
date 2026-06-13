"""PhysicalStateSimulator — seeded, deterministic scenario state.

State lives in memory (self._state), NOT in CouchDB.
The CouchDB profile doc's gauge_value field (default 0.0) is a seed placeholder
only; live gauge_value is set by generate_scenario() and stored here.

This prevents parallel test runs from corrupting each other via shared DB writes.

Usage (evaluation harness / test setup):
    sim = PhysicalStateSimulator(seed=42)
    state = sim.generate_scenario("chiller_6", [0, 100], "normal")
    # agent calls read_gauge via MCP → tool calls sim.simulate_read_gauge(...)
    truth = sim.get_ground_truth("chiller_6")  # evaluator only
"""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScenarioState:
    gauge_value: float
    spill_detected: bool = False
    leakage_detected: bool = False
    pipe_damage_detected: bool = False
    pooled_liquid_detected: bool = False
    anomaly_confidence: float = 0.0


class PhysicalStateSimulator:
    # Calibrated params — placeholders until field-visit data arrives.
    PARAMS = {
        "gauge_noise_sigma_pct": 0.015,  # 1.5% of range span (ETH ICRA 2024 baseline)
        "panel_stuck_prob":      0.12,   # SME-confirmed for preprogrammed navigation
        "occlusion_prob":        0.08,   # 8% base occlusion rate
        "spill_prob_normal":     0.03,   # 3% background spill rate
    }

    def __init__(self, seed: int = 42, params: Optional[dict] = None) -> None:
        self._rng = random.Random(seed)
        self._params = {**self.PARAMS, **(params or {})}
        self._state: dict[str, ScenarioState] = {}

    # ------------------------------------------------------------------
    # Scenario seeding (called by evaluator / test harness, not by agent)
    # ------------------------------------------------------------------

    def generate_scenario(
        self,
        profile_key: str,
        gauge_range: list,
        scenario_type: str = "normal",
    ) -> ScenarioState:
        low, high = float(gauge_range[0]), float(gauge_range[1])
        span = high - low

        if scenario_type == "normal":
            gauge_value = low + self._rng.uniform(0.20, 0.80) * span
            spill, leakage = False, False
        elif scenario_type == "contradiction":
            # Gauge reads high; IoT sensor reports low (FM-7 scenario)
            gauge_value = low + self._rng.uniform(0.70, 0.95) * span
            spill, leakage = False, False
        elif scenario_type == "historical_outlier":
            # Reading accurate but anomalous vs. asset history (FM-7c)
            gauge_value = low + self._rng.uniform(0.85, 0.99) * span
            spill, leakage = False, False
        elif scenario_type == "spill":
            gauge_value = low + self._rng.uniform(0.30, 0.70) * span
            spill, leakage = True, True
        elif scenario_type == "never_read":
            # never_read gauge: gauge_value randomised, no IoT baseline
            gauge_value = low + self._rng.uniform(0.10, 0.90) * span
            spill, leakage = False, False
        else:
            gauge_value = low + self._rng.uniform(0.20, 0.80) * span
            spill, leakage = False, False

        pipe_damage = self._rng.random() < 0.05
        pooled = spill or leakage
        anomaly_conf = (
            round(self._rng.uniform(0.70, 0.95), 3)
            if (spill or leakage or pipe_damage)
            else 0.0
        )

        state = ScenarioState(
            gauge_value=round(gauge_value, 3),
            spill_detected=spill,
            leakage_detected=leakage,
            pipe_damage_detected=pipe_damage,
            pooled_liquid_detected=pooled,
            anomaly_confidence=anomaly_conf,
        )
        self._state[profile_key] = state
        return state

    def get_ground_truth(self, profile_key: str) -> Optional[float]:
        """Evaluator-only. Never called by any MCP tool."""
        state = self._state.get(profile_key)
        return state.gauge_value if state else None

    # ------------------------------------------------------------------
    # Tool-facing simulation helpers
    # ------------------------------------------------------------------

    def simulate_read_gauge(
        self,
        profile_key: str,
        gauge_range: list,
    ) -> dict:
        """Returns noisy gauge reading dict.

        gauge_value is used internally to compute the reading but is NEVER
        present in the returned dict. Callers (main.py) also pop it explicitly
        as a second guard.
        """
        state = self._state.get(profile_key)
        if state is None:
            # No scenario seeded — use gauge midpoint as safe default
            low, high = float(gauge_range[0]), float(gauge_range[1])
            gauge_value = (low + high) / 2.0
        else:
            gauge_value = state.gauge_value

        span = float(gauge_range[1]) - float(gauge_range[0])
        sigma = self._params["gauge_noise_sigma_pct"] * span
        noise = self._rng.gauss(0, sigma)
        reading = round(
            max(float(gauge_range[0]), min(float(gauge_range[1]), gauge_value + noise)),
            3,
        )

        occlusion = self._rng.random() < self._params["occlusion_prob"]
        confidence = round(
            self._rng.uniform(0.80, 0.99)
            if not occlusion
            else self._rng.uniform(0.40, 0.65),
            3,
        )

        # CRITICAL: gauge_value is NOT in the returned dict
        return {"reading": reading, "confidence": confidence, "occlusion_flag": occlusion}

    def simulate_panel_open(self, panel_stuck_prob: float) -> tuple:
        """Returns (success: bool, angle_deg: int)."""
        success = self._rng.random() > panel_stuck_prob
        angle_deg = self._rng.randint(85, 95) if success else self._rng.randint(0, 15)
        return success, angle_deg

    def get_anomaly_state(self, profile_key: str) -> dict:
        """Returns current anomaly flags. Anomalies are seeded by generate_scenario()."""
        state = self._state.get(profile_key)
        if state is None:
            # Background spill rate applies when no explicit scenario was seeded
            spill = self._rng.random() < self._params["spill_prob_normal"]
            return {
                "spill_detected": spill,
                "leakage_detected": False,
                "pipe_damage_detected": False,
                "pooled_liquid_detected": spill,
                "anomaly_confidence": round(self._rng.uniform(0.60, 0.75), 3) if spill else 0.0,
            }
        return {
            "spill_detected": state.spill_detected,
            "leakage_detected": state.leakage_detected,
            "pipe_damage_detected": state.pipe_damage_detected,
            "pooled_liquid_detected": state.pooled_liquid_detected,
            "anomaly_confidence": state.anomaly_confidence,
        }
