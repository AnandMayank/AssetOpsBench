"""paired_stats.py — Paired statistics for matched-scenario comparisons (V3).

Every comparison in the Rev-2 plan is paired: the same scenarios under two
conditions (baseline vs informed prompt, deployment vs benchmark framing, full
vs modality-ablated input). Paired designs are what make small-N benchmark
claims defensible — an unpaired test on 54 scenarios wastes most of the signal,
because scenario difficulty varies far more than the condition effect does.

The plan's rule: **no claim from raw percentages alone.** Report the paired
difference with a confidence interval, and for binary outcomes use McNemar,
which conditions on exactly the discordant pairs that carry the information.

Design notes:

* McNemar uses the **exact binomial** test rather than the χ² approximation.
  With b+c discordant pairs often under 25 here, the asymptotic version is
  anticonservative precisely in the regime we operate in.
* Bootstrap CIs resample **scenarios**, not observations, so the resampling
  unit matches the unit of independence.
* ``paired_bootstrap_ci`` accepts an arbitrary statistic so the same routine
  serves accuracy, MAE, violation rate and Brier.
* ``hierarchical_bootstrap_ci`` resamples families then scenarios within
  families, for the per-category breakdowns where scenarios are nested (the
  A-series convention, after Qin et al. Appendix C).

Everything returns plain dataclasses so results serialise straight into the
report JSON.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class McNemarResult:
    """Exact McNemar test on paired binary outcomes."""

    n_pairs: int
    b: int              # condition A correct, B wrong
    c: int              # condition A wrong, B correct
    n_discordant: int
    p_value: float
    proportion_a: float
    proportion_b: float
    delta: float        # proportion_b - proportion_a
    significant: bool

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"A={self.proportion_a:.3f} B={self.proportion_b:.3f} "
                f"delta={self.delta:+.3f} (b={self.b}, c={self.c}, "
                f"n_disc={self.n_discordant}, p={self.p_value:.4f})")


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    alpha: float
    resamples: int
    n: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def summary(self) -> str:
        pct = int(round((1 - self.alpha) * 100))
        return f"{self.point:+.4f} [{pct}% CI {self.lo:+.4f}, {self.hi:+.4f}] (n={self.n})"


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------

def align(a: Mapping[str, object], b: Mapping[str, object],
          keys: Optional[Sequence[str]] = None) -> Tuple[List[str], List, List]:
    """Restrict two per-scenario result maps to their common, non-null keys.

    Returns ``(scenario_ids, a_values, b_values)`` in a deterministic order.
    Silently dropping unmatched scenarios is the whole point: a paired test on
    a mismatched set is not a paired test.
    """
    shared = sorted(set(a) & set(b)) if keys is None else [k for k in keys if k in a and k in b]
    shared = [k for k in shared if a[k] is not None and b[k] is not None]
    return shared, [a[k] for k in shared], [b[k] for k in shared]


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------

def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly."""
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def _z_quantile(p: float) -> float:
    """Standard-normal inverse CDF (Acklam's rational approximation).

    Kept dependency-free so the statistics layer imports cleanly in the minimal
    environments the ablation scripts run in.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _z_two_sided(alpha: float) -> float:
    return _z_quantile(1 - alpha / 2)


def mcnemar(a: Sequence[bool], b: Sequence[bool],
            alpha: float = DEFAULT_ALPHA) -> McNemarResult:
    """Exact two-sided McNemar test on paired binary outcomes.

    ``a`` and ``b`` are aligned per-scenario success indicators. Only discordant
    pairs inform the test; concordant pairs cancel by construction.
    """
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no paired observations")

    b_count = sum(1 for x, y in zip(a, b) if x and not y)
    c_count = sum(1 for x, y in zip(a, b) if not x and y)
    n_disc = b_count + c_count

    if n_disc == 0:
        p = 1.0
    else:
        k = max(b_count, c_count)
        p = min(1.0, 2 * _binom_sf(k, n_disc))

    pa = sum(1 for x in a if x) / len(a)
    pb = sum(1 for x in b if x) / len(b)
    return McNemarResult(
        n_pairs=len(a), b=b_count, c=c_count, n_discordant=n_disc,
        p_value=p, proportion_a=pa, proportion_b=pb, delta=pb - pa,
        significant=p < alpha,
    )


def holm_correct(p_values: Mapping[str, float],
                 alpha: float = DEFAULT_ALPHA) -> Dict[str, Dict]:
    """Holm-Bonferroni step-down correction.

    Used for the three pairwise framing contrasts in E3; controls family-wise
    error without Bonferroni's power loss.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict] = {}
    prev = 0.0
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        adjusted = max(prev, min(1.0, p * (m - i)))
        prev = adjusted
        out[name] = {"p_raw": p, "p_adjusted": adjusted,
                     "threshold": threshold, "significant": adjusted < alpha}
    return out


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float],
                        statistic: Callable[[Sequence[float]], float] = None,
                        resamples: int = DEFAULT_RESAMPLES,
                        alpha: float = DEFAULT_ALPHA,
                        seed: int = 20260811) -> BootstrapCI:
    """Percentile bootstrap CI for ``statistic(b) - statistic(a)``.

    Resamples **scenario indices**, keeping each scenario's pair intact, so the
    CI reflects between-scenario variability rather than treating the two
    conditions as independent samples.
    """
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no paired observations")
    stat = statistic or (lambda xs: sum(xs) / len(xs))

    point = stat(b) - stat(a)
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(stat([b[i] for i in idx]) - stat([a[i] for i in idx]))
    deltas.sort()
    lo = deltas[int((alpha / 2) * resamples)]
    hi = deltas[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return BootstrapCI(point=point, lo=lo, hi=hi, alpha=alpha,
                       resamples=resamples, n=n)


def hierarchical_bootstrap_ci(a: Mapping[str, float], b: Mapping[str, float],
                              family_of: Mapping[str, str],
                              statistic: Callable[[Sequence[float]], float] = None,
                              resamples: int = DEFAULT_RESAMPLES,
                              alpha: float = DEFAULT_ALPHA,
                              seed: int = 20260811) -> BootstrapCI:
    """Two-level bootstrap: resample families, then scenarios within families.

    For per-category reporting, where scenarios within a failure family are more
    alike than across families. A flat bootstrap understates the CI there.
    """
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    shared, av, bv = align(a, b)
    if not shared:
        raise ValueError("no paired observations")

    by_family: Dict[str, List[int]] = defaultdict(list)
    for i, sid in enumerate(shared):
        by_family[family_of.get(sid, "_")].append(i)
    families = sorted(by_family)

    point = stat(bv) - stat(av)
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        idx: List[int] = []
        for _ in range(len(families)):
            fam = families[rng.randrange(len(families))]
            members = by_family[fam]
            idx += [members[rng.randrange(len(members))] for _ in range(len(members))]
        if not idx:
            continue
        deltas.append(stat([bv[i] for i in idx]) - stat([av[i] for i in idx]))
    deltas.sort()
    lo = deltas[int((alpha / 2) * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int((1 - alpha / 2) * len(deltas)))]
    return BootstrapCI(point=point, lo=lo, hi=hi, alpha=alpha,
                       resamples=len(deltas), n=len(shared))


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------

def min_pairs_for_effect(effect_pp: float, discordance: float = 0.3,
                         alpha: float = DEFAULT_ALPHA,
                         power: float = 0.8) -> int:
    """Paired-design N to detect an ``effect_pp`` percentage-point shift.

    Connor (1987) normal approximation for McNemar::

        n = [z_{alpha/2}*sqrt(pi_d) + z_beta*sqrt(pi_d - delta^2)]^2 / delta^2

    where ``pi_d`` is the discordance rate (fraction of pairs on which the two
    conditions disagree at all) and ``delta`` the marginal difference.

    **This corrects an over-optimistic figure in the Rev-2 plan.** That document
    asserted ">=100 pairs for 10 pp, >=60 for 20 pp" without stating a
    discordance assumption. Those numbers only hold when conditions largely
    agree (pi_d ~ 0.13). At a moderate pi_d = 0.3 the requirement is ~233 pairs
    for 10 pp — more than the 184 paired scenarios E3 plans to run. E3 is
    therefore powered for a ~12 pp effect at that discordance, not 10 pp; the
    realised discordance should be read off the McNemar ``n_discordant`` and the
    achievable resolution recomputed before any null is called.

    Indicative for planning only; always report the realised CI.
    """
    if effect_pp <= 0:
        raise ValueError("effect_pp must be positive")
    if not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0, 1]")
    delta = effect_pp / 100.0
    if delta ** 2 >= discordance:
        raise ValueError(
            f"effect ({delta}) cannot exceed sqrt(discordance) ({discordance**0.5:.3f}): "
            "the marginal difference is bounded by the discordance rate"
        )
    z_a = _z_two_sided(alpha)
    z_b = _z_quantile(power)
    n = ((z_a * math.sqrt(discordance)
          + z_b * math.sqrt(discordance - delta ** 2)) ** 2) / (delta ** 2)
    return max(1, math.ceil(n))


def detectable_effect_pp(n_pairs: int, discordance: float = 0.3,
                         alpha: float = DEFAULT_ALPHA,
                         power: float = 0.8) -> float:
    """Smallest effect (percentage points) detectable with ``n_pairs``.

    The inverse of :func:`min_pairs_for_effect`, for reporting what a given
    scenario budget can actually resolve — the number to quote when declaring
    a null result.
    """
    lo, hi = 1e-4, math.sqrt(discordance) * 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if min_pairs_for_effect(mid * 100, discordance, alpha, power) > n_pairs:
            lo = mid
        else:
            hi = mid
    # Round *up* so the reported resolution is achievable: rounding down would
    # claim an effect size the budget cannot actually detect.
    eff = math.ceil(hi * 100 * 100) / 100
    while min_pairs_for_effect(eff, discordance, alpha, power) > n_pairs:
        eff = round(eff + 0.01, 2)
    return eff


def describe(name: str, a: Sequence[bool], b: Sequence[bool],
             alpha: float = DEFAULT_ALPHA) -> Dict:
    """Convenience: McNemar plus a bootstrap CI on the same paired binaries."""
    mc = mcnemar(a, b, alpha=alpha)
    ci = paired_bootstrap_ci([float(x) for x in a], [float(x) for x in b], alpha=alpha)
    return {
        "comparison": name,
        "mcnemar": mc.to_dict(),
        "delta_ci": ci.to_dict(),
        "summary": f"{name}: {mc.summary()}  delta {ci.summary()}",
    }
