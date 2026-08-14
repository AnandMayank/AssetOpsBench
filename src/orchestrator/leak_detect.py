"""leak_detect.py — Conditional decision-rule leak detector (ledger B3).

``classc_audit.py`` / ``classd_audit.py`` carried a detector that matched only
the literal phrasing ``"the answer is"`` / ``"you should commit"`` and had
fired on 0 of 32 audited class-B/C/D scenarios. The dominant leak form in this
corpus is a **conditional decision rule** stated directly in the prompt —
"if <condition>, <action>" — which that pattern never matched at all. Measured
extent, scanning every family-B/C/D question text: 15 of 32.

A rule leak means CC may be measuring instruction-following rather than
evidence-grounded reasoning. Per the reviewed decision, this is treated as a
**measured factor** — de-leaked twins exist for all 15 (``kind=de_leaked`` in
each twin's manifest) rather than as an automatic DEFECT verdict: a leaking
scenario is still runnable, its leak status is just reported and controllable.
"""
from __future__ import annotations

import re
from typing import Pattern

#: "if <condition>, <escalate/abort/commit/defer/flag/do-not-X>" within 120
#: characters of the "if". Matches the corpus's dominant leak form, e.g.
#: R014: "If gauge_path is null ... escalate -- do not fabricate a reading".
RULE_LEAK: Pattern[str] = re.compile(
    r"\bif\b[^.;]{0,120}?\b(escalate|abort|commit|defer|flag|do not|don't)\b", re.I)


def has_rule_leak(question_body: str) -> bool:
    """True if the prompt states a conditional decision rule, not just the
    scenario's context. ``question_body`` should already have the JSON
    response-format instruction stripped (``Return \\{.*`` onward)."""
    return bool(RULE_LEAK.search(question_body))
