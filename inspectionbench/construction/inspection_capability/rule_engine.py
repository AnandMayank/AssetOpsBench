"""Safe expression evaluator for `inspection_capabilities.json`'s
`success_criteria` rule-DSL.

Capabilities declare verdict logic as JSON, not Python, so the JSON is the
single source of truth for thresholds (e.g. `tau_commit=0.82`) — the same
role `RI_Failure_Mode_Taxonomy.md`'s hand-written C/A/H formula plays today,
just made data-driven instead of re-implemented per capability.

Expressions are parsed with `ast.parse(..., mode="eval")` and walked by a
strict allowlist interpreter — never `eval()`. Only arithmetic, boolean,
comparison, ternary (`a if cond else b`), a small numeric function set
(`abs`, `len`, `min`, `max`, `std`, `mean`), and `thresholds.<name>` attribute
lookups are permitted; anything else (attribute access on non-`thresholds`
objects, imports, comprehensions, subscripts, etc.) raises `RuleEngineError`
at evaluation time.
"""
from __future__ import annotations

import ast
import operator
import statistics
from typing import Any

__all__ = ["RuleEngineError", "evaluate_expression", "evaluate_success_criteria"]


class RuleEngineError(Exception):
    """Raised for any expression the rule-DSL does not permit, or that fails
    to evaluate against the supplied inputs (e.g. a missing variable)."""


_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARYOPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}

_COMPAREOPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# `std` is population stdev (matches the FM-7 verifier's `std(readings)`
# usage over a small, complete sample of readings, not an inferential
# sample stdev).
_FUNCS: dict[str, Any] = {
    "abs": abs,
    "len": len,
    "min": min,
    "max": max,
    "std": statistics.pstdev,
    "mean": statistics.mean,
}


def evaluate_expression(expr: str, env: dict[str, Any]) -> Any:
    """Evaluate one rule-DSL expression string against `env` (a flat
    variable-name -> value mapping; `env["thresholds"]` is a dict consulted
    for `thresholds.<name>` attribute lookups)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise RuleEngineError(f"invalid expression {expr!r}: {exc}") from exc
    try:
        return _eval_node(tree.body, env)
    except RuleEngineError:
        raise
    except (ZeroDivisionError, statistics.StatisticsError, TypeError, KeyError, IndexError) as exc:
        raise RuleEngineError(f"error evaluating {expr!r}: {exc}") from exc


def _eval_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)) or node.value is None:
            return node.value
        raise RuleEngineError(f"disallowed constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id not in env:
            raise RuleEngineError(f"unknown variable: {node.id!r}")
        return env[node.id]

    if isinstance(node, ast.Attribute):
        if not (isinstance(node.value, ast.Name) and node.value.id == "thresholds"):
            raise RuleEngineError(
                f"attribute access only allowed on 'thresholds', got '.{node.attr}' on a non-thresholds object"
            )
        thresholds = env.get("thresholds", {})
        if node.attr not in thresholds:
            raise RuleEngineError(f"unknown threshold: {node.attr!r}")
        return thresholds[node.attr]

    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise RuleEngineError(f"disallowed operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left, env), _eval_node(node.right, env))

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise RuleEngineError(f"disallowed unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand, env))

    if isinstance(node, ast.BoolOp):
        # Short-circuit, matching Python's own `and`/`or` semantics — load
        # -bearing for expressions like `gas_available and gas_ppm >= tau`,
        # where `gas_ppm` may not exist in env at all when gas_available is
        # False (e.g. that sensor payload isn't mounted this run). Operands
        # must be evaluated lazily, one at a time, not gathered eagerly.
        if isinstance(node.op, ast.And):
            result: Any = True
            for operand in node.values:
                result = _eval_node(operand, env)
                if not result:
                    return result
            return result
        result = False
        for operand in node.values:
            result = _eval_node(operand, env)
            if result:
                return result
        return result

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            op_fn = _COMPAREOPS.get(type(op))
            if op_fn is None:
                raise RuleEngineError(f"disallowed comparison operator: {type(op).__name__}")
            right = _eval_node(comparator, env)
            result = result and op_fn(left, right)
            left = right
        return result

    if isinstance(node, ast.IfExp):
        return _eval_node(node.body, env) if _eval_node(node.test, env) else _eval_node(node.orelse, env)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            name = node.func.id if isinstance(node.func, ast.Name) else "<expr>"
            raise RuleEngineError(f"disallowed function call: {name!r}")
        if node.keywords:
            raise RuleEngineError("keyword arguments are not permitted in rule-DSL calls")
        args = [_eval_node(a, env) for a in node.args]
        return _FUNCS[node.func.id](*args)

    if isinstance(node, ast.List):
        return [_eval_node(e, env) for e in node.elts]

    raise RuleEngineError(f"disallowed expression element: {type(node).__name__}")


def evaluate_success_criteria(success_criteria: dict[str, Any], inputs: dict[str, Any]) -> str:
    """Evaluate a capability's `success_criteria` block against `inputs`
    (the primitive variables the capability's glue code resolved from
    `ObservationRecord`s, e.g. `n_readings`, `readings`, `iot_value`).

    Returns the matched verdict string. `variables` are evaluated once, in
    declaration order, each visible to later variables and to `rules`
    (mirrors the FM-7 formula's `score = 0.35*C + 0.35*A + 0.30*H` pattern,
    where `score` references `C`/`A`/`H` computed just before it). `rules`
    are evaluated in order; the first whose `if` expression is truthy wins;
    a `{"default": true, "verdict": ...}` entry (if present) always matches
    last. Raises `RuleEngineError` if no rule matches, or if the matched
    verdict is not declared in `verdict_states`.
    """
    thresholds = success_criteria.get("thresholds", {})
    env: dict[str, Any] = dict(inputs)
    env["thresholds"] = thresholds

    for name, expr in success_criteria.get("variables", {}).items():
        env[name] = evaluate_expression(expr, env)

    verdict_states = success_criteria.get("verdict_states", [])
    for rule in success_criteria.get("rules", []):
        if rule.get("default"):
            verdict = rule["verdict"]
        elif evaluate_expression(rule["if"], env):
            verdict = rule["verdict"]
        else:
            continue
        if verdict_states and verdict not in verdict_states:
            raise RuleEngineError(
                f"rule matched verdict {verdict!r} not declared in verdict_states {verdict_states!r}"
            )
        return verdict

    raise RuleEngineError("no rule matched and no default rule present")
