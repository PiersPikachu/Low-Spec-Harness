"""calculator - exact arithmetic for the model (safe: parsed, never eval'd)."""
import ast, math, operator

NAME = "calculator"
DESCRIPTION = "Evaluate an arithmetic expression exactly, e.g. '17*23', '(2**10 - 1) / 3', 'sqrt(2) * pi'. Supports + - * / // % **, parentheses, and math functions (sqrt, sin, cos, tan, log, log10, exp, floor, ceil, factorial) and constants pi, e."
PARAMETERS = {"type": "object", "properties": {"expression": {"type": "string", "description": "the expression to evaluate"}}, "required": ["expression"]}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUN = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "log", "log10", "exp", "floor", "ceil", "factorial", "fabs")}
_FUN.update(abs=abs, round=round)
_CONST = {"pi": math.pi, "e": math.e}


def _ev(n):
    if isinstance(n, ast.Expression): return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 10000: raise ValueError("exponent too large")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.Name) and n.id in _CONST: return _CONST[n.id]
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords:
        return _FUN[n.func.id](*[_ev(x) for x in n.args])
    raise ValueError(f"not allowed: {ast.dump(n)[:60]}")


def run(expression: str = "") -> str:
    try:
        return repr(_ev(ast.parse(expression.replace("^", "**"), mode="eval")))
    except Exception as ex:
        return f"error: {ex}"
