"""AST guard for model-written sandbox code.

A usability layer, not the security boundary: it turns the obvious wrong moves
(shelling out, networking, reflection, reading arbitrary files) into a clear
error with a hint about what IS allowed, before the code reaches the container.
The container (no network, no capabilities, read-only root, no data mount) is
what actually contains code that slips past this check.
"""

import ast

from analyst.errors import InvalidRequestError

ALLOWED_IMPORTS: tuple[str, ...] = (
    "duckdb", "pandas", "polars", "numpy", "scipy", "statsmodels", "statsforecast",
    "plotly", "math", "statistics", "datetime", "json", "re", "collections",
    "itertools", "functools",
)
_ALLOWED = frozenset(ALLOWED_IMPORTS)

_HELPERS = (
    "Read data with load('<handle id>') (returns a pandas DataFrame) and keep results "
    "with save(df, 'name'); files may only be opened under /artifacts/ "
    "(read: /artifacts/in or /artifacts/out, write: /artifacts/out)."
)
_IMPORT_HINT = f"Allowed imports: {', '.join(ALLOWED_IMPORTS)}. {_HELPERS}"

_BLOCKED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "globals", "locals", "vars",
    "breakpoint", "input", "exit", "quit", "help", "memoryview",
})
# getattr-family builtins are allowed only as direct calls with a literal, non-dunder name.
_ATTR_FUNCS = frozenset({"getattr", "setattr", "delattr", "hasattr"})
# Frame/generator internals that expose globals or builtins without a dunder.
_BLOCKED_ATTRS = frozenset({
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code",
    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "tb_frame",
})
_WRITE_MODE_CHARS = frozenset("wax+")


def _reject(message: str, hint: str = _HELPERS, node: ast.AST | None = None) -> None:
    line = getattr(node, "lineno", None)
    where = f" (line {line})" if line else ""
    raise InvalidRequestError(f"{message}{where}", hint=hint)


def _dotted(node: ast.AST) -> str | None:
    """'a.b.c' for a Name/Attribute chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _str_const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _Guard(ast.NodeVisitor):
    def __init__(self) -> None:
        # Name nodes already validated as the callee of an allowed call.
        self._ok_callees: set[int] = set()

    # -- imports ---------------------------------------------------------------
    def _check_module(self, module: str, node: ast.AST) -> None:
        top = module.split(".", 1)[0]
        if top not in _ALLOWED:
            _reject(f"import of '{module}' is not allowed in the sandbox", _IMPORT_HINT, node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            _reject("relative imports are not allowed in the sandbox", _IMPORT_HINT, node)
        self._check_module(node.module or "", node)
        for alias in node.names:
            if _is_dunder(alias.name):
                _reject(f"importing '{alias.name}' (dunder name) is not allowed", _IMPORT_HINT, node)

    # -- names and attributes ----------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if id(node) in self._ok_callees:
            return
        name = node.id
        if name in _BLOCKED_NAMES:
            _reject(f"'{name}' is not allowed in the sandbox", node=node)
        if name in _ATTR_FUNCS:
            _reject(f"'{name}' may only be called with a literal, non-dunder attribute name",
                    node=node)
        if name == "open":
            _reject("'open' may only be called directly with a literal /artifacts/ path",
                    node=node)
        if _is_dunder(name):
            _reject(f"dunder name '{name}' is not allowed in the sandbox", node=node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if _is_dunder(attr):
            _reject(f"dunder attribute access '.{attr}' is not allowed in the sandbox", node=node)
        if attr in _BLOCKED_ATTRS:
            _reject(f"attribute '.{attr}' (interpreter internals) is not allowed", node=node)
        base = _dotted(node.value)
        last = base.rsplit(".", 1)[-1] if base else None
        if attr == "modules" and last == "sys":
            _reject("sys.modules is not allowed in the sandbox", node=node)
        if attr == "system" and last == "os":
            _reject("os.system is not allowed in the sandbox", node=node)
        self.generic_visit(node)

    # -- calls -------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _ATTR_FUNCS:
                self._check_attr_call(node)
                self._ok_callees.add(id(func))
            elif func.id == "open":
                self._check_open(node)
                self._ok_callees.add(id(func))
        self.generic_visit(node)

    def _check_attr_call(self, node: ast.Call) -> None:
        fn = node.func.id  # type: ignore[attr-defined]
        name = _str_const(node.args[1]) if len(node.args) >= 2 else None
        if name is None or _is_dunder(name) or name in _BLOCKED_ATTRS:
            _reject(f"{fn}() needs a literal, non-dunder attribute name", node=node)

    def _check_open(self, node: ast.Call) -> None:
        path = _str_const(node.args[0]) if node.args else None
        if path is None:
            for kw in node.keywords:
                if kw.arg == "file":
                    path = _str_const(kw.value)
        if path is None or not path.startswith("/artifacts/"):
            _reject("open() is only allowed with a literal path under /artifacts/", node=node)
        if ".." in path.split("/"):
            _reject("open() paths may not contain '..'", node=node)
        mode_node = node.args[1] if len(node.args) >= 2 else next(
            (kw.value for kw in node.keywords if kw.arg == "mode"), None)
        if mode_node is None:
            return
        mode = _str_const(mode_node)
        if mode is None:
            _reject("open() needs a literal mode string", node=node)
        if set(mode) & _WRITE_MODE_CHARS and not path.startswith("/artifacts/out/"):
            _reject("open() may only write under /artifacts/out/", node=node)


def check_code(code: str) -> None:
    """Raise InvalidRequestError if ``code`` uses something the sandbox disallows."""
    try:
        tree = ast.parse(code, "<sandbox>", "exec")
    except SyntaxError as exc:
        _reject(f"syntax error: {exc.msg} (line {exc.lineno})",
                "Fix the Python syntax and resend the whole cell.")
    except ValueError as exc:  # e.g. NUL bytes in source
        _reject(f"syntax error: {exc}", "Fix the Python source and resend the whole cell.")
    _Guard().visit(tree)
