import pytest

from analyst.errors import AnalystError, InvalidRequestError
from analyst.sandbox.ast_guard import ALLOWED_IMPORTS, check_code

ALLOWED = [
    "df = load('r_0123456789abcdef')\nprint(df.describe())",
    "import pandas as pd\ndf = load('r_0123456789')\ndf.groupby('a')['b'].sum()",
    "import duckdb\ndf = load('r_0123456789')\nduckdb.sql('select count(*) from df').df()",
    "import numpy as np\nimport scipy.stats as st\nst.ttest_ind(np.ones(3), np.zeros(3))",
    "import plotly.express as px\nfig = px.line(load('r_0123456789'), x='a', y='b')",
    "from statsforecast.models import AutoETS\nfrom collections import Counter",
    "import polars as pl, statsmodels.api as sm, math, statistics, datetime, json, re",
    "import itertools, functools",
    "save(load('r_0123456789').head(), 'top rows')",
    "with open('/artifacts/out/notes.txt', 'w') as f:\n    f.write('hi')",
    "print(open('/artifacts/in/r_0123456789.parquet', 'rb').read(4))",
    "print(open('/artifacts/out/notes.txt').read())",
    "x = getattr(load('r_0123456789'), 'shape')",
    "class A:\n    def __init__(self):\n        self.x = 1",
    "",
    "import warnings\nwarnings.filterwarnings('ignore')",
    "from typing import Any\nfrom dataclasses import dataclass",
    "input = load('r_0123456789')",
    "help, eval = 1, 2\ndel help",
]


@pytest.mark.parametrize("code", ALLOWED)
def test_allowed(code):
    check_code(code)


REJECTED = [
    # (code, fragment of the message)
    ("import os", "os"),
    ("import subprocess", "subprocess"),
    ("import socket", "socket"),
    ("import ctypes", "ctypes"),
    ("import importlib", "importlib"),
    ("import sys", "sys"),
    ("from os import system", "os"),
    ("from os.path import join", "os"),
    ("import pandas, os", "os"),
    ("from . import x", "relative"),
    ("__import__('os')", "__import__"),
    ("eval('1+1')", "eval"),
    ("exec('x=1')", "exec"),
    ("compile('1', 'x', 'eval')", "compile"),
    ("e = eval", "eval"),
    ("globals()['x']", "globals"),
    ("locals()", "locals"),
    ("vars()", "vars"),
    ("breakpoint()", "breakpoint"),
    ("input()", "input"),
    ("input = load('r_0123456789')\nprint(input)", "input"),
    ("h = help", "help"),
    ("getattr(x, name)", "getattr"),
    ("getattr(x, '__class__')", "getattr"),
    ("setattr(x, name, 1)", "setattr"),
    ("setattr(x, '__class__', y)", "setattr"),
    ("delattr(x, '__dict__')", "delattr"),
    ("g = getattr", "getattr"),
    ("().__class__.__bases__[0].__subclasses__()", "__"),
    ("f.__globals__", "__globals__"),
    ("x.__builtins__", "__builtins__"),
    ("__builtins__", "__builtins__"),
    ("fn.__code__", "__code__"),
    ("pd.io.common.sys.modules", "sys.modules"),
    ("pd.io.common.os.system('id')", "os.system"),
    ("frame.f_globals", "f_globals"),
    ("open('/etc/shadow')", "open"),
    ("open(path)", "open"),
    ("open(f'/artifacts/{x}')", "open"),
    ("open('/artifacts/../etc/passwd')", "open"),
    ("open('/artifacts/in/x', 'w')", "/artifacts/out"),
    ("open('/artifacts/in/x', mode='a')", "/artifacts/out"),
    ("open('/artifacts/out/x', mode)", "mode"),
    ("o = open", "open"),
    ("def f(:\n  pass", "syntax"),
]


@pytest.mark.parametrize(("code", "fragment"), REJECTED)
def test_rejected(code, fragment):
    with pytest.raises(InvalidRequestError) as exc:
        check_code(code)
    err = exc.value
    assert isinstance(err, AnalystError) and isinstance(err, ValueError)
    assert fragment.lower() in err.message.lower()
    assert err.hint


def test_import_hint_lists_the_allowlist():
    with pytest.raises(InvalidRequestError) as exc:
        check_code("import subprocess")
    for mod in ALLOWED_IMPORTS:
        assert mod in exc.value.hint
    assert "load(" in exc.value.hint and "save(" in exc.value.hint


def test_open_hint_points_to_artifacts_and_helpers():
    with pytest.raises(InvalidRequestError) as exc:
        check_code("open('/etc/passwd')")
    assert "/artifacts/" in exc.value.hint and "load(" in exc.value.hint


def test_allowlist_matches_constraints():
    assert set(ALLOWED_IMPORTS) == {
        "duckdb", "pandas", "polars", "numpy", "scipy", "statsmodels", "statsforecast",
        "plotly", "math", "statistics", "datetime", "json", "re", "collections",
        "itertools", "functools", "warnings", "typing", "dataclasses",
    }


def test_syntax_error_reports_line():
    with pytest.raises(InvalidRequestError) as exc:
        check_code("x = 1\ny = (")
    assert "line 2" in exc.value.message


@pytest.mark.parametrize(("code", "fragment"), [
    ("x" + "+x" * 5000, "too deeply nested"),  # RecursionError in the visitor
    ("x" + "+x" * 200_000, "too deeply nested"),  # RecursionError in ast.parse
    ("(" * 300 + ")" * 300, "syntax error"),
])
def test_too_deep_or_large_is_invalid_request(code, fragment):
    with pytest.raises(InvalidRequestError) as exc:
        check_code(code)
    assert fragment in exc.value.message and exc.value.hint
