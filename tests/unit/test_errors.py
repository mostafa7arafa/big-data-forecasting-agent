import pytest

from analyst.data.engine import QueryError
from analyst.errors import (
    AnalystError,
    ForecastTooSlowError,
    InvalidRequestError,
    UnknownNameError,
)


def test_errors_keep_old_bases_and_hint():
    e = UnknownNameError("unknown metric 'trip'", hint="call describe('metrics')")
    assert isinstance(e, KeyError) and isinstance(e, AnalystError)
    assert e.hint == "call describe('metrics')" and str(e) == "unknown metric 'trip'"
    assert issubclass(QueryError, AnalystError)
    assert isinstance(InvalidRequestError("x"), ValueError)


def test_hint_defaults_to_empty_string():
    assert AnalystError("boom").hint == ""
    assert str(AnalystError("boom")) == "boom"


def test_forecast_too_slow_is_an_analyst_error():
    e = ForecastTooSlowError("forecast exceeded 120s", hint="run it in background")
    assert isinstance(e, AnalystError)
    assert not isinstance(e, (KeyError, ValueError))
    assert e.hint == "run it in background"


def test_query_error_keeps_its_message():
    e = QueryError("query may only read the dataset views")
    assert str(e) == "query may only read the dataset views"
    assert isinstance(e, AnalystError)


def test_catchable_as_analyst_error():
    with pytest.raises(AnalystError):
        raise UnknownNameError("unknown dimension 'zone'")
