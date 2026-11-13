"""Agent-facing errors: a message plus the next action to take.

Every error the agent can correct keeps its historical base class
(``KeyError``/``ValueError``) so existing callers and tests keep working,
while also being an :class:`AnalystError` so the agent layer can catch one
type and hand ``hint`` back to the model.
"""


class AnalystError(Exception):
    """An error the agent can see, carrying a message and an optional hint."""

    def __init__(self, message: str, *, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message


class UnknownNameError(AnalystError, KeyError):
    """A name (metric, dimension, handle) that does not exist."""


class InvalidRequestError(AnalystError, ValueError):
    """A well-formed request the caller asked for in an unsupported way."""


class ForecastTooSlowError(AnalystError):
    """A forecast that exceeds its time budget."""
