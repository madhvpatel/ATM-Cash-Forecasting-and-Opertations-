"""ATM cash forecasting proof of concept."""

from typing import Any


def run_poc(*args: Any, **kwargs: Any) -> Any:
    """Load the reporting-heavy pipeline only when a full POC run is requested."""
    from .pipeline import run_poc as _run_poc

    return _run_poc(*args, **kwargs)

__all__ = ["run_poc"]
