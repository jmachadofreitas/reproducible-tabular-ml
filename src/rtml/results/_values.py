import math


def boolean_value(value: object) -> bool | None:
    """Read a boolean value from an in-memory or serialized report row."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def finite_number(value: object) -> float | None:
    """Read a finite number from an in-memory or serialized report row."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
