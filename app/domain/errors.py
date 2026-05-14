"""Domain-specific errors."""


class IllegalDrawError(ValueError):
    """Raised when a draw is not allowed for the given player state."""
