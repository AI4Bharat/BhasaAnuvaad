class UnrecoverableError(Exception):
    """Custom exception class for specific error handling."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        if self.error_code:
            return f"[Error {self.error_code}]: {self.message}"
        return self.message
