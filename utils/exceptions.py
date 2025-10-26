class OGException(Exception):
    """Base exception for Roampal OG system."""
    pass

class OllamaException(OGException):
    """Exception for Ollama-specific errors."""
    pass