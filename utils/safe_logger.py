"""
Safe logging utilities that handle Unicode characters properly
"""

def sanitize_for_logging(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for safe logging by removing problematic Unicode characters
    and truncating to a reasonable length.
    """
    if not text:
        return ""
    
    # Replace common problematic Unicode characters
    text = text.replace('\u2011', '-')  # non-breaking hyphen
    text = text.replace('\u2022', '*')  # bullet point
    text = text.replace('\u25cf', '*')  # black circle
    text = text.replace('\u2019', "'")  # right single quote
    text = text.replace('\u201c', '"')  # left double quote
    text = text.replace('\u201d', '"')  # right double quote
    text = text.replace('\u2013', '-')  # en dash
    text = text.replace('\u2014', '--') # em dash
    
    # Remove any other non-ASCII characters
    text = ''.join(char if ord(char) < 128 else '?' for char in text)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text