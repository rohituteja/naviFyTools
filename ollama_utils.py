"""
Utility functions for working with Ollama API URLs.
"""

def normalize_ollama_url(base_url: str) -> str:
    """
    Normalize Ollama URL for API calls.
    
    Handles various URL formats:
    - If URL ends with /v1, remove it (OpenAI compatibility layer)
    - If URL ends with /api, keep it (native Ollama API)
    - Otherwise, add /api (assume base URL without path)
    
    Args:
        base_url: The Ollama base URL from configuration
        
    Returns:
        Normalized URL ready for Ollama API endpoint appending (e.g., "/tags", "/embeddings")
        
    Examples:
        "http://localhost:11434/v1" -> "http://localhost:11434/api"
        "http://localhost:11434/api" -> "http://localhost:11434/api"
        "http://localhost:11434" -> "http://localhost:11434/api"
    """
    if not base_url:
        return None
    
    base_url = base_url.rstrip("/")
    
    # If it ends with /v1, remove it (OpenAI compatibility layer)
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    
    # If it doesn't end with /api, add it
    if not base_url.endswith("/api"):
        base_url = f"{base_url}/api"
    
    return base_url
