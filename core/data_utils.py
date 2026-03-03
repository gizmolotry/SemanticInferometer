"""
core/data_utils.py - Canonical Schema Utilities
"""

def extract_article_text(article_dict: dict) -> str:
    """
    Extracts text from an article dictionary prioritizing:
    1. 'content' (Synthetic standard)
    2. 'text' (Legacy standard)
    3. 'body' (Alternative standard)
    
    Raises ValueError if no valid text is found.
    """
    if not isinstance(article_dict, dict):
        # Handle cases where articles might be passed as raw strings
        text = str(article_dict).strip()
    else:
        # Priority-based extraction
        text = article_dict.get('content') or article_dict.get('text') or article_dict.get('body')
    
    if text is None or not str(text).strip():
        keys = list(article_dict.keys()) if isinstance(article_dict, dict) else "N/A (not a dict)"
        raise ValueError(f"Schema Violation: Article missing valid text payload. Keys found: {keys}")
        
    return str(text).strip()
