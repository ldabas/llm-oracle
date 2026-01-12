import logging
import json
from typing import Dict, List, Any
import config

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.DATE_FORMAT
    )
    logger = logging.getLogger("llm_oracle")
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

def extract_json_from_response(response_text: str) -> str:
    """Helper to extract JSON from LLM response that may have markdown formatting"""
    response_text = response_text.strip()
    
    # Remove markdown code blocks
    if response_text.startswith("```"):
        parts = response_text.split("```")
        if len(parts) >= 2:
            response_text = parts[1]
            # Remove language identifier (e.g., "json")
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
    
    return response_text

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Convert Gradio history format to a conversation string"""
    if not history:
        return ""
    
    conversation_parts = []
    # Handle both list of lists (old Gradio) and list of dicts (new Gradio/OpenAI format)
    # The ChatInterface type="messages" uses list of dicts: [{'role': 'user', 'content': '...'}, ...]
    
    # Keep last 6 messages for context
    recent_history = history[-6:] if len(history) > 6 else history
    
    for message in recent_history:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            # Fallback for older Gradio versions if needed, though type="messages" prevents this
            role = "user" # Assumption not quite right for list of lists, but we are using type="messages"
            content = message[0] # Wait, list of lists is [[user, bot], [user, bot]]
            # Let's stick to the dict format as per gradio_chat.py
            continue
        else:
            continue

        if role == "user":
            conversation_parts.append(f"User: {content}")
        elif role == "assistant":
            # Strip out sources section for cleaner context
            content_main = content
            if "\n\n---\n\n**📚 Sources Used:**" in content:
                content_main = content.split("\n\n---\n\n**📚 Sources Used:**")[0]
            elif "\n\n**Sources:**" in content:
                content_main = content.split("\n\n**Sources:**")[0]
            
            conversation_parts.append(f"Assistant: {content_main}")
    
    return "\n".join(conversation_parts)

def merge_citations_with_dedup(all_citations: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Merge and deduplicate citations, keeping only the most relevant ones.
    Filters out blocked non-technical domains.
    """
    merged = {
        'web_sources': [],
        'datastore_sources': []
    }
    
    seen_web_urls = set()
    seen_datastore_titles = set()
    blocked_count = 0
    
    for citation_set in all_citations:
        for web_source in citation_set.get('web_sources', []):
            url = web_source['url']
            domain = web_source.get('domain', '')
            
            # Filter out blocked domains
            if domain in config.BLOCKED_DOMAINS:
                blocked_count += 1
                continue
            
            if url not in seen_web_urls:
                seen_web_urls.add(url)
                merged['web_sources'].append(web_source)
        
        for ds_source in citation_set.get('datastore_sources', []):
            # Use title + filename as unique key for better deduplication
            unique_key = f"{ds_source['title']}_{ds_source.get('filename', '')}"
            if unique_key not in seen_datastore_titles:
                seen_datastore_titles.add(unique_key)
                merged['datastore_sources'].append(ds_source)
    
    if blocked_count > 0:
        logger.info(f"   🚫 Filtered out {blocked_count} non-technical sources")
    
    return merged
