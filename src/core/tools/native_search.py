
"""
Native Search Tools
External MCP servers (mcp-server-duckduckgo via uvx) can be flaky or blocked.
This module provides robust, native Python implementations of search tools using requests and BeautifulSoup.
"""

import logging
import requests
from bs4 import BeautifulSoup
import time
import random
from typing import List, Dict, Any, Optional
import urllib.parse

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
]

def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo using HTML frontend (no JS/API key required).
    Robust fallback for failing MCP servers.
    """
    params = {
        'q': query,
        'kl': 'us-en'  # Default to US English, can be made configurable
    }
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': 'https://duckduckgo.com/'
    }
    
    url = "https://html.duckduckgo.com/html/"
    
    logger.info(f"🦆 Native DDG Search: '{query}' (max: {max_results})")
    
    results = []
    
    try:
        # Retry logic
        for attempt in range(3):
            try:
                response = requests.post(url, data=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    break
                elif response.status_code == 429: # Rate limit
                     time.sleep(2 + random.random() * 2)
                     continue
                else:
                    logger.warning(f"DDG HTTP {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"DDG Connection error (attempt {attempt+1}): {e}")
                time.sleep(1)
        
        if 'response' not in locals() or response.status_code != 200:
            logger.error("Failed to fetch DDG results after retries")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse results
        # DDG HTML structure: .result -> .result__title -> a.result__a (link)
        # .result__snippet (snippet)
        
        for result in soup.select('.result'):
            if len(results) >= max_results:
                break
                
            title_tag = result.select_one('.result__title a.result__a')
            if not title_tag:
                continue
                
            link = title_tag['href']
            title = title_tag.get_text(strip=True)
            snippet_tag = result.select_one('.result__snippet')
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            
            # Filter internal DDG links (ads or related searches)
            if 'duckduckgo.com' in link and 'uddg=' in link:
                 # Decode actual URL
                 parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                 if 'uddg' in parsed:
                     link = parsed['uddg'][0]

            results.append({
                "title": title,
                "url": link,
                "snippet": snippet
            })
            
        logger.info(f"✅ Found {len(results)} native results")
        return results
        
    except Exception as e:
        logger.error(f"❌ Native DDG Search Error: {e}")
        return []

def search_duckduckgo_json(query: str, max_results: int = 5) -> str:
    """Wrapper that returns JSON string for tool compatibility."""
    import json
    data = search_duckduckgo(query, max_results)
    return json.dumps(data)
