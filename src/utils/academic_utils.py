"""
Academic search utilities - arXiv API integration.

Extracted from embedded_mcp_servers for direct use in mcp_integration.py.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

logger = logging.getLogger(__name__)

# arXiv API utilities
ARXIV_API_BASE = "http://export.arxiv.org/api/query"


def parse_arxiv_datetime(dt_str: str) -> str:
    """Parse arXiv datetime format to ISO format."""
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(dt_str)
        return dt.isoformat()
    except Exception:
        return dt_str


def parse_arxiv_entry(entry_xml: str) -> Dict[str, Any]:
    """Parse a single arXiv entry from XML."""
    try:
        root = ET.fromstring(entry_xml)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        
        entry = {
            "id": "",
            "title": "",
            "summary": "",
            "authors": [],
            "published": "",
            "updated": "",
            "categories": [],
            "pdf_url": "",
            "primary_category": "",
            "comment": "",
            "journal_ref": "",
            "doi": ""
        }
        
        # Extract basic info
        id_elem = root.find("atom:id", ns)
        if id_elem is not None and id_elem.text:
            entry["id"] = id_elem.text.strip()
        
        title_elem = root.find("atom:title", ns)
        if title_elem is not None and title_elem.text:
            entry["title"] = re.sub(r'\s+', ' ', title_elem.text.strip())
        
        summary_elem = root.find("atom:summary", ns)
        if summary_elem is not None and summary_elem.text:
            entry["summary"] = re.sub(r'\s+', ' ', summary_elem.text.strip())
        
        for author in root.findall("atom:author", ns):
            name_elem = author.find("atom:name", ns)
            if name_elem is not None and name_elem.text:
                entry["authors"].append(name_elem.text.strip())
        
        published_elem = root.find("atom:published", ns)
        if published_elem is not None and published_elem.text:
            entry["published"] = parse_arxiv_datetime(published_elem.text.strip())
        
        updated_elem = root.find("atom:updated", ns)
        if updated_elem is not None and updated_elem.text:
            entry["updated"] = parse_arxiv_datetime(updated_elem.text.strip())
        
        for category in root.findall("atom:category", ns):
            cat_term = category.get("term", "")
            if cat_term:
                entry["categories"].append(cat_term)
                if not entry["primary_category"]:
                    entry["primary_category"] = cat_term
        
        for link in root.findall("atom:link", ns):
            if link.get("title") == "pdf":
                entry["pdf_url"] = link.get("href", "")
                break
        
        for comment in root.findall("arxiv:comment", ns):
            if comment.text:
                entry["comment"] = comment.text.strip()
            break
        
        for journal in root.findall("arxiv:journal_ref", ns):
            if journal.text:
                entry["journal_ref"] = journal.text.strip()
            break
        
        for doi in root.findall("arxiv:doi", ns):
            if doi.text:
                entry["doi"] = doi.text.strip()
            break
        
        return entry
    
    except Exception as e:
        logger.error(f"Error parsing arXiv entry: {e}")
        return {}


def parse_atom_feed(feed_xml: str) -> Dict[str, Any]:
    """Parse arXiv ATOM feed response."""
    try:
        root = ET.fromstring(feed_xml)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        opensearch_ns = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
        
        entries = []
        total_results = 0
        
        # Get total results
        total_elem = root.find("opensearch:totalResults", opensearch_ns)
        if total_elem is not None and total_elem.text:
            total_results = int(total_elem.text)
        
        # Parse entries
        for entry_elem in root.findall("atom:entry", ns):
            entry_xml = ET.tostring(entry_elem, encoding='unicode')
            entry = parse_arxiv_entry(entry_xml)
            if entry.get("id"):  # Only add entries with valid IDs
                entries.append(entry)
        
        return {
            "total_results": total_results if total_results > 0 else len(entries),
            "entries": entries
        }
    
    except Exception as e:
        logger.error(f"Error parsing arXiv feed: {e}")
        return {"total_results": 0, "entries": []}


async def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance",
                       sort_order: str = "descending", timeout: int = 30) -> Dict[str, Any]:
    """Search arXiv for papers."""
    if not HTTPX_AVAILABLE:
        return {
            "success": False,
            "query": query,
            "error": "httpx not available"
        }
    
    try:
        params = {
            "search_query": f"all:{query}",
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        url = f"{ARXIV_API_BASE}?{urlencode(params)}"
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            result = parse_atom_feed(response.text)
            
            return {
                "success": True,
                "query": query,
                "total_results": result["total_results"],
                "results": result["entries"],
                "count": len(result["entries"]),
                "search_params": {"sort_by": sort_by, "sort_order": sort_order, "max_results": max_results},
                "timestamp": datetime.now().isoformat()
            }
    
    except httpx.HTTPStatusError as e:
        logger.warning(f"arXiv API error: {e.response.status_code}")
        return {"success": False, "query": query, "error": f"HTTP {e.response.status_code}"}
    
    except httpx.TimeoutException:
        logger.warning(f"arXiv request timeout")
        return {"success": False, "query": query, "error": "Request timed out"}
    
    except Exception as e:
        logger.error(f"arXiv search error: {str(e)}", exc_info=True)
        return {"success": False, "query": query, "error": str(e)}
