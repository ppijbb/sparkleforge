"""
ArXiv MCP Server - Embedded replacement for external @modelcontextprotocol/server-arxiv.

Provides academic paper search and metadata retrieval from arXiv using the official API.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse
from xml.etree import ElementTree as ET

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field
    import httpx
    FASTMCP_AVAILABLE = True
except ImportError as e:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    BaseModel = None
    Field = None
    httpx = None

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("arxiv")


class SearchInput(BaseModel):
    """Input schema for arXiv search."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    max_results: int = Field(default=10, ge=1, le=50)
    sort_by: str = Field(default="relevance")
    sort_order: str = Field(default="descending")


class GetPaperInput(BaseModel):
    """Input schema for getting paper details."""
    paper_id: str = Field(..., description="arXiv paper ID", min_length=1, max_length=20)


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
        
        entries = []
        for entry in root.findall("atom:entry", ns):
            entry_xml = ET.tostring(entry, encoding='unicode')
            parsed = parse_arxiv_entry(entry_xml)
            if parsed.get("id"):
                entries.append(parsed)
        
        opensearch = root.find("{http://a9.com/-/opensearch}totalResults")
        total_results = int(opensearch.text) if opensearch is not None else len(entries)
        
        return {
            "entries": entries,
            "total_results": total_results,
            "offset": 0,
            "items_per_page": len(entries)
        }
    
    except Exception as e:
        logger.error(f"Error parsing ATOM feed: {e}")
        return {"entries": [], "total_results": 0, "error": str(e)}


async def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance",
                       sort_order: str = "descending", timeout: int = 30) -> Dict[str, Any]:
    """Search arXiv for papers."""
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


async def get_paper(paper_id: str, timeout: int = 30) -> Dict[str, Any]:
    """Get detailed information about a specific arXiv paper."""
    paper_id = paper_id.strip()
    if not paper_id.startswith("arXiv:"):
        paper_id = f"arXiv:{paper_id}"
    
    try:
        params = {"id_list": paper_id, "max_results": 1}
        url = f"{ARXIV_API_BASE}?{urlencode(params)}"
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            result = parse_atom_feed(response.text)
            
            if result["entries"]:
                return {"success": True, "paper": result["entries"][0], "timestamp": datetime.now().isoformat()}
            else:
                return {"success": False, "paper_id": paper_id, "error": "Paper not found"}
    
    except Exception as e:
        logger.error(f"Error fetching paper {paper_id}: {str(e)}", exc_info=True)
        return {"success": False, "paper_id": paper_id, "error": str(e)}


ARXIV_CATEGORIES = {
    "Physics": {
        "astro-ph": "Astrophysics", "cond-mat": "Condensed Matter", "gr-qc": "General Relativity and Quantum Cosmology",
        "hep-ex": "High Energy Physics - Experiment", "hep-lat": "High Energy Physics - Lattice",
        "hep-ph": "High Energy Physics - Phenomenology", "hep-th": "High Energy Physics - Theory",
        "math-ph": "Mathematical Physics", "nlin": "Nonlinear Sciences", "nucl-ex": "Nuclear Experiment",
        "nucl-th": "Nuclear Theory", "physics": "Physics (other)", "quant-ph": "Quantum Physics"
    },
    "Mathematics": {
        "math.AG": "Algebraic Geometry", "math.AT": "Algebraic Topology", "math.CA": "Classical Analysis and ODEs",
        "math.CO": "Combinatorics", "math.CT": "Category Theory", "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry", "math.DS": "Dynamical Systems", "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics", "math.GN": "General Topology", "math.GR": "Group Theory",
        "math.GT": "Geometric Topology", "math.HO": "History and Overview", "math.IT": "Information Theory",
        "math.KT": "K-Theory and Homology", "math.LO": "Logic", "math.MG": "Metric Geometry",
        "math.MP": "Mathematical Physics", "math.NA": "Numerical Analysis", "math.NT": "Number Theory",
        "math.OA": "Operator Algebras", "math.OC": "Optimization and Control", "math.PR": "Probability",
        "math.QA": "Quantum Algebra", "math.RA": "Representation Theory", "math.RT": "Ring and Operator Algebras",
        "math.SG": "Spectral Geometry", "math.SP": "Spectral Theory", "math.ST": "Statistics Theory"
    },
    "Computer Science": {
        "cs.AI": "Artificial Intelligence", "cs.AR": "Hardware Architecture", "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering, Finance, and Science", "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language", "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition", "cs.CY": "Computational Complexity",
        "cs.DS": "Data Structures and Algorithms", "cs.DB": "Databases", "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics", "cs.DC": "Distributed Computing", "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory", "cs.GL": "General Literature", "cs.GR": "Graphics",
        "cs.GT": "Computer Science and Game Theory", "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval", "cs.IT": "Information Theory", "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science", "cs.MA": "Multiagent Systems", "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software", "cs.NA": "Numerical Analysis", "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture", "cs.OH": "Other Computer Science", "cs.OS": "Operating Systems",
        "cs.PF": "Performance", "cs.PL": "Programming Languages", "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation", "cs.SD": "Sound", "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks", "cs.SY": "Systems and Control"
    },
    "Quantitative Biology": {
        "q-bio.BM": "Biomolecules", "q-bio.CB": "Cell Behavior", "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks", "q-bio.NC": "Neurons and Cognition",
        "q-bio.OT": "Other Quantitative Biology", "q-bio.PE": "Populations and Evolution",
        "q-bio.QM": "Quantitative Methods", "q-bio.TO": "Tissues and Organs"
    },
    "Quantitative Finance": {
        "q-fin.CP": "Computational Finance", "q-fin.EC": "Economics", "q-fin.GN": "General Finance",
        "q-fin.MF": "Mathematical Finance", "q-fin.PM": "Portfolio Management", "q-fin.PR": "Pricing of Securities",
        "q-fin.RM": "Risk Management", "q-fin.ST": "Statistical Finance", "q-fin.TR": "Trading and Market Microstructure"
    },
    "Statistics": {
        "stat.AP": "Applications", "stat.CO": "Computation", "stat.ME": "Methodology", "stat.ML": "Machine Learning",
        "stat.OT": "Other Statistics", "stat.TH": "Theory"
    }
}


@mcp.tool()
async def search(input: SearchInput) -> str:
    """Search arXiv for academic papers."""
    result = await search_arxiv(input.query, input.max_results, input.sort_by, input.sort_order)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_paper_details(input: GetPaperInput) -> str:
    """Get detailed information about a specific arXiv paper."""
    result = await get_paper(input.paper_id)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_categories() -> str:
    """List all available arXiv categories."""
    categories = get_categories()
    flat = []
    for field, subs in categories.items():
        for code, name in subs.items():
            flat.append({"code": code, "field": field, "name": name})
    
    return json.dumps({
        "categories": categories,
        "flat_list": flat,
        "total_categories": len(flat),
        "fields": list(categories.keys())
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_by_category(category: str, max_results: int = 10, sort_by: str = "submittedDate",
                            sort_order: str = "descending") -> str:
    """Search papers in a specific arXiv category."""
    result = await search_arxiv(f"cat:{category}", max_results, sort_by, sort_order)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_recent_papers(days_back: int = 7, max_results: int = 10, category: str = "") -> str:
    """Search for recently submitted papers."""
    query = f"submittedDate:[NOW-{days_back}DAY TO NOW]"
    if category:
        query += f" AND cat:{category}"
    
    result = await search_arxiv(query, max_results, "submittedDate", "descending")
    return json.dumps(result, ensure_ascii=False, indent=2)


def get_categories() -> Dict[str, Dict[str, str]]:
    """Return all available arXiv categories."""
    return ARXIV_CATEGORIES


def run():
    """Run the arXiv MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
