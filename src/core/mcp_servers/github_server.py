"""
GitHub MCP Server - Embedded replacement for @modelcontextprotocol/server-github.

Provides GitHub API operations for repositories, issues, and more.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

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
mcp = FastMCP("github")


class GitHubConfig:
    """GitHub configuration."""
    API_BASE = "https://api.github.com"
    
    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get headers including auth if available."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SparkleForge-MCP/1.0"
        }
        
        token = None
        # Try multiple sources for token
        import os
        for env_var in ["GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PERSONAL_ACCESS_TOKEN"]:
            token = os.getenv(env_var)
            if token:
                break
        
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        return headers


class SearchReposInput(BaseModel):
    """Input for searching repositories."""
    query: str = Field(default="", description="Search query")
    language: Optional[str] = Field(default=None, description="Programming language")
    sort: str = Field(default="stars", description="Sort by: stars, forks, updated")
    order: str = Field(default="desc", description="Order: asc, desc")
    per_page: int = Field(default=10, ge=1, le=100)


class GetRepoInput(BaseModel):
    """Input for getting repository info."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class ListIssuesInput(BaseModel):
    """Input for listing issues."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    state: str = Field(default="open", description="Issue state: open, closed, all")
    labels: Optional[str] = Field(default=None, description="Label filter")
    per_page: int = Field(default=10, ge=1, le=100)


class GetIssueInput(BaseModel):
    """Input for getting a single issue."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    issue_number: int = Field(..., description="Issue number")


class CreateIssueInput(BaseModel):
    """Input for creating an issue."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    title: str = Field(..., description="Issue title", min_length=1)
    body: str = Field(default="", description="Issue body")
    labels: Optional[List[str]] = Field(default=None, description="Labels")


class ListPullRequestsInput(BaseModel):
    """Input for listing pull requests."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    state: str = Field(default="open", description="PR state: open, closed, all")
    per_page: int = Field(default=10, ge=1, le=100)


class GetFileInput(BaseModel):
    """Input for getting file contents."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    path: str = Field(..., description="File path")
    ref: Optional[str] = Field(default=None, description="Branch/tag/SHA")


class ListContentsInput(BaseModel):
    """Input for listing repository contents."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    path: str = Field(default="", description="Directory path")
    ref: Optional[str] = Field(default=None, description="Branch/tag/SHA")


class SearchCodeInput(BaseModel):
    """Input for searching code."""
    query: str = Field(..., description="Code search query")
    language: Optional[str] = Field(default=None, description="Language filter")
    repo: Optional[str] = Field(default=None, description="Repository filter (owner/repo)")
    per_page: int = Field(default=10, ge=1, le=100)


async def github_request(
    endpoint: str,
    method: str = "GET",
    data: Dict = None,
    params: Dict = None
) -> Dict[str, Any]:
    """Make a request to the GitHub API."""
    url = f"{GitHubConfig.API_BASE}/{endpoint.lstrip('/')}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers=GitHubConfig.get_headers(),
                json=data,
                params=params
            )
            
            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "GitHub authentication failed. Check GITHUB_TOKEN.",
                    "status_code": 401
                }
            
            if response.status_code == 403:
                return {
                    "success": False,
                    "error": "GitHub API rate limit exceeded or forbidden.",
                    "status_code": 403
                }
            
            if response.status_code == 404:
                return {
                    "success": False,
                    "error": "Resource not found",
                    "status_code": 404
                }
            
            response.raise_for_status()
            
            return {
                "success": True,
                "data": response.json()
            }
    
    except httpx.HTTPStatusError as e:
        logger.warning(f"GitHub API error: {e}")
        return {
            "success": False,
            "error": str(e),
            "status_code": e.response.status_code
        }
    
    except Exception as e:
        logger.error(f"GitHub request error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def search_repositories(input: SearchReposInput) -> str:
    """
    Search GitHub repositories.
    
    Input:
    - query: Search query (e.g., "machine learning")
    - language: Programming language filter
    - sort: Sort by stars, forks, or updated
    - order: asc or desc
    - per_page: Results per page (max 100)
    
    Returns JSON with matching repositories.
    """
    params = {
        "q": input.query,
        "sort": input.sort,
        "order": input.order,
        "per_page": input.per_page
    }
    
    if input.language:
        params["q"] += f" language:{input.language}"
    
    result = await github_request("/search/repositories", params=params)
    
    if result["success"]:
        repos = []
        for item in result["data"].get("items", []):
            repos.append({
                "id": item["id"],
                "name": item["name"],
                "full_name": item["full_name"],
                "owner": item["owner"]["login"],
                "description": item.get("description", ""),
                "url": item["html_url"],
                "stars": item["stargazers_count"],
                "forks": item["forks_count"],
                "language": item.get("language"),
                "created_at": item["created_at"],
                "updated_at": item["updated_at"]
            })
        
        return json.dumps({
            "success": True,
            "total_count": result["data"].get("total_count", 0),
            "repositories": repos
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_repository(input: GetRepoInput) -> str:
    """
    Get information about a repository.
    
    Input:
    - owner: Repository owner
    - repo: Repository name
    
    Returns JSON with repository info.
    """
    endpoint = f"/repos/{input.owner}/{input.repo}"
    result = await github_request(endpoint)
    
    if result["success"]:
        data = result["data"]
        return json.dumps({
            "success": True,
            "repository": {
                "id": data["id"],
                "name": data["name"],
                "full_name": data["full_name"],
                "owner": data["owner"]["login"],
                "description": data.get("description", ""),
                "url": data["html_url"],
                "stars": data["stargazers_count"],
                "forks": data["forks_count"],
                "watchers": data["watchers_count"],
                "language": data.get("language"),
                "default_branch": data.get("default_branch", "main"),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "pushed_at": data.get("pushed_at"),
                "open_issues": data["open_issues_count"]
            }
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_issues(input: ListIssuesInput) -> str:
    """
    List issues in a repository.
    
    Input:
    - owner: Repository owner
    - repo: Repository name
    - state: open, closed, or all
    - labels: Comma-separated label names
    - per_page: Results per page
    
    Returns JSON with issues list.
    """
    params = {
        "state": input.state,
        "per_page": input.per_page
    }
    
    if input.labels:
        params["labels"] = input.labels
    
    endpoint = f"/repos/{input.owner}/{input.repo}/issues"
    result = await github_request(endpoint, params=params)
    
    if result["success"]:
        issues = []
        for item in result["data"]:
            # Skip pull requests (they're also returned by issues endpoint)
            if "pull_request" in item:
                continue
            
            issues.append({
                "number": item["number"],
                "title": item["title"],
                "state": item["state"],
                "url": item["html_url"],
                "user": item["user"]["login"],
                "labels": [l["name"] for l in item.get("labels", [])],
                "created_at": item["created_at"],
                "updated_at": item["updated_at"],
                "body": item.get("body", "")[:500]
            })
        
        return json.dumps({
            "success": True,
            "repository": f"{input.owner}/{input.repo}",
            "count": len(issues),
            "issues": issues
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_issue(input: GetIssueInput) -> str:
    """
    Get a single issue.
    
    Input:
    - owner: Repository owner
    - repo: Repository name
    - issue_number: Issue number
    
    Returns JSON with issue details.
    """
    endpoint = f"/repos/{input.owner}/{input.repo}/issues/{input.issue_number}"
    result = await github_request(endpoint)
    
    if result["success"]:
        data = result["data"]
        return json.dumps({
            "success": True,
            "issue": {
                "number": data["number"],
                "title": data["title"],
                "state": data["state"],
                "url": data["html_url"],
                "user": data["user"]["login"],
                "labels": [l["name"] for l in data.get("labels", [])],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "body": data.get("body", ""),
                "comments": data["comments"]
            }
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def create_issue(input: CreateIssueInput) -> str:
    """
    Create a new issue.
    
    Input:
    - owner: Repository owner
    - repo: Repository name
    - title: Issue title
    - body: Issue body (optional)
    - labels: List of label names (optional)
    
    Returns JSON with created issue.
    """
    data = {
        "title": input.title,
        "body": input.body
    }
    
    if input.labels:
        data["labels"] = input.labels
    
    endpoint = f"/repos/{input.owner}/{input.repo}/issues"
    result = await github_request(endpoint, method="POST", data=data)
    
    if result["success"]:
        data = result["data"]
        return json.dumps({
            "success": True,
            "issue": {
                "number": data["number"],
                "title": data["title"],
                "url": data["html_url"],
                "state": data["state"]
            }
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_pull_requests(input: ListPullRequestsInput) -> str:
    """
    List pull requests in a repository.
    
    Input:
    - owner: Repository owner
    - repo: Repository name
    - state: open, closed, or all
    - per_page: Results per page
    
    Returns JSON with PRs list.
    """
    params = {
        "state": input.state,
        "per_page": input.per_page
    }
    
    endpoint = f"/repos/{input.owner}/{input.repo}/pulls"
    result = await github_request(endpoint, params=params)
    
    if result["success"]:
        prs = []
        for item in result["data"]:
            prs.append({
                "number": item["number"],
                "title": item["title"],
                "state": item["state"],
                "url": item["html_url"],
                "user": item["user"]["login"],
                "head": item["head"]["ref"],
                "base": item["base"]["ref"],
                "created_at": item["created_at"],
                "updated_at": item["updated_at"],
                "draft": item.get("draft", False),
                "mergeable": item.get("mergeable")
            })
        
        return json.dumps({
            "success": True,
            "repository": f"{input.owner}/{input.repo}",
            "count": len(prs),
            "pull_requests": prs
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_file_contents(input: GetFileInput) -> str:
    """
    Get contents of a file or directory.
    
    Input:
    - owner: Repository owner
    - repo: Repository name
    - path: File path
    - ref: Branch/tag/SHA (optional)
    
    Returns JSON with file content or directory listing.
    """
    params = {}
    if input.ref:
        params["ref"] = input.ref
    
    endpoint = f"/repos/{input.owner}/{input.repo}/contents/{input.path}"
    result = await github_request(endpoint, params=params)
    
    if result["success"]:
        data = result["data"]
        
        if data.get("type") == "file":
            import base64
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return json.dumps({
                "success": True,
                "type": "file",
                "path": data["path"],
                "content": content,
                "size": data["size"],
                "sha": data["sha"],
                "url": data["html_url"]
            }, ensure_ascii=False, indent=2)
        
        elif data.get("type") == "dir":
            contents = []
            for item in data.get("contents", []):
                contents.append({
                    "name": item["name"],
                    "path": item["path"],
                    "type": item["type"],
                    "size": item.get("size"),
                    "sha": item["sha"]
                })
            
            return json.dumps({
                "success": True,
                "type": "dir",
                "path": data["path"],
                "contents": contents,
                "count": len(contents)
            }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_code(input: SearchCodeInput) -> str:
    """
    Search code across GitHub.
    
    Input:
    - query: Code search query
    - language: Programming language filter
    - repo: Repository to search (owner/repo)
    - per_page: Results per page
    
    Returns JSON with code search results.
    """
    query = input.query
    if input.language:
        query += f" language:{input.language}"
    if input.repo:
        query += f" repo:{input.repo}"
    
    params = {
        "q": query,
        "per_page": input.per_page
    }
    
    result = await github_request("/search/code", params=params)
    
    if result["success"]:
        results = []
        for item in result["data"].get("items", []):
            results.append({
                "name": item["name"],
                "path": item["path"],
                "repository": item["repository"]["full_name"],
                "url": item["html_url"],
                "score": item.get("score"),
                "text_matches": item.get("text_matches", [])
            })
        
        return json.dumps({
            "success": True,
            "total_count": result["data"].get("total_count", 0),
            "results": results
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_user_info(username: str = Field(default="", description="Username (empty for authenticated user)")) -> str:
    """
    Get GitHub user information.
    
    Input:
    - username: Username (empty for current user)
    
    Returns JSON with user info.
    """
    endpoint = f"/users/{username}" if username else "/user"
    result = await github_request(endpoint)
    
    if result["success"]:
        data = result["data"]
        return json.dumps({
            "success": True,
            "user": {
                "login": data.get("login"),
                "name": data.get("name"),
                "email": data.get("email"),
                "bio": data.get("bio"),
                "public_repos": data.get("public_repos", 0),
                "public_gists": data.get("public_gists", 0),
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "url": data["html_url"],
                "created_at": data["created_at"]
            }
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_user_repos(
    username: str = Field(default="", description="Username (empty for authenticated user)"),
    sort: str = Field(default="updated", description="Sort by: updated, created, pushed, full_name"),
    per_page: int = Field(default=10, ge=1, le=100)
) -> str:
    """
    List repositories for a user.
    
    Input:
    - username: Username (empty for current user)
    - sort: Sort order
    - per_page: Results per page
    
    Returns JSON with repositories list.
    """
    endpoint = f"/users/{username}/repos" if username else "/user/repos"
    result = await github_request(endpoint, params={"sort": sort, "per_page": per_page})
    
    if result["success"]:
        repos = []
        for item in result["data"]:
            repos.append({
                "name": item["name"],
                "full_name": item["full_name"],
                "description": item.get("description", ""),
                "url": item["html_url"],
                "stars": item["stargazers_count"],
                "language": item.get("language"),
                "updated_at": item["updated_at"]
            })
        
        return json.dumps({
            "success": True,
            "repositories": repos,
            "count": len(repos)
        }, ensure_ascii=False, indent=2)
    
    return json.dumps(result, ensure_ascii=False, indent=2)


def run():
    """Run the GitHub MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
