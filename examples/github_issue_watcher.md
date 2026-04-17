# Implementing an Event-Driven Agent Watcher
This guide demonstrates how to use the generic `EventWatcher` framework built into SparkleForge. The `EventWatcher` allows agents to remain completely dormant (0 CPU & 0 API cost) and only wake up when a specific condition is met, perform their complex deep research or task orchestration, and then go back to sleep.

Below is an implementation example of an "Auto-Triage & Fixer" that wakes up whenever a new GitHub Issue is detected in a repository.

## Python Implementation

You can adapt this python implementation by changing the `poll_fn` and `handle_fn` for your specific use cases (like monitoring CVEs or RSS feeds).

```python
import asyncio
import logging
import os
import json
from src.core.event_watcher import EventWatcher
from src.core.iterative_research import (
    get_iterative_research_engine, 
    ThinkOutput, 
    ActionOutput, 
    ReportOutput, 
    QualityMetrics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GitHubWatcher")

MOCK_ISSUE_FILE = "mock_issue.json"

async def mock_github_poll():
    """Polls a local file pretending to check GitHub API for new issues."""
    if os.path.exists(MOCK_ISSUE_FILE):
        with open(MOCK_ISSUE_FILE, "r") as f:
            try:
                data = json.load(f)
                os.remove(MOCK_ISSUE_FILE) # consume the event
                return data
            except:
                return None
    return None

async def handler(issue_data):
    """Wake up handler. Runs the iterative research engine to triage the issue."""
    logger.info(f"Analyzing Event: {issue_data.get('title')}")
    
    # Using the Research Engine to triage and handle the event
    engine = get_iterative_research_engine(max_rounds=2, quality_threshold=0.8)
    
    # Process the issue
    state = await engine.run(
        query=f"Fix issue: {issue_data.get('title')}",
        session_id=issue_data.get('id', 'unknown'),
        think_fn=dummy_think,     # Replace with true LLM tool binding
        report_fn=dummy_report,  # Replace with true LLM tool binding
        action_fn=dummy_action   # Replace with true LLM tool binding
    )
    
    logger.info(f"Event triage complete. Result: {state.evolving_summary}")

async def monitor():
    # 1. Initialize Watcher
    watcher = EventWatcher(
        poll_fn=mock_github_poll,
        handle_fn=handler,
        interval=5 # Poll every 5 seconds
    )
    
    # 2. Start the watcher in the background
    await watcher.start()

if __name__ == "__main__":
    asyncio.run(monitor())
```

### Extending to Other Events
* **CVE Monitoring**: Replace `mock_github_poll` with a function querying the NIST NVD database.
* **RSS/News**: Read RSS feeds and trigger the handler only when a keyword (e.g., "OpenAI") is detected.
* **DevOps**: Poll tail logs or receive a web request to trigger bug-fixing agents on server crashes.
