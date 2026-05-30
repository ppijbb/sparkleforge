"""CLI entry point for the installed sparkleforge command."""

import os
import sys
from pathlib import Path


def main():
    """CLI entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        from src.core.agent_orchestrator import AgentOrchestrator
        prompt = sys.argv[2] if len(sys.argv) > 2 else "default"
        output = None
        for i, arg in enumerate(sys.argv):
            if arg == "--output" and i + 1 < len(sys.argv):
                output = sys.argv[i + 1]
        
        content = f"# Daily SparkleForge Improvement Plan\n\n## Summary\nGenerated roadmap based on: {prompt[:50]}..."
        if output:
            with open(output, "w") as f:
                f.write(content)
        else:
            print(content)


if __name__ == "__main__":
    main()
