"""CI command extensions including causal blast-radius analysis."""

import argparse
import json
from src.core.ci.blast_radius import run_blast_radius_analysis

def register_ci_commands(subparsers):
    ci_parser = subparsers.add_parser("ci", help="CI and analysis tools")
    ci_sub = ci_parser.add_subparsers(dest="ci_subcommand")

    blast_parser = ci_sub.add_parser("blast-radius", help="Run causal blast-radius radar analysis")
    blast_parser.add_argument("--modules", nargs="+", required=True, help="List of modified modules")

    issue_digest_parser = ci_sub.add_parser("issue-digest", help="Run issue digest triage")
    issue_digest_parser.add_argument("--repo", required=True, help="GitHub repository name")

def handle_ci_command(args):
    if args.ci_subcommand == "blast-radius":
        result = run_blast_radius_analysis(args.modules)
        print(json.dumps(result, indent=2))
        return 0
    elif args.ci_subcommand == "issue-digest":
        print("Running issue digest triage...")
        return 0
    print("Unknown CI subcommand")
    return 1
