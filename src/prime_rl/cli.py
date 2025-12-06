"""
CLI commands for PRIME-RL.

Provides subcommands for various PRIME-RL operations including
scenario listing and evaluation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from prime_rl.registry import get_registry
from loguru import logger


def list_scenarios_command(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    format: str = "table",
) -> None:
    """
    List available scenarios.
    
    Args:
        category: Filter by category
        tag: Filter by tag
        search: Search query
        format: Output format ("table" or "json")
    """
    registry = get_registry()
    
    # Get scenarios
    if search:
        scenarios = registry.search(search)
    elif category:
        scenarios = registry.list_by_category(category)
    elif tag:
        scenarios = registry.list_by_tag(tag)
    else:
        scenarios = registry.list_all()
    
    # Output
    if format == "json":
        import json
        scenarios_dict = [s.to_dict() for s in scenarios]
        print(json.dumps(scenarios_dict, indent=2))
    else:
        # Table format
        print(f"\nFound {len(scenarios)} scenario(s):\n")
        print(f"{'ID':<40} {'Name':<30} {'Category':<20}")
        print("-" * 90)
        
        for scenario in scenarios:
            print(f"{scenario.id:<40} {scenario.name:<30} {scenario.category:<20}")
            if scenario.description:
                print(f"  {scenario.description}")
            if scenario.config_path:
                print(f"  Config: {scenario.config_path}")
            print()


def main() -> None:
    """
    Main CLI entrypoint.
    
    Usage:
        prime-rl list-scenarios [--category CATEGORY] [--tag TAG] [--search QUERY] [--format FORMAT]
        prime-rl eval --config CONFIG --checkpoint CHECKPOINT [options]
        prime-rl train --config CONFIG [options]
    """
    parser = argparse.ArgumentParser(
        description="PRIME-RL command-line interface",
        prog="prime-rl",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # list-scenarios command
    list_parser = subparsers.add_parser(
        "list-scenarios",
        help="List available scenarios",
    )
    list_parser.add_argument(
        "--category",
        type=str,
        help="Filter by category",
    )
    list_parser.add_argument(
        "--tag",
        type=str,
        help="Filter by tag",
    )
    list_parser.add_argument(
        "--search",
        type=str,
        help="Search query",
    )
    list_parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    
    # eval command (delegates to eval.py)
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation on a trained model",
    )
    eval_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML configuration file",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    eval_parser.add_argument(
        "--project-id",
        type=str,
        help="Project identifier",
    )
    eval_parser.add_argument(
        "--run-id",
        type=str,
        help="Run identifier",
    )
    
    args = parser.parse_args()
    
    if args.command == "list-scenarios":
        list_scenarios_command(
            category=args.category,
            tag=args.tag,
            search=args.search,
            format=args.format,
        )
    elif args.command == "eval":
        # Import and run eval command
        from prime_rl.eval import eval_command
        eval_command(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            num_episodes=args.num_episodes,
            project_id=args.project_id,
            run_id=args.run_id,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

