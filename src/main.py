"""Main entry point for Pylon Data Extractor."""

import argparse
import sys
from datetime import datetime
from typing import Optional
import json

import structlog
from rich.console import Console
from rich.table import Table

# Initialize colorama for cross-platform color support
try:
    import colorama

    colorama.init()
except ImportError:
    pass  # colorama not available, colors may not work on Windows

from .bigquery_utils import BigQueryManager
from .config import get_config
from .replicator import PylonReplicator
from .classifier import PylonClassifier
from .ticket_closer import PylonTicketCloser

# Configure structured logging with colors
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()


def setup_logging(log_level: str) -> None:
    """Setup logging configuration."""
    import logging

    # Configure the standard library logging to work with structlog
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",  # Let structlog handle the formatting
        force=True,  # Override any existing configuration
    )


def truncate_table(table_name: str, log_level: str = "INFO") -> None:
    """Truncate a BigQuery table."""
    # Setup logging
    setup_logging(log_level)

    try:
        # Load configuration
        config = get_config()

        # Create BigQuery manager
        bq_manager = BigQueryManager(config)

        logger.info("Starting table truncation", table_name=table_name)

        # Truncate the table
        bq_manager.truncate_table(table_name)

        logger.info("Table truncated successfully", table_name=table_name)

        # Close connections
        bq_manager.close()

    except Exception as e:
        logger.error("Table truncation failed", table_name=table_name, error=str(e))
        sys.exit(1)


def run_ticket_closing(
    batch_size: int = 10,
    log_level: str = "INFO",
    max_records: Optional[int] = None,
    issue_id: Optional[str] = None,
) -> None:
    """Close eligible Slack tickets based on question analysis."""
    # Setup logging
    setup_logging(log_level)

    try:
        # Load configuration
        config = get_config()

        # Create ticket closer
        ticket_closer = PylonTicketCloser(config)

        logger.info("Starting ticket closing", batch_size=batch_size, max_records=max_records, issue_id=issue_id)

        # Run ticket closing
        start_time = datetime.now()
        result = ticket_closer.close_tickets(
            batch_size=batch_size,
            max_records=max_records,
            issue_id=issue_id
        )

        # Display results
        duration = datetime.now() - start_time

        table = Table(title="Ticket Closing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Issues Processed", str(result["total_processed"]))
        table.add_row("Issues Analyzed", str(result["total_analyzed"]))
        table.add_row("Issues Classified", str(result["total_classified"]))
        table.add_row("Issues Closed", str(result["total_closed"]))
        table.add_row("Total Errors", str(result["total_errors"]))
        table.add_row("Success", "✅" if result["success"] else "❌")
        table.add_row("Duration", str(duration))

        console.print(table)

        if not result["success"]:
            logger.error("Ticket closing failed", total_errors=result["total_errors"])
        else:
            logger.info("Ticket closing completed successfully", duration=str(duration))

        # Close connections
        ticket_closer.close()

    except Exception as e:
        logger.error("Ticket closing failed", error=str(e))
        sys.exit(1)


def run_classification(
    fields: list[str] = ["resolution", "category"],
    batch_size: int = 10,
    log_level: str = "INFO",
    max_records: Optional[int] = None,
    issue_id: Optional[str] = None,
    created_start: Optional[str] = None,
    created_end: Optional[str] = None,
) -> None:
    """Classify closed Pylon issues with missing resolution or category."""
    # Setup logging
    setup_logging(log_level)

    try:
        # Load configuration
        config = get_config()

        # Create classifier
        classifier = PylonClassifier(config)

        logger.info("Starting classification", fields=fields, batch_size=batch_size, max_records=max_records, issue_id=issue_id)

        # Run classification
        start_time = datetime.now()
        result = classifier.classify_issues(
            fields=fields,
            batch_size=batch_size,
            max_records=max_records,
            issue_id=issue_id,
            created_start=created_start,
            created_end=created_end
        )

        # Display results
        duration = datetime.now() - start_time

        table = Table(title="Classification Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Fields Classified", ", ".join(fields))
        table.add_row("Issues Processed", str(result["total_processed"]))
        table.add_row("Issues Classified", str(result["total_classified"]))
        table.add_row("Issues Updated", str(result["total_updated"]))
        table.add_row("Total Errors", str(result["total_errors"]))
        table.add_row("Success", "✅" if result["success"] else "❌")
        table.add_row("Duration", str(duration))

        console.print(table)

        if not result["success"]:
            logger.error("Classification failed", total_errors=result["total_errors"])
        else:
            logger.info("Classification completed successfully", duration=str(duration))

        # Close connections
        classifier.close()

    except Exception as e:
        logger.error("Classification failed", error=str(e))
        sys.exit(1)


def run_replication(
    object_type: str,
    batch_size: Optional[int] = None,
    log_level: str = "INFO",
    save_each_page: bool = False,
    max_records: Optional[int] = None,
    created_start: Optional[str] = None,
    created_end: Optional[str] = None,
    states: Optional[list[str]] = None,
    issues_filter_json: Optional[str] = None,
    issue_id: Optional[str] = None,
) -> None:
    """Extract and replicate data from Pylon to BigQuery."""

    # Validate object type first
    valid_objects = [
        "accounts",
        "issues",
        "messages",
        "contacts",
        "users",
        "teams",
        "all",
    ]
    if object_type not in valid_objects:
        logger.error(
            "Invalid object type", object_type=object_type, valid_objects=valid_objects
        )
        sys.exit(1)

    # Setup logging
    setup_logging(log_level)

    try:
        # Load configuration
        config = get_config()

        # Override batch size if provided
        if batch_size:
            # Affects downstream BigQuery batching semantics
            config.replication.batch_size = batch_size
            # Also use the same value for API pagination page size as requested
            config.replication.api_batch_size = batch_size

        # Override max records limit if provided
        if max_records:
            config.replication.max_records_limit = max_records

        # Note: updated_since filtering is not supported by the API

        # Create replicator
        replicator = PylonReplicator(config)

        logger.info("Starting replication", object_type=object_type)
        logger.info("Full replication")

        # Run replication
        start_time = datetime.now()

        # Build issues query options if provided
        issues_query = None
        if created_start or created_end or states or issues_filter_json or issue_id:
            issues_query = {}
            if created_start:
                issues_query["created_start"] = datetime.fromisoformat(created_start)
            if created_end:
                issues_query["created_end"] = datetime.fromisoformat(created_end)
            if states:
                issues_query["states"] = states
            if issues_filter_json:
                try:
                    issues_query["extra_filters"] = json.loads(issues_filter_json)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON provided for --issues-filter-json", error=str(e))
                    sys.exit(1)
            if issue_id:
                issues_query["issue_id"] = issue_id

        # Build messages filter from the same CLI arguments
        messages_filter = None
        if object_type == "messages" and (issue_id or created_start or created_end or states):
            messages_filter = {}
            if issue_id:
                messages_filter["issue_id"] = issue_id
            if created_start:
                messages_filter["start_time"] = datetime.fromisoformat(created_start)
            if created_end:
                messages_filter["end_time"] = datetime.fromisoformat(created_end)
            if states:
                # Convert from API format to display format for state filtering
                state_mapping = {
                    "new": "New",
                    "waiting_on_you": "Waiting on You",
                    "waiting_on_customer": "Waiting on Customer",
                    "on_hold": "On Hold",
                    "closed": "Closed"
                }
                # Use first state if multiple provided
                if len(states) > 1:
                    logger.warning("Multiple states provided for message filtering, using first one", states=states)
                messages_filter["state"] = state_mapping.get(states[0], states[0])

        if object_type == "all":
            results = replicator.replicate_all(
                batch_size, save_each_page, max_records
            )
            display_results(results, start_time)
        else:
            result = replicator.replicate_object(
                object_type, batch_size, save_each_page, max_records,
                issues_query=issues_query, messages_filter=messages_filter
            )
            display_single_result(result, start_time)

        # Close connections
        replicator.close()

    except Exception as e:
        logger.error("Replication failed", error=str(e))
        sys.exit(1)


def display_single_result(result: dict, start_time: datetime) -> None:
    """Display results for a single object type."""
    duration = datetime.now() - start_time

    table = Table(title=f"Replication Results - {result['object_type'].title()}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Object Type", result["object_type"])
    table.add_row("Table Name", result["table_name"])
    table.add_row("Total Processed", str(result["total_processed"]))
    table.add_row("Total Errors", str(result["total_errors"]))
    if "total_batches" in result:
        table.add_row("Total Batches", str(result["total_batches"]))
    table.add_row("Success", "✅" if result["success"] else "❌")
    table.add_row("Duration", str(duration))

    console.print(table)

    if not result["success"]:
        logger.error("Replication failed", total_errors=result["total_errors"])
    else:
        logger.info("Replication completed successfully", duration=str(duration))


def display_results(results: dict, start_time: datetime) -> None:
    """Display results for all object types."""
    duration = datetime.now() - start_time

    # Summary table
    summary_table = Table(title="Replication Summary")
    summary_table.add_column("Object Type", style="cyan")
    summary_table.add_column("Processed", style="green")
    summary_table.add_column("Errors", style="red")
    summary_table.add_column("Success", style="yellow")

    for object_type, result in results["results"].items():
        summary_table.add_row(
            object_type.title(),
            str(result["total_processed"]),
            str(result["total_errors"]),
            "✅" if result["success"] else "❌",
        )

    console.print(summary_table)

    # Overall stats
    overall_table = Table(title="Overall Statistics")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="magenta")

    overall_table.add_row("Total Processed", str(results["total_processed"]))
    overall_table.add_row("Total Errors", str(results["total_errors"]))
    overall_table.add_row("Overall Success", "✅" if results["success"] else "❌")
    overall_table.add_row("Duration", str(duration))

    console.print(overall_table)

    if not results["success"]:
        logger.error(
            "Replication completed with errors", total_errors=results["total_errors"]
        )
    else:
        logger.info("All replications completed successfully", duration=str(duration))


def main() -> None:
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Pylon Data Extractor - Extended data replication utility for Pylon using the Pylon REST API"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Replication command
    replicate_parser = subparsers.add_parser(
        "replicate", help="Replicate data from Pylon to BigQuery"
    )
    replicate_parser.add_argument(
        "object_type",
        help="Object type to replicate",
        choices=["accounts", "issues", "messages", "contacts", "users", "teams", "all"],
    )
    replicate_parser.add_argument(
        "--batch-size", type=int, help="Batch size for processing records", default=None
    )
    replicate_parser.add_argument(
        "--log-level",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    # Note: configuration file flag removed; environment/config module used instead
    replicate_parser.add_argument(
        "--save-each-page",
        action="store_true",
        help="Save data after each API page (more frequent saves, useful for large datasets)",
    )
    replicate_parser.add_argument(
        "--max-records",
        type=int,
        help="Maximum number of records to download (overrides config file setting)",
        default=None,
    )
    # Issues search filters
    replicate_parser.add_argument(
        "--created-start",
        help="Created_at start (RFC3339/ISO format)",
        default=None,
    )
    replicate_parser.add_argument(
        "--created-end",
        help="Created_at end (RFC3339/ISO format)",
        default=None,
    )
    replicate_parser.add_argument(
        "--states",
        nargs="+",
        help="Issue states filter (one or more)",
        choices=["new", "waiting_on_you", "waiting_on_customer", "on_hold", "closed"],
        default=None,
    )
    replicate_parser.add_argument(
        "--issues-filter-json",
        help="Additional JSON filters to merge into issues search body",
        default=None,
    )
    replicate_parser.add_argument(
        "--issue-id",
        help="Replicate a single issue by ID (overrides other issue filters)",
        default=None,
    )

    # Truncate command
    truncate_parser = subparsers.add_parser(
        "truncate", help="Truncate a BigQuery table"
    )
    truncate_parser.add_argument(
        "table_name", help="Name of the table to truncate (e.g., int__pylon_users)"
    )
    truncate_parser.add_argument(
        "--log-level",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    # Close command
    close_parser = subparsers.add_parser(
        "close", help="Close eligible Slack tickets based on question analysis"
    )
    close_parser.add_argument(
        "--log-level",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    close_parser.add_argument(
        "--batch-size", type=int, help="Batch size for processing issues", default=10
    )
    close_parser.add_argument(
        "--max-records", type=int, help="Maximum number of issues to process", default=None
    )
    close_parser.add_argument(
        "--issue-id",
        help="Process a single issue by ID",
        default=None,
    )

    # Classify command
    classify_parser = subparsers.add_parser(
        "classify", help="Classify closed Pylon issues with missing resolution or category"
    )
    classify_parser.add_argument(
        "--log-level",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    classify_parser.add_argument(
        "--batch-size", type=int, help="Batch size for processing issues", default=100
    )
    classify_parser.add_argument(
        "--max-records", type=int, help="Maximum number of issues to classify", default=None
    )
    classify_parser.add_argument(
        "--fields",
        nargs="+",
        help="Fields to classify",
        choices=["resolution", "category"],
        default=["resolution", "category"],
    )
    classify_parser.add_argument(
        "--issue-id",
        help="Classify a single issue by ID",
        default=None,
    )
    classify_parser.add_argument(
        "--created-start",
        help="Created_at start (RFC3339/ISO format)",
        default=None,
    )
    classify_parser.add_argument(
        "--created-end",
        help="Created_at end (RFC3339/ISO format)",
        default=None,
    )

    args = parser.parse_args()

    # Handle different commands
    if args.command == "replicate":
        run_replication(
            object_type=args.object_type,
            batch_size=args.batch_size,
            log_level=args.log_level,
            save_each_page=args.save_each_page,
            max_records=args.max_records,
            created_start=args.created_start,
            created_end=args.created_end,
            states=args.states,
            issues_filter_json=args.issues_filter_json,
            issue_id=args.issue_id,
        )
    elif args.command == "truncate":
        truncate_table(table_name=args.table_name, log_level=args.log_level)
    elif args.command == "close":
        run_ticket_closing(
            batch_size=args.batch_size,
            log_level=args.log_level,
            max_records=args.max_records,
            issue_id=args.issue_id,
        )
    elif args.command == "classify":
        run_classification(
            fields=args.fields,
            batch_size=args.batch_size,
            log_level=args.log_level,
            max_records=args.max_records,
            issue_id=args.issue_id,
            created_start=args.created_start,
            created_end=args.created_end,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
