"""Main entry point for Pylon Data Extractor."""

import argparse
import sys
from datetime import datetime
from typing import Optional

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


def run_replication(
    object_type: str,
    updated_dt: Optional[str] = None,
    batch_size: Optional[int] = None,
    log_level: str = "INFO",
    config_file: Optional[str] = None,
    save_each_page: bool = False,
    max_records: Optional[int] = None,
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
            config.replication.batch_size = batch_size

        # Override max records limit if provided
        if max_records:
            config.replication.max_records_limit = max_records

        # Parse updated_dt if provided
        updated_since = None
        if updated_dt:
            try:
                # Try parsing as ISO format first
                updated_since = datetime.fromisoformat(
                    updated_dt.replace("Z", "+00:00")
                )
            except ValueError:
                try:
                    # Try parsing as date only
                    updated_since = datetime.strptime(updated_dt, "%Y-%m-%d")
                except ValueError:
                    logger.error(
                        "Invalid date format",
                        updated_dt=updated_dt,
                        expected_formats=["YYYY-MM-DD", "YYYY-MM-DDTHH:MM:SS"],
                    )
                    sys.exit(1)

        # Create replicator
        replicator = PylonReplicator(config)

        logger.info("Starting replication", object_type=object_type)
        if updated_since:
            logger.info("Incremental replication", updated_since=updated_since)
        else:
            logger.info("Full replication")

        # Run replication
        start_time = datetime.now()

        if object_type == "all":
            results = replicator.replicate_all(
                updated_since, batch_size, save_each_page, max_records
            )
            display_results(results, start_time)
        else:
            result = replicator.replicate_object(
                object_type, updated_since, batch_size, save_each_page, max_records
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
        "--updated-dt",
        help="Start date for incremental replication (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
        default=None,
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
    replicate_parser.add_argument(
        "--config", help="Path to configuration file", default=None
    )
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
    truncate_parser.add_argument(
        "--config", help="Path to configuration file", default=None
    )

    args = parser.parse_args()

    # Handle different commands
    if args.command == "replicate":
        run_replication(
            object_type=args.object_type,
            updated_dt=args.updated_dt,
            batch_size=args.batch_size,
            log_level=args.log_level,
            config_file=args.config,
            save_each_page=args.save_each_page,
            max_records=args.max_records,
        )
    elif args.command == "truncate":
        truncate_table(table_name=args.table_name, log_level=args.log_level)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
