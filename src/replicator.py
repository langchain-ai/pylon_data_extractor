"""Data replication module for Pylon to BigQuery."""

from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from .bigquery_utils import BigQueryError, BigQueryManager
from .config import Config, get_config
from .pylon_client import PylonAPIError, PylonClient

logger = structlog.get_logger(__name__)


class ReplicationError(Exception):
    """Exception raised for replication errors."""

    pass


class PylonReplicator:
    """Main replicator class for syncing Pylon data to BigQuery."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.pylon_client = PylonClient(self.config)
        self.bigquery_manager = BigQueryManager(self.config)

        # Ensure dataset and tables exist
        self.bigquery_manager.ensure_dataset_exists()

        # Table configurations
        self.table_configs = {
            "accounts": {
                "table_name": "int__pylon_accounts",
                "primary_keys": ["account_id"],
                "iterator": self.pylon_client.iter_all_accounts,
            },
            "issues": {
                "table_name": "int__pylon_issues",
                "primary_keys": ["issue_id"],
                "iterator": self.pylon_client.iter_all_issues,
            },
            "messages": {
                "table_name": "int__pylon_messages",
                "primary_keys": ["issue_id", "message_id"],
                "iterator": None,  # Special handling needed
            },
            "contacts": {
                "table_name": "int__pylon_contacts",
                "primary_keys": ["contact_id"],
                "iterator": self.pylon_client.iter_all_contacts,
            },
            "users": {
                "table_name": "int__pylon_users",
                "primary_keys": ["user_id"],
                "iterator": self.pylon_client.iter_all_users,
            },
            "teams": {
                "table_name": "int__pylon_teams",
                "primary_keys": ["team_id"],
                "iterator": self.pylon_client.iter_all_teams,
            },
        }

    def replicate_object(
        self,
        object_type: str,
        batch_size: Optional[int] = None,
        save_each_page: bool = False,
        max_records: Optional[int] = None,
        issues_query: Optional[Dict[str, Any]] = None,
        messages_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replicate a specific object type to BigQuery.

        Args:
            object_type: Type of object to replicate
            batch_size: Override default batch size
            save_each_page: If True, save data after each API page (more frequent saves)
            max_records: Maximum number of records to download (None for no limit)
            issues_query: Query parameters for filtering issues
            messages_filter: Filter parameters for message replication (start_time, end_time, issue_id, state)
        """
        if object_type not in self.table_configs:
            raise ReplicationError(f"Unknown object type: {object_type}")

        config = self.table_configs[object_type]
        table_name = config["table_name"]
        primary_keys = config["primary_keys"]
        batch_size = batch_size or self.config.replication.batch_size

        # Use max_records from parameter or config
        max_records = max_records or self.config.replication.max_records_limit

        logger.info(
            "Starting replication",
            object_type=object_type,
            table_name=table_name,
            save_each_page=save_each_page,
            max_records=max_records,
        )

        # Ensure table exists
        self.bigquery_manager.ensure_table_exists(table_name)

        # Special handling for messages
        if object_type == "messages":
            return self._replicate_messages(
                batch_size, save_each_page, max_records, messages_filter
            )

        # Replicate other object types
        return self._replicate_standard_object(
            object_type, batch_size, save_each_page, max_records, issues_query=issues_query
        )

    def _replicate_standard_object(
        self,
        object_type: str,
        batch_size: int = 1000,
        save_each_page: bool = False,
        max_records: Optional[int] = None,
        issues_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replicate standard object types (accounts, issues, contacts, users, teams)."""
        config = self.table_configs[object_type]
        table_name = config["table_name"]
        primary_keys = config["primary_keys"]
        iterator = config["iterator"]

        # Use configured batch sizes for API calls and BigQuery operations
        api_batch_size = self.config.replication.api_batch_size
        bigquery_batch_size = self.config.replication.bigquery_batch_size

        # Override with provided batch_size if it's smaller (for backward compatibility)
        if batch_size < bigquery_batch_size:
            bigquery_batch_size = batch_size

        # If save_each_page is True, use API batch size for BigQuery operations
        if save_each_page:
            bigquery_batch_size = api_batch_size
            logger.info(
                "Using save_each_page mode",
                api_batch_size=api_batch_size,
                bigquery_batch_size=bigquery_batch_size,
            )

        total_processed = 0
        total_errors = 0
        batch = []
        page_count = 0
        records_downloaded = 0

        try:
            if object_type == "issues":
                # Use new search-based iteration with optional filters
                issue_id = issues_query.get("issue_id") if issues_query else None
                if issue_id:
                    # Fetch a single issue and iterate over that single item
                    single = self.pylon_client.get_issue(issue_id)
                    source_iter = iter([single.get("data", single)])
                else:
                    created_start = issues_query.get("created_start") if issues_query else None
                    created_end = issues_query.get("created_end") if issues_query else None
                    states = issues_query.get("states") if issues_query else None
                    extra_filters = issues_query.get("extra_filters") if issues_query else None
                    source_iter = self.pylon_client.iter_all_issues(
                        created_start=created_start,
                        created_end=created_end,
                        states=states,
                        extra_filters=extra_filters,
                    )
            else:
                source_iter = iterator(
                    batch_size=api_batch_size
                )

            for item in source_iter:
                # Check if we've reached the record limit
                if max_records and records_downloaded >= max_records:
                    logger.info(
                        "Reached maximum record limit",
                        max_records=max_records,
                        records_downloaded=records_downloaded,
                    )
                    break
                # Transform item for BigQuery
                transformed_item = self._transform_item(item, object_type)
                batch.append(transformed_item)
                records_downloaded += 1

                # Process batch when it reaches the BigQuery batch size
                if len(batch) >= bigquery_batch_size:
                    try:
                        self.bigquery_manager.upsert_data(
                            table_name, batch, primary_keys
                        )
                        total_processed += len(batch)
                        page_count += 1
                        logger.info(
                            "Batch processed",
                            object_type=object_type,
                            batch_size=len(batch),
                            total_processed=total_processed,
                            page_count=page_count,
                            save_each_page=save_each_page,
                        )
                    except BigQueryError as e:
                        total_errors += len(batch)
                        logger.error(
                            "Batch failed",
                            object_type=object_type,
                            error=str(e),
                            batch_size=len(batch),
                        )

                    batch = []

            # Process remaining items in the last batch
            if batch:
                try:
                    self.bigquery_manager.upsert_data(table_name, batch, primary_keys)
                    total_processed += len(batch)
                    page_count += 1
                    logger.info(
                        "Final batch processed",
                        object_type=object_type,
                        batch_size=len(batch),
                        total_processed=total_processed,
                        page_count=page_count,
                    )
                except BigQueryError as e:
                    total_errors += len(batch)
                    logger.error(
                        "Final batch failed",
                        object_type=object_type,
                        error=str(e),
                        batch_size=len(batch),
                    )

            logger.info(
                "Replication completed",
                object_type=object_type,
                total_processed=total_processed,
                total_errors=total_errors,
                total_batches=page_count,
            )

            return {
                "object_type": object_type,
                "table_name": table_name,
                "total_processed": total_processed,
                "total_errors": total_errors,
                "total_batches": page_count,
                "success": total_errors == 0,
            }

        except PylonAPIError as e:
            logger.error(
                "API error during replication", object_type=object_type, error=str(e)
            )
            raise ReplicationError(
                f"API error during {object_type} replication: {str(e)}"
            )

    def _replicate_messages(
        self,
        batch_size: int = 1000,
        save_each_page: bool = False,
        max_records: Optional[int] = None,
        messages_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replicate messages for issues using DB-first approach or direct API call for single issue.

        Args:
            messages_filter: Optional filter with start_time, end_time, issue_id, or state
        """
        table_name = "int__pylon_messages"
        primary_keys = ["issue_id", "message_id"]

        # Use configured batch sizes for API calls and BigQuery operations
        api_batch_size = self.config.replication.api_batch_size
        bigquery_batch_size = self.config.replication.bigquery_batch_size

        # Override with provided batch_size if it's smaller (for backward compatibility)
        if batch_size < bigquery_batch_size:
            bigquery_batch_size = batch_size

        # If save_each_page is True, use API batch size for BigQuery operations
        if save_each_page:
            bigquery_batch_size = api_batch_size
            logger.info(
                "Using save_each_page mode for messages",
                api_batch_size=api_batch_size,
                bigquery_batch_size=bigquery_batch_size,
            )

        total_processed = 0
        total_errors = 0
        batch = []
        page_count = 0
        records_downloaded = 0

        try:
            # Extract filter parameters
            messages_filter = messages_filter or {}
            issue_id = messages_filter.get("issue_id")
            start_time = messages_filter.get("start_time")
            end_time = messages_filter.get("end_time")
            state = messages_filter.get("state")

            # If specific issue_id is provided, use it directly without DB query
            if issue_id:
                logger.info("Processing messages for specific issue", issue_id=issue_id)
                issue_ids = [issue_id]
            else:
                # Get issue IDs from the database using comprehensive filtering
                logger.info(
                    "Querying database for issue IDs",
                    start_time=start_time.isoformat() if start_time else None,
                    end_time=end_time.isoformat() if end_time else None,
                    state=state,
                )
                issue_ids = self.bigquery_manager.get_issue_ids_for_filtering(
                    start_time=start_time,
                    end_time=end_time,
                    state=state
                )

                if not issue_ids:
                    logger.info("No issues found matching the specified criteria")
                    return {
                        "object_type": "messages",
                        "table_name": table_name,
                        "total_processed": 0,
                        "total_errors": 0,
                        "total_batches": 0,
                        "success": True,
                    }

            logger.info("Found issues to process", issue_count=len(issue_ids))

            # Process messages for each issue ID
            for i, issue_id in enumerate(issue_ids, 1):
                # Check if we've reached the record limit
                if max_records and records_downloaded >= max_records:
                    logger.info(
                        "Reached maximum record limit",
                        max_records=max_records,
                        records_downloaded=records_downloaded,
                    )
                    break

                logger.info(
                    "Processing messages for issue",
                    issue_id=issue_id,
                    progress=f"{i}/{len(issue_ids)}",
                )

                try:
                    # Get all messages for this issue with rate limiting
                    for message in self.pylon_client.iter_all_issue_messages(
                        issue_id=issue_id,
                    ):
                        # Check if we've reached the record limit
                        if max_records and records_downloaded >= max_records:
                            logger.info(
                                "Reached maximum record limit",
                                max_records=max_records,
                                records_downloaded=records_downloaded,
                            )
                            break
                        # Transform message for BigQuery
                        transformed_message = self._transform_item(
                            message, "messages", issue_id=issue_id
                        )
                        batch.append(transformed_message)
                        records_downloaded += 1

                        # Process batch when it reaches the BigQuery batch size
                        if len(batch) >= bigquery_batch_size:
                            try:
                                self.bigquery_manager.upsert_data(
                                    table_name, batch, primary_keys
                                )
                                total_processed += len(batch)
                                page_count += 1
                                logger.info(
                                    "Message batch processed",
                                    batch_size=len(batch),
                                    total_processed=total_processed,
                                    page_count=page_count,
                                    issue_id=issue_id,
                                )
                            except BigQueryError as e:
                                total_errors += len(batch)
                                logger.error(
                                    "Message batch failed",
                                    error=str(e),
                                    batch_size=len(batch),
                                    issue_id=issue_id,
                                )

                            batch = []

                except PylonAPIError as e:
                    logger.debug(
                        "API error while processing messages for issue",
                        issue_id=issue_id,
                        error=str(e),
                    )
                    # Continue with next issue instead of failing completely
                    total_errors += 1
                    continue

            # Process remaining messages in the last batch
            if batch:
                try:
                    self.bigquery_manager.upsert_data(table_name, batch, primary_keys)
                    total_processed += len(batch)
                    page_count += 1
                    logger.info(
                        "Final message batch processed",
                        batch_size=len(batch),
                        total_processed=total_processed,
                        page_count=page_count,
                    )
                except BigQueryError as e:
                    total_errors += len(batch)
                    logger.error(
                        "Final message batch failed",
                        error=str(e),
                        batch_size=len(batch),
                    )

            logger.info(
                "Message replication completed",
                total_processed=total_processed,
                total_errors=total_errors,
                total_batches=page_count,
                issues_processed=len(issue_ids),
            )

            return {
                "object_type": "messages",
                "table_name": table_name,
                "total_processed": total_processed,
                "total_errors": total_errors,
                "total_batches": page_count,
                "success": total_errors == 0,
            }

        except Exception as e:
            logger.error("Error during message replication", error=str(e))
            raise ReplicationError(f"Error during message replication: {str(e)}")

    def _transform_item(
        self, item: Dict[str, Any], object_type: str, **kwargs
    ) -> Dict[str, Any]:
        """Transform an item from Pylon API to BigQuery format."""
        # Extract the primary key based on object type
        primary_key_mapping = {
            "accounts": "id",
            "issues": "id",
            "messages": "id",
            "contacts": "id",
            "users": "id",
            "teams": "id",
        }

        primary_key_field = primary_key_mapping.get(object_type, "id")
        primary_key_value = item.get(primary_key_field)

        if not primary_key_value:
            raise ReplicationError(
                f"Missing primary key {primary_key_field} for {object_type}"
            )

        # Build the transformed item
        transformed = {
            f"{object_type[:-1]}_id": primary_key_value,  # Remove 's' from plural
            "data": item,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        # Add special handling for messages (need issue_id)
        if object_type == "messages":
            issue_id = kwargs.get("issue_id")
            if not issue_id:
                raise ReplicationError("Missing issue_id for message")
            transformed["issue_id"] = issue_id
            transformed["message_id"] = primary_key_value

        # Add account_id for contacts if available
        if object_type == "contacts" and "account_id" in item:
            transformed["account_id"] = item["account_id"]

        return transformed

    def replicate_all(
        self,
        batch_size: Optional[int] = None,
        save_each_page: bool = False,
        max_records: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Replicate all object types to BigQuery."""
        results = {}
        total_processed = 0
        total_errors = 0

        logger.info("Starting full replication")

        for object_type in self.table_configs.keys():
            try:
                result = self.replicate_object(
                    object_type, batch_size, save_each_page, max_records
                )
                results[object_type] = result
                total_processed += result["total_processed"]
                total_errors += result["total_errors"]
            except Exception as e:
                logger.error(
                    "Replication failed for object type",
                    object_type=object_type,
                    error=str(e),
                )
                results[object_type] = {
                    "object_type": object_type,
                    "total_processed": 0,
                    "total_errors": 1,
                    "success": False,
                    "error": str(e),
                }
                total_errors += 1

        logger.info(
            "Full replication completed",
            total_processed=total_processed,
            total_errors=total_errors,
        )

        return {
            "results": results,
            "total_processed": total_processed,
            "total_errors": total_errors,
            "success": total_errors == 0,
        }

    def close(self) -> None:
        """Close all connections."""
        if self.pylon_client:
            self.pylon_client.close()
        if self.bigquery_manager:
            self.bigquery_manager.close()
