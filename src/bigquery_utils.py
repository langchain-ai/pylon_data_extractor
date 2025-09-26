"""BigQuery utilities for data replication."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, SchemaField, Table, WriteDisposition
from google.oauth2 import service_account

from .config import Config, get_config

logger = structlog.get_logger(__name__)


class BigQueryError(Exception):
    """Exception raised for BigQuery operations."""

    pass


class BigQueryManager:
    """Manager for BigQuery operations."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.client = self._create_client()
        self.dataset_ref = self.client.dataset(self.config.bigquery.dataset_id)

    def _create_client(self) -> bigquery.Client:
        """Create BigQuery client with proper authentication."""
        try:
            # Try to load service account credentials
            if Path(self.config.bigquery.service_account_key_path).exists():
                logger.info(
                    "Using service account key",
                    path=self.config.bigquery.service_account_key_path,
                )
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.bigquery.service_account_key_path
                )
                return bigquery.Client(
                    credentials=credentials,
                    project=self.config.bigquery.project_id,
                    location=self.config.bigquery.location,
                )
            else:
                # Fall back to default credentials
                logger.info("Using default credentials")
                credentials, project = default()
                return bigquery.Client(
                    credentials=credentials,
                    project=self.config.bigquery.project_id or project,
                    location=self.config.bigquery.location,
                )
        except (DefaultCredentialsError, FileNotFoundError) as e:
            raise BigQueryError(f"Failed to create BigQuery client: {str(e)}")

    def ensure_dataset_exists(self) -> None:
        """Ensure the dataset exists, create if it doesn't."""
        try:
            self.client.get_dataset(self.dataset_ref)
            logger.info("Dataset exists", dataset_id=self.config.bigquery.dataset_id)
        except Exception:
            logger.info("Creating dataset", dataset_id=self.config.bigquery.dataset_id)
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = self.config.bigquery.location
            dataset.description = "Pylon extended data replication"

            self.client.create_dataset(dataset)
            logger.info("Dataset created", dataset_id=self.config.bigquery.dataset_id)

    def create_table(
        self, table_name: str, schema: List[SchemaField], description: str = ""
    ) -> Table:
        """Create a table with the given schema."""
        table_ref = self.dataset_ref.table(table_name)

        try:
            # Check if table exists
            table = self.client.get_table(table_ref)
            logger.info("Table already exists", table_name=table_name)
            return table
        except Exception:
            # Create table
            table = Table(table_ref, schema=schema)
            table.description = description

            # Add partitioning on updated_at if it exists
            updated_at_fields = [
                field for field in schema if field.name == "updated_at"
            ]
            if updated_at_fields:
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY, field="updated_at"
                )

            table = self.client.create_table(table)
            logger.info("Table created", table_name=table_name)
            return table

    def get_table_schema(self, table_name: str) -> List[SchemaField]:
        """Get the schema for a specific table based on the table name."""
        # Define schemas for each table type
        schemas = {
            "int__pylon_accounts": [
                SchemaField("account_id", "STRING", mode="REQUIRED"),
                SchemaField("data", "JSON", mode="NULLABLE"),
                SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "int__pylon_issues": [
                SchemaField("issue_id", "STRING", mode="REQUIRED"),
                SchemaField("data", "JSON", mode="NULLABLE"),
                SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "int__pylon_messages": [
                SchemaField("issue_id", "STRING", mode="REQUIRED"),
                SchemaField("message_id", "STRING", mode="REQUIRED"),
                SchemaField("data", "JSON", mode="NULLABLE"),
                SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "int__pylon_contacts": [
                SchemaField("contact_id", "STRING", mode="REQUIRED"),
                SchemaField("account_id", "STRING", mode="NULLABLE"),
                SchemaField("data", "JSON", mode="NULLABLE"),
                SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "int__pylon_users": [
                SchemaField("user_id", "STRING", mode="REQUIRED"),
                SchemaField("data", "JSON", mode="NULLABLE"),
                SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "int__pylon_teams": [
                SchemaField("team_id", "STRING", mode="REQUIRED"),
                SchemaField("data", "JSON", mode="NULLABLE"),
                SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ],
        }

        return schemas.get(table_name, [])

    def ensure_table_exists(self, table_name: str) -> None:
        """Ensure a table exists with the correct schema."""
        schema = self.get_table_schema(table_name)
        if not schema:
            raise BigQueryError(f"Unknown table schema for {table_name}")

        self.create_table(table_name, schema, f"Intermediate table for {table_name}")

    def upsert_data(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        primary_key_columns: List[str],
    ) -> None:
        """Upsert data into a BigQuery table using MERGE statement."""
        if not data:
            logger.info("No data to upsert", table_name=table_name)
            return

        table_ref = self.dataset_ref.table(table_name)

        # Add replication metadata
        replication_timestamp = datetime.utcnow().isoformat() + "Z"
        for row in data:
            row["updated_at"] = replication_timestamp

        # Deduplicate data by primary keys to avoid MERGE conflicts
        deduplicated_data = self._deduplicate_data(data, primary_key_columns)

        if len(deduplicated_data) != len(data):
            logger.warning(
                "Deduplicated data",
                table_name=table_name,
                original_count=len(data),
                deduplicated_count=len(deduplicated_data),
                duplicates_removed=len(data) - len(deduplicated_data),
            )

        # Serialize datetime objects to ISO format strings
        serialized_data = self._serialize_datetime_objects(deduplicated_data)

        # Create a temporary table for the data
        temp_table_name = f"{table_name}_temp_{int(datetime.utcnow().timestamp())}"
        temp_table_ref = self.dataset_ref.table(temp_table_name)

        logger.info(
            "Upserting data to BigQuery",
            table_name=table_name,
            row_count=len(data),
            primary_keys=primary_key_columns,
        )

        try:
            # Load data into temporary table
            schema = self.get_table_schema(table_name)
            job_config = LoadJobConfig(
                write_disposition=WriteDisposition.WRITE_TRUNCATE,
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                autodetect=False,
                schema=schema,
            )

            job = self.client.load_table_from_json(
                serialized_data, temp_table_ref, job_config=job_config
            )
            job.result()  # Wait for the job to complete

            # Build MERGE statement
            primary_key_conditions = []
            for col in primary_key_columns:
                primary_key_conditions.append(f"target.{col} = source.{col}")

            merge_condition = " AND ".join(primary_key_conditions)

            # Get all column names from the data
            data_columns = list(serialized_data[0].keys())

            # Build UPDATE SET clause
            update_set_clauses = []
            for col in data_columns:
                update_set_clauses.append(f"target.{col} = source.{col}")
            update_set_clause = ", ".join(update_set_clauses)

            # Build INSERT VALUES clause
            insert_columns = ", ".join(data_columns)
            insert_values = ", ".join([f"source.{col}" for col in data_columns])

            merge_query = f"""
            MERGE `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{table_name}` AS target
            USING `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{temp_table_name}` AS source
            ON {merge_condition}
            WHEN MATCHED THEN
                UPDATE SET {update_set_clause}
            WHEN NOT MATCHED THEN
                INSERT ({insert_columns})
                VALUES ({insert_values})
            """

            logger.debug("Executing MERGE query", query=merge_query)

            # Execute the MERGE query
            query_job = self.client.query(merge_query)
            query_job.result()  # Wait for the query to complete

            # Clean up temporary table
            self.client.delete_table(temp_table_ref)

            logger.info(
                "Data upserted successfully",
                table_name=table_name,
                rows_processed=len(serialized_data),
            )

        except Exception as e:
            # Clean up temporary table if it exists
            try:
                self.client.delete_table(temp_table_ref)
            except:
                pass

            # Provide more detailed error information
            error_msg = str(e)
            if "UPDATE/MERGE must match at most one source row" in error_msg:
                logger.error(
                    "MERGE operation failed due to duplicate primary keys in source data",
                    table_name=table_name,
                    primary_keys=primary_key_columns,
                    data_count=len(serialized_data),
                    error=error_msg,
                )
                raise BigQueryError(
                    f"MERGE operation failed for {table_name}: Duplicate primary keys found in source data. "
                    f"Primary keys: {primary_key_columns}. "
                    f"This usually indicates duplicate records in the batch being processed. "
                    f"Error details: {error_msg}"
                )
            else:
                logger.error(
                    "Failed to upsert data", table_name=table_name, error=error_msg
                )
                raise BigQueryError(
                    f"Failed to upsert data to {table_name}: {error_msg}"
                )

    def get_max_timestamp(
        self, table_name: str, timestamp_column: str = "updated_at"
    ) -> Optional[datetime]:
        """Get the maximum timestamp from a table for incremental replication."""
        try:
            query = f"""
            SELECT MAX({timestamp_column}) as max_timestamp
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{table_name}`
            """

            logger.info(
                "Getting max timestamp", table_name=table_name, column=timestamp_column
            )

            query_job = self.client.query(query)
            results = query_job.result()

            for row in results:
                if row.max_timestamp:
                    logger.info(
                        "Max timestamp found",
                        table_name=table_name,
                        max_timestamp=row.max_timestamp,
                    )
                    return row.max_timestamp

            logger.info("No timestamp found", table_name=table_name)
            return None

        except Exception as e:
            logger.warning(
                "Failed to get max timestamp", table_name=table_name, error=str(e)
            )
            return None

    def get_table_row_count(self, table_name: str) -> int:
        """Get the row count for a table."""
        query = f"""
        SELECT COUNT(*) as row_count
        FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{table_name}`
        """

        try:
            query_job = self.client.query(query)
            results = query_job.result()

            for row in results:
                return row.row_count

            return 0

        except Exception as e:
            logger.warning(
                "Failed to get row count", table_name=table_name, error=str(e)
            )
            return 0

    def truncate_table(self, table_name: str) -> None:
        """Truncate a table by deleting all rows."""
        table_ref = self.dataset_ref.table(table_name)

        try:
            # Check if table exists first
            self.client.get_table(table_ref)

            # Get row count before truncation
            row_count_before = self.get_table_row_count(table_name)

            # Truncate the table using DELETE statement
            truncate_query = f"""
            DELETE FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.{table_name}`
            WHERE TRUE
            """

            logger.info(
                "Truncating table", table_name=table_name, rows_before=row_count_before
            )

            query_job = self.client.query(truncate_query)
            query_job.result()  # Wait for the query to complete

            # Verify truncation
            row_count_after = self.get_table_row_count(table_name)

            logger.info(
                "Table truncated successfully",
                table_name=table_name,
                rows_before=row_count_before,
                rows_after=row_count_after,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Failed to truncate table", table_name=table_name, error=error_msg
            )
            raise BigQueryError(f"Failed to truncate table {table_name}: {error_msg}")

    def _deduplicate_data(
        self, data: List[Dict[str, Any]], primary_key_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """Deduplicate data by primary keys, keeping the latest record based on updated_at."""
        if not data or not primary_key_columns:
            return data

        # Create a dictionary to store unique records by primary key
        unique_records = {}
        duplicates_found = 0

        for record in data:
            # Create a composite key from primary key columns
            primary_key_values = tuple(record.get(col) for col in primary_key_columns)

            # Skip records with missing primary key values
            if None in primary_key_values:
                logger.warning(
                    "Skipping record with missing primary key",
                    record=record,
                    primary_key_columns=primary_key_columns,
                )
                continue

            # If we haven't seen this primary key, or if this record is newer, keep it
            if primary_key_values not in unique_records or record.get(
                "updated_at", ""
            ) > unique_records[primary_key_values].get("updated_at", ""):

                if primary_key_values in unique_records:
                    duplicates_found += 1
                    logger.debug(
                        "Replacing duplicate record with newer version",
                        primary_key=primary_key_values,
                        old_updated_at=unique_records[primary_key_values].get(
                            "updated_at"
                        ),
                        new_updated_at=record.get("updated_at"),
                    )

                unique_records[primary_key_values] = record

        if duplicates_found > 0:
            logger.info(
                "Found and resolved duplicates",
                duplicates_found=duplicates_found,
                total_records=len(data),
            )

        return list(unique_records.values())

    def _serialize_datetime_objects(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Serialize datetime objects to ISO format strings for JSON compatibility."""
        import copy

        serialized_data = copy.deepcopy(data)

        def serialize_datetime_in_dict(obj):
            """Recursively serialize datetime objects in a dictionary or list."""
            if isinstance(obj, dict):
                return {k: serialize_datetime_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_datetime_in_dict(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat() + "Z"
            else:
                return obj

        return [serialize_datetime_in_dict(row) for row in serialized_data]

    def get_issue_ids_for_filtering(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        issue_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[str]:
        """Get issue IDs from both pylon_issues and int__pylon_issues tables with comprehensive filtering.

        Args:
            start_time: Start time for filtering (inclusive)
            end_time: End time for filtering (inclusive)
            issue_id: Specific issue ID to filter for
            state: Issue state to filter for

        Returns:
            List of issue IDs
        """
        try:
            params = []

            # If filtering by specific issue_id, use simple query
            if issue_id:
                query = f"""
                SELECT '{issue_id}' as issue_id
                """
                logger.info("Querying for specific issue ID", issue_id=issue_id)

                query_job = self.client.query(query)
                results = query_job.result()
                issue_ids = [row.issue_id for row in results]

                logger.info("Retrieved specific issue ID", count=len(issue_ids))
                return issue_ids

            # Build comprehensive query with CTE for date range parameters
            cte_parts = []
            if start_time or end_time:
                if start_time:
                    params.append(bigquery.ScalarQueryParameter("start_time", "STRING", start_time.strftime('%Y-%m-%d')))
                    cte_parts.append("@start_time as start_time")
                else:
                    cte_parts.append("'1900-01-01' as start_time")

                if end_time:
                    params.append(bigquery.ScalarQueryParameter("end_time", "STRING", end_time.strftime('%Y-%m-%d')))
                    cte_parts.append("@end_time as end_time")
                else:
                    cte_parts.append("'2099-12-31' as end_time")
            else:
                cte_parts.extend(["'1900-01-01' as start_time", "'2099-12-31' as end_time"])

            # Build the main query
            query_parts = []

            # Add state parameter if provided
            if state:
                params.append(bigquery.ScalarQueryParameter("state_filter", "STRING", state))

            # CTE for time parameters
            query = f"""
            WITH times as (
                SELECT {', '.join(cte_parts)}
            )
            """

            # First part: query src_pylon.pylon_issues table
            pylon_issues_conditions = []
            if start_time or end_time:
                pylon_issues_conditions.append("""
                    (created_at BETWEEN CAST(times.start_time AS TIMESTAMP) AND CAST(times.end_time AS TIMESTAMP)
                     OR latest_message_time BETWEEN CAST(times.start_time AS TIMESTAMP) AND CAST(times.end_time AS TIMESTAMP))
                """)

            if state:
                # Handle state mapping from "Waiting on Customer" to "waiting_on_customer"
                pylon_issues_conditions.append("state = @state_filter")

            pylon_issues_where = " AND ".join(pylon_issues_conditions) if pylon_issues_conditions else "TRUE"

            query += f"""
            SELECT id as issue_id
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.pylon_issues`, times
            WHERE {pylon_issues_where}
            """

            # Second part: query int__pylon_issues table
            int_issues_conditions = []
            if start_time or end_time:
                int_issues_conditions.append("""
                    (TIMESTAMP(JSON_VALUE(data, "$.created_at"))
                      BETWEEN CAST(times.start_time AS TIMESTAMP) AND CAST(times.end_time AS TIMESTAMP)
                     OR TIMESTAMP(JSON_VALUE(data, "$.latest_message_time"))
                      BETWEEN CAST(times.start_time AS TIMESTAMP) AND CAST(times.end_time AS TIMESTAMP))
                """)

            if state:
                # Convert display state to internal state format
                internal_state = state.lower().replace(' ', '_')
                params.append(bigquery.ScalarQueryParameter("internal_state_filter", "STRING", internal_state))
                int_issues_conditions.append("JSON_VALUE(data, \"$.state\") = @internal_state_filter")

            int_issues_where = " AND ".join(int_issues_conditions) if int_issues_conditions else "TRUE"

            query += f"""
            UNION DISTINCT
            SELECT issue_id
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.int__pylon_issues`, times
            WHERE {int_issues_where}
            ORDER BY issue_id
            """

            logger.info(
                "Querying issue IDs from BigQuery with comprehensive filtering",
                start_time=start_time.isoformat() if start_time else None,
                end_time=end_time.isoformat() if end_time else None,
                state=state,
                has_time_filter=bool(start_time or end_time),
                has_state_filter=bool(state),
            )

            query_job = self.client.query(
                query, job_config=bigquery.QueryJobConfig(query_parameters=params)
            )
            results = query_job.result()

            issue_ids = [row.issue_id for row in results]

            logger.info(
                "Retrieved issue IDs from BigQuery",
                count=len(issue_ids),
                start_time=start_time.isoformat() if start_time else None,
                end_time=end_time.isoformat() if end_time else None,
                state=state,
            )

            return issue_ids

        except Exception as e:
            logger.error(
                "Failed to get issue IDs from BigQuery",
                error=str(e),
                start_time=start_time.isoformat() if start_time else None,
                end_time=end_time.isoformat() if end_time else None,
                state=state,
            )
            raise BigQueryError(f"Failed to get issue IDs from BigQuery: {str(e)}")

    def query_to_list(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results as a list of dictionaries."""
        try:
            logger.debug("Executing query", query=query)
            query_job = self.client.query(query)
            results = query_job.result()

            data = []
            for row in results:
                # Convert row to dictionary
                row_dict = {}
                for key, value in row.items():
                    row_dict[key] = value
                data.append(row_dict)

            logger.debug("Query executed successfully", row_count=len(data))
            return data

        except Exception as e:
            logger.error("Failed to execute query", query=query, error=str(e))
            raise BigQueryError(f"Failed to execute query: {str(e)}")

    def close(self) -> None:
        """Close the BigQuery client."""
        if hasattr(self.client, "close"):
            self.client.close()
