"""Configuration management for Pylon Data Extractor."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class PylonConfig(BaseModel):
    """Configuration for Pylon API connection."""

    api_base_url: str = Field(
        default="https://app.usepylon.com/api/v1", description="Pylon API base URL"
    )
    api_key: Optional[str] = Field(default=None, description="Pylon API key")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(
        default=3, description="Maximum number of retries for API requests"
    )
    retry_delay: int = Field(
        default=1, description="Base delay between retries in seconds"
    )
    rate_limit_retries: int = Field(
        default=5, description="Maximum number of retries for rate limit errors"
    )
    rate_limit_base_delay: int = Field(
        default=1, description="Base delay for rate limit backoff in seconds"
    )
    rate_limit_max_delay: int = Field(
        default=300, description="Maximum delay for rate limit backoff in seconds"
    )
    rate_limit_jitter: bool = Field(
        default=True, description="Add jitter to rate limit backoff delays"
    )
    requests_per_minute: int = Field(
        default=60, description="Expected requests per minute limit"
    )


class BigQueryConfig(BaseModel):
    """Configuration for BigQuery connection."""

    project_id: str = Field(description="BigQuery project ID")
    dataset_id: str = Field(default="src_pylon", description="BigQuery dataset ID")
    service_account_key_path: str = Field(
        description="Path to service account key JSON file"
    )
    location: str = Field(default="US", description="BigQuery dataset location")


class ReplicationConfig(BaseModel):
    """Configuration for data replication process."""

    batch_size: int = Field(
        default=1000, description="Batch size for processing records"
    )
    api_batch_size: int = Field(
        default=100, description="Batch size for API pagination calls"
    )
    bigquery_batch_size: int = Field(
        default=500,
        description="Batch size for BigQuery operations (saves data more frequently)",
    )
    max_workers: int = Field(
        default=4, description="Maximum number of concurrent workers"
    )
    incremental_column: str = Field(
        default="updated_at", description="Column used for incremental replication"
    )
    full_refresh: bool = Field(
        default=False,
        description="Whether to perform full refresh instead of incremental",
    )
    max_records_limit: Optional[int] = Field(
        default=None,
        description="Maximum number of records to download (None for no limit)",
    )


class Config(BaseModel):
    """Main configuration class."""

    pylon: PylonConfig
    bigquery: BigQueryConfig
    replication: ReplicationConfig
    log_level: str = Field(default="INFO", description="Log level")

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        project_root = Path(__file__).parent.parent

        # Default service account key path
        default_key_path = (
            project_root / "config" / "pylon_bigquery_prod_service_key.json"
        )

        pylon_config = PylonConfig(
            api_base_url=os.getenv(
                "PYLON_API_BASE_URL", "https://app.usepylon.com/api/v1"
            ),
            api_key=os.getenv("PYLON_API_KEY"),
            timeout=int(os.getenv("PYLON_API_TIMEOUT", "60")),
            max_retries=int(os.getenv("PYLON_MAX_RETRIES", "3")),
            retry_delay=int(os.getenv("PYLON_RETRY_DELAY", "1")),
            rate_limit_retries=int(os.getenv("PYLON_RATE_LIMIT_RETRIES", "5")),
            rate_limit_base_delay=int(os.getenv("PYLON_RATE_LIMIT_BASE_DELAY", "1")),
            rate_limit_max_delay=int(os.getenv("PYLON_RATE_LIMIT_MAX_DELAY", "300")),
            rate_limit_jitter=os.getenv("PYLON_RATE_LIMIT_JITTER", "true").lower()
            == "true",
            requests_per_minute=int(os.getenv("PYLON_REQUESTS_PER_MINUTE", "60")),
        )

        bigquery_config = BigQueryConfig(
            project_id=os.getenv("BIGQUERY_PROJECT_ID", ""),
            dataset_id=os.getenv("BIGQUERY_DATASET_ID", "src_pylon"),
            service_account_key_path=os.getenv(
                "BIGQUERY_SERVICE_ACCOUNT_KEY_PATH", str(default_key_path)
            ),
            location=os.getenv("BIGQUERY_LOCATION", "US"),
        )

        replication_config = ReplicationConfig(
            batch_size=int(os.getenv("REPLICATION_BATCH_SIZE", "1000")),
            api_batch_size=int(os.getenv("REPLICATION_API_BATCH_SIZE", "100")),
            bigquery_batch_size=int(
                os.getenv("REPLICATION_BIGQUERY_BATCH_SIZE", "500")
            ),
            max_workers=int(os.getenv("REPLICATION_MAX_WORKERS", "4")),
            incremental_column=os.getenv(
                "REPLICATION_INCREMENTAL_COLUMN", "updated_at"
            ),
            full_refresh=os.getenv("REPLICATION_FULL_REFRESH", "false").lower()
            == "true",
            max_records_limit=(
                int(os.getenv("REPLICATION_MAX_RECORDS_LIMIT"))
                if os.getenv("REPLICATION_MAX_RECORDS_LIMIT")
                else None
            ),
        )

        return cls(
            pylon=pylon_config,
            bigquery=bigquery_config,
            replication=replication_config,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate_config(self) -> None:
        """Validate the configuration."""
        if not self.pylon.api_key:
            raise ValueError("PYLON_API_KEY environment variable must be set")

        if not self.bigquery.project_id:
            raise ValueError("BIGQUERY_PROJECT_ID environment variable must be set")

        if not Path(self.bigquery.service_account_key_path).exists():
            raise ValueError(
                f"Service account key file not found: {self.bigquery.service_account_key_path}"
            )


# Global configuration instance
config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = Config.from_env()
        config.validate_config()
    return config


def set_config(new_config: Config) -> None:
    """Set a new global configuration instance."""
    global config
    config = new_config
