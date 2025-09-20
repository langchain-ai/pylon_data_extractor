"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.config import BigQueryConfig, Config, PylonConfig, ReplicationConfig


class TestConfig:
    """Test configuration functionality."""

    def test_config_from_env(self):
        """Test configuration creation from environment variables."""
        with patch.dict(
            os.environ,
            {
                "PYLON_API_KEY": "test_key",
                "BIGQUERY_PROJECT_ID": "test_project",
                "BIGQUERY_SERVICE_ACCOUNT_KEY_PATH": "test_key.json",
                "LOG_LEVEL": "DEBUG",
            },
        ):
            config = Config.from_env()

            assert config.pylon.api_key == "test_key"
            assert config.bigquery.project_id == "test_project"
            assert config.bigquery.service_account_key_path == "test_key.json"
            assert config.log_level == "DEBUG"

    def test_config_validation_success(self, mock_config):
        """Test successful configuration validation."""
        with patch("pathlib.Path.exists", return_value=True):
            mock_config.validate_config()  # Should not raise

    def test_config_validation_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = Config(
            pylon=PylonConfig(api_key=None),
            bigquery=BigQueryConfig(
                project_id="test", service_account_key_path="test.json"
            ),
            replication=ReplicationConfig(),
        )

        with pytest.raises(ValueError, match="PYLON_API_KEY"):
            config.validate_config()

    def test_config_validation_missing_project_id(self):
        """Test configuration validation with missing project ID."""
        config = Config(
            pylon=PylonConfig(api_key="test"),
            bigquery=BigQueryConfig(
                project_id="", service_account_key_path="test.json"
            ),
            replication=ReplicationConfig(),
        )

        with pytest.raises(ValueError, match="BIGQUERY_PROJECT_ID"):
            config.validate_config()

    def test_config_validation_missing_key_file(self, mock_config):
        """Test configuration validation with missing key file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError, match="Service account key file not found"):
                mock_config.validate_config()

    def test_pylon_config_defaults(self):
        """Test Pylon configuration defaults."""
        config = PylonConfig()

        assert config.api_base_url == "https://app.usepylon.com/api/v1"
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.retry_delay == 1

    def test_bigquery_config_defaults(self):
        """Test BigQuery configuration defaults."""
        config = BigQueryConfig(project_id="test", service_account_key_path="test.json")

        assert config.dataset_id == "src_pylon"
        assert config.location == "US"

    def test_replication_config_defaults(self):
        """Test replication configuration defaults."""
        config = ReplicationConfig()

        assert config.batch_size == 1000
        assert config.max_workers == 4
        assert config.incremental_column == "updated_at"
        assert config.full_refresh is False
