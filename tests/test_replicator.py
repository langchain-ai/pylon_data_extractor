"""Tests for replication module."""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.replicator import PylonReplicator, ReplicationError


class TestPylonReplicator:
    """Test replication functionality."""

    def test_init(self, mock_config):
        """Test replicator initialization."""
        with (
            patch("src.replicator.PylonClient"),
            patch("src.replicator.BigQueryManager") as mock_bq_manager,
        ):

            replicator = PylonReplicator(mock_config)

            assert replicator.config == mock_config
            mock_bq_manager.return_value.ensure_dataset_exists.assert_called_once()

    def test_transform_item_account(self, mock_config):
        """Test transforming account data."""
        with (
            patch("src.replicator.PylonClient"),
            patch("src.replicator.BigQueryManager"),
        ):

            replicator = PylonReplicator(mock_config)

            account_data = {
                "id": "acc_123",
                "name": "Test Account",
                "domain": "test.com",
                "created_at": "2024-01-01T00:00:00Z",
            }

            result = replicator._transform_item(account_data, "accounts")

            assert result["account_id"] == "acc_123"
            assert result["data"] == account_data
            assert "updated_at" in result

    def test_transform_item_message(self, mock_config):
        """Test transforming message data."""
        with (
            patch("src.replicator.PylonClient"),
            patch("src.replicator.BigQueryManager"),
        ):

            replicator = PylonReplicator(mock_config)

            message_data = {
                "id": "msg_123",
                "content": "Test message",
                "created_at": "2024-01-01T00:00:00Z",
            }

            result = replicator._transform_item(
                message_data, "messages", issue_id="iss_123"
            )

            assert result["message_id"] == "msg_123"
            assert result["issue_id"] == "iss_123"
            assert result["data"] == message_data
            assert "updated_at" in result

    def test_transform_item_missing_primary_key(self, mock_config):
        """Test transforming item with missing primary key."""
        with (
            patch("src.replicator.PylonClient"),
            patch("src.replicator.BigQueryManager"),
        ):

            replicator = PylonReplicator(mock_config)

            account_data = {"name": "Test Account", "domain": "test.com"}

            with pytest.raises(ReplicationError, match="Missing primary key"):
                replicator._transform_item(account_data, "accounts")

    def test_transform_item_missing_issue_id_for_message(self, mock_config):
        """Test transforming message without issue_id."""
        with (
            patch("src.replicator.PylonClient"),
            patch("src.replicator.BigQueryManager"),
        ):

            replicator = PylonReplicator(mock_config)

            message_data = {"id": "msg_123", "content": "Test message"}

            with pytest.raises(ReplicationError, match="Missing issue_id"):
                replicator._transform_item(message_data, "messages")

    def test_replicate_object_accounts(
        self, mock_config, mock_pylon_client, mock_bigquery_manager
    ):
        """Test replicating accounts."""
        with (
            patch("src.replicator.PylonClient", return_value=mock_pylon_client),
            patch("src.replicator.BigQueryManager", return_value=mock_bigquery_manager),
        ):

            replicator = PylonReplicator(mock_config)

            result = replicator.replicate_object("accounts")

            assert result["object_type"] == "accounts"
            assert result["table_name"] == "int__pylon_accounts"
            assert result["total_processed"] == 2
            assert result["total_errors"] == 0
            assert result["success"] is True

            mock_bigquery_manager.ensure_table_exists.assert_called_once_with(
                "int__pylon_accounts"
            )
            mock_bigquery_manager.upsert_data.assert_called()

    def test_replicate_object_issues(
        self, mock_config, mock_pylon_client, mock_bigquery_manager
    ):
        """Test replicating issues."""
        with (
            patch("src.replicator.PylonClient", return_value=mock_pylon_client),
            patch("src.replicator.BigQueryManager", return_value=mock_bigquery_manager),
        ):

            replicator = PylonReplicator(mock_config)

            result = replicator.replicate_object("issues")

            assert result["object_type"] == "issues"
            assert result["table_name"] == "int__pylon_issues"
            assert result["total_processed"] == 2
            assert result["total_errors"] == 0
            assert result["success"] is True

    def test_replicate_object_messages(
        self, mock_config, mock_pylon_client, mock_bigquery_manager
    ):
        """Test replicating messages."""
        with (
            patch("src.replicator.PylonClient", return_value=mock_pylon_client),
            patch("src.replicator.BigQueryManager", return_value=mock_bigquery_manager),
        ):

            replicator = PylonReplicator(mock_config)

            result = replicator.replicate_object("messages")

            assert result["object_type"] == "messages"
            assert result["table_name"] == "int__pylon_messages"
            assert result["total_processed"] == 4  # 2 issues * 2 messages each
            assert result["total_errors"] == 0
            assert result["success"] is True

    def test_replicate_object_unknown_type(self, mock_config):
        """Test replicating unknown object type."""
        with (
            patch("src.replicator.PylonClient"),
            patch("src.replicator.BigQueryManager"),
        ):

            replicator = PylonReplicator(mock_config)

            with pytest.raises(ReplicationError, match="Unknown object type"):
                replicator.replicate_object("unknown")

    def test_replicate_all(self, mock_config, mock_pylon_client, mock_bigquery_manager):
        """Test replicating all object types."""
        with (
            patch("src.replicator.PylonClient", return_value=mock_pylon_client),
            patch("src.replicator.BigQueryManager", return_value=mock_bigquery_manager),
        ):

            replicator = PylonReplicator(mock_config)

            result = replicator.replicate_all()

            assert "results" in result
            assert "total_processed" in result
            assert "total_errors" in result
            assert "success" in result

            # Check that all object types were processed
            expected_objects = [
                "accounts",
                "issues",
                "messages",
                "contacts",
                "users",
                "teams",
            ]
            for obj_type in expected_objects:
                assert obj_type in result["results"]
                assert result["results"][obj_type]["success"] is True

    def test_replicate_with_default_params(
        self, mock_config, mock_pylon_client, mock_bigquery_manager
    ):
        """Test replication with default parameters."""
        with (
            patch("src.replicator.PylonClient", return_value=mock_pylon_client),
            patch("src.replicator.BigQueryManager", return_value=mock_bigquery_manager),
        ):

            replicator = PylonReplicator(mock_config)

            result = replicator.replicate_object("accounts")

            assert result["success"] is True
            # Verify that the cursor iterator was called with default parameters
            mock_pylon_client.iter_all_accounts.assert_called_once_with(
                batch_size=100,  # Mock config has batch_size=100
            )

    def test_replicate_with_batch_size(
        self, mock_config, mock_pylon_client, mock_bigquery_manager
    ):
        """Test replication with custom batch size."""
        with (
            patch("src.replicator.PylonClient", return_value=mock_pylon_client),
            patch("src.replicator.BigQueryManager", return_value=mock_bigquery_manager),
        ):

            replicator = PylonReplicator(mock_config)

            result = replicator.replicate_object("accounts", batch_size=50)

            assert result["success"] is True
            # Verify that the cursor iterator was called with custom batch size
            mock_pylon_client.iter_all_accounts.assert_called_once_with(
                batch_size=50
            )

    def test_close(self, mock_config, mock_pylon_client, mock_bigquery_manager):
        """Test closing the replicator."""
        with (
            patch("src.replicator.PylonClient", return_value=mock_pylon_client),
            patch("src.replicator.BigQueryManager", return_value=mock_bigquery_manager),
        ):

            replicator = PylonReplicator(mock_config)

            with (
                patch.object(mock_pylon_client, "close") as mock_pylon_close,
                patch.object(mock_bigquery_manager, "close") as mock_bq_close,
            ):

                replicator.close()

                mock_pylon_close.assert_called_once()
                mock_bq_close.assert_called_once()
