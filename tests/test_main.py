"""Tests for main module."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.main import main, run_replication, truncate_table


class TestMain:
    """Test main CLI functionality."""

    def test_main_help(self):
        """Test that help is displayed correctly."""
        # Capture stdout
        captured_output = StringIO()

        # Mock sys.argv and stdout
        with (
            patch("sys.argv", ["main.py", "--help"]),
            patch("sys.stdout", captured_output),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        # argparse exits with code 0 for help
        assert exc_info.value.code == 0
        output = captured_output.getvalue()
        assert "Pylon Data Extractor" in output
        assert "replicate" in output

    def test_run_replication_invalid_object_type(self):
        """Test that invalid object type raises error."""
        captured_output = StringIO()

        with (
            patch("sys.stderr", captured_output),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_replication("invalid_object")

        assert exc_info.value.code == 1

    @patch("src.main.get_config")
    @patch("src.main.PylonReplicator")
    def test_run_replication_accounts(self, mock_replicator_class, mock_get_config):
        """Test running replication for accounts."""
        # Setup mocks
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_replicator = MagicMock()
        mock_replicator_class.return_value = mock_replicator
        mock_replicator.replicate_object.return_value = {
            "object_type": "accounts",
            "table_name": "int__pylon_accounts",
            "total_processed": 10,
            "total_errors": 0,
            "success": True,
        }

        # Run the function
        run_replication("accounts")

        # Verify calls
        mock_replicator_class.assert_called_once_with(mock_config)
        mock_replicator.replicate_object.assert_called_once_with(
            "accounts", None, None, False, None
        )
        mock_replicator.close.assert_called_once()

    @patch("src.main.get_config")
    @patch("src.main.PylonReplicator")
    def test_run_replication_with_parameters(
        self, mock_replicator_class, mock_get_config
    ):
        """Test running replication with custom parameters."""
        # Setup mocks
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_replicator = MagicMock()
        mock_replicator_class.return_value = mock_replicator
        mock_replicator.replicate_object.return_value = {
            "object_type": "accounts",
            "table_name": "int__pylon_accounts",
            "total_processed": 5,
            "total_errors": 0,
            "success": True,
        }

        # Run the function with parameters
        run_replication(
            object_type="accounts",
            updated_dt="2024-01-01",
            batch_size=500,
            max_records=1000,
            save_each_page=True,
        )

        # Verify calls
        mock_replicator.replicate_object.assert_called_once_with(
            "accounts", "2024-01-01", 500, True, 1000
        )

    @patch("src.main.get_config")
    @patch("src.main.PylonReplicator")
    def test_run_replication_all_objects(self, mock_replicator_class, mock_get_config):
        """Test running replication for all objects."""
        # Setup mocks
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_replicator = MagicMock()
        mock_replicator_class.return_value = mock_replicator
        mock_replicator.replicate_all.return_value = {
            "results": {
                "accounts": {
                    "object_type": "accounts",
                    "total_processed": 5,
                    "total_errors": 0,
                    "success": True,
                },
                "issues": {
                    "object_type": "issues",
                    "total_processed": 3,
                    "total_errors": 0,
                    "success": True,
                },
            },
            "total_processed": 8,
            "total_errors": 0,
            "success": True,
        }

        # Run the function
        run_replication("all")

        # Verify calls
        mock_replicator.replicate_all.assert_called_once_with(None, None, False, None)

    @patch("src.main.get_config")
    @patch("src.main.BigQueryManager")
    def test_truncate_table_success(self, mock_bq_manager_class, mock_get_config):
        """Test successful table truncation."""
        # Setup mocks
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_bq_manager = MagicMock()
        mock_bq_manager_class.return_value = mock_bq_manager

        # Run the function
        truncate_table("test_table")

        # Verify calls
        mock_bq_manager_class.assert_called_once_with(mock_config)
        mock_bq_manager.truncate_table.assert_called_once_with("test_table")
        mock_bq_manager.close.assert_called_once()

    def test_main_no_command(self):
        """Test main function with no command shows help."""
        captured_output = StringIO()

        with (
            patch("sys.argv", ["main.py"]),
            patch("sys.stdout", captured_output),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        output = captured_output.getvalue()
        assert "usage:" in output

    @patch("src.main.run_replication")
    def test_main_replicate_command(self, mock_run_replication):
        """Test main function with replicate command."""
        with patch("sys.argv", ["main.py", "replicate", "accounts"]):
            main()

        mock_run_replication.assert_called_once_with(
            object_type="accounts",
            updated_dt=None,
            batch_size=None,
            log_level="INFO",
            config_file=None,
            save_each_page=False,
            max_records=None,
        )

    @patch("src.main.truncate_table")
    def test_main_truncate_command(self, mock_truncate_table):
        """Test main function with truncate command."""
        with patch("sys.argv", ["main.py", "truncate", "test_table"]):
            main()

        mock_truncate_table.assert_called_once_with(
            table_name="test_table", log_level="INFO"
        )
