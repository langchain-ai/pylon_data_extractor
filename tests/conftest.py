"""Test configuration and fixtures."""

from unittest.mock import MagicMock

import pytest

from src.config import BigQueryConfig, Config, PylonConfig, ReplicationConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        pylon=PylonConfig(
            api_base_url="https://test.usepylon.com/api/v1",
            api_key="test_api_key",
            timeout=30,
            max_retries=3,
            retry_delay=1,
        ),
        bigquery=BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            service_account_key_path="test_key.json",
            location="US",
        ),
        replication=ReplicationConfig(
            batch_size=100,
            max_workers=2,
            incremental_column="updated_at",
            full_refresh=False,
        ),
        log_level="DEBUG",
    )


@pytest.fixture
def mock_pylon_client():
    """Create a mock Pylon client for testing."""
    client = MagicMock()
    client.iter_all_accounts.return_value = [
        {"id": "acc1", "name": "Account 1", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "acc2", "name": "Account 2", "created_at": "2024-01-02T00:00:00Z"},
    ]
    client.iter_all_issues.return_value = [
        {
            "id": "iss1",
            "title": "Issue 1",
            "account_id": "acc1",
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "iss2",
            "title": "Issue 2",
            "account_id": "acc2",
            "created_at": "2024-01-02T00:00:00Z",
        },
    ]
    client.iter_all_issue_messages.return_value = [
        {"id": "msg1", "content": "Message 1", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "msg2", "content": "Message 2", "created_at": "2024-01-02T00:00:00Z"},
    ]
    client.iter_all_contacts.return_value = [
        {
            "id": "con1",
            "name": "Contact 1",
            "account_id": "acc1",
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "con2",
            "name": "Contact 2",
            "account_id": "acc2",
            "created_at": "2024-01-02T00:00:00Z",
        },
    ]
    client.iter_all_users.return_value = [
        {
            "id": "user1",
            "name": "User 1",
            "email": "user1@test.com",
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "user2",
            "name": "User 2",
            "email": "user2@test.com",
            "created_at": "2024-01-02T00:00:00Z",
        },
    ]
    client.iter_all_teams.return_value = [
        {"id": "team1", "name": "Team 1", "created_at": "2024-01-01T00:00:00Z"},
        {"id": "team2", "name": "Team 2", "created_at": "2024-01-02T00:00:00Z"},
    ]
    return client


@pytest.fixture
def mock_bigquery_manager():
    """Create a mock BigQuery manager for testing."""
    manager = MagicMock()
    manager.ensure_dataset_exists.return_value = None
    manager.ensure_table_exists.return_value = None
    manager.upsert_data.return_value = None
    manager.get_max_timestamp.return_value = None
    manager.get_table_row_count.return_value = 0
    return manager


@pytest.fixture
def sample_account_data():
    """Sample account data for testing."""
    return {
        "id": "acc_123",
        "name": "Test Account",
        "domain": "test.com",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "custom_fields": {"industry": "Technology", "size": "Medium"},
    }


@pytest.fixture
def sample_issue_data():
    """Sample issue data for testing."""
    return {
        "id": "iss_123",
        "title": "Test Issue",
        "status": "open",
        "account_id": "acc_123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "custom_fields": {"priority": "High", "category": "Bug"},
    }


@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "id": "msg_123",
        "content": "Test message content",
        "author_id": "user_123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "attachments": [],
    }
