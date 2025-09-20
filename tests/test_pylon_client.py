"""Tests for Pylon API client."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.pylon_client import PylonAPIError, PylonClient, PylonRateLimitError


class TestPylonClient:
    """Test Pylon API client functionality."""

    def test_init(self, mock_config):
        """Test client initialization."""
        client = PylonClient(mock_config)

        assert client.config == mock_config
        assert client.session is not None
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer test_api_key"

    def test_make_request_success(self, mock_config):
        """Test successful API request."""
        client = PylonClient(mock_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [], "total": 0}
        mock_response.elapsed.total_seconds.return_value = 0.5

        with patch.object(client.session, "request", return_value=mock_response):
            result = client._make_request("GET", "/test")

            assert result == {"data": [], "total": 0}

    def test_make_request_http_error(self, mock_config):
        """Test API request with HTTP error."""
        client = PylonClient(mock_config)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Bad request"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Client Error"
        )

        with patch.object(client.session, "request", return_value=mock_response):
            with pytest.raises(Exception):  # Will be RetryError due to retry logic
                client._make_request("GET", "/test")

    def test_make_request_rate_limit(self, mock_config):
        """Test API request with rate limiting."""
        client = PylonClient(mock_config)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "429 Too Many Requests"
        )

        with patch.object(client.session, "request", return_value=mock_response):
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(Exception):  # Will be RetryError due to retry logic
                    client._make_request("GET", "/test")
                # Should have called sleep for rate limiting
                assert mock_sleep.called

    def test_get_accounts(self, mock_config):
        """Test getting accounts."""
        client = PylonClient(mock_config)

        mock_response = {"data": [{"id": "acc1"}], "total": 1}
        with patch.object(
            client, "_make_request_with_rate_limit", return_value=mock_response
        ):
            result = client.get_accounts(limit=10, offset=0)

            assert result == mock_response
            client._make_request_with_rate_limit.assert_called_once_with(
                "GET", "/accounts", params={"limit": 10, "offset": 0}
            )

    def test_get_accounts_with_updated_since(self, mock_config):
        """Test getting accounts with updated_since parameter."""
        client = PylonClient(mock_config)

        updated_since = datetime(2024, 1, 1)
        mock_response = {"data": [], "total": 0}

        with patch.object(
            client, "_make_request_with_rate_limit", return_value=mock_response
        ):
            client.get_accounts(updated_since=updated_since)

            client._make_request_with_rate_limit.assert_called_once_with(
                "GET",
                "/accounts",
                params={
                    "limit": 100,
                    "offset": 0,
                    "updated_since": "2024-01-01T00:00:00",
                },
            )

    def test_iter_all_accounts(self, mock_config):
        """Test iterating through all accounts."""
        client = PylonClient(mock_config)

        # Mock multiple pages of data
        responses = [
            {"data": [{"id": f"acc{i}"} for i in range(1, 3)], "total": 5},
            {"data": [{"id": f"acc{i}"} for i in range(3, 5)], "total": 5},
            {"data": [{"id": "acc5"}], "total": 5},
            {"data": [], "total": 5},  # Empty page to stop iteration
        ]

        with patch.object(client, "get_accounts", side_effect=responses):
            accounts = list(client.iter_all_accounts(batch_size=2))

            assert len(accounts) == 5
            assert accounts[0]["id"] == "acc1"
            assert accounts[4]["id"] == "acc5"

    def test_iter_all_issues(self, mock_config):
        """Test iterating through all issues."""
        client = PylonClient(mock_config)

        responses = [
            {"data": [{"id": f"iss{i}"} for i in range(1, 3)], "total": 3},
            {"data": [{"id": "iss3"}], "total": 3},
            {"data": [], "total": 3},
        ]

        with patch.object(client, "get_issues", side_effect=responses):
            issues = list(client.iter_all_issues(batch_size=2))

            assert len(issues) == 3
            assert issues[0]["id"] == "iss1"
            assert issues[2]["id"] == "iss3"

    def test_iter_all_issue_messages(self, mock_config):
        """Test iterating through all issue messages."""
        client = PylonClient(mock_config)

        responses = [
            {"data": [{"id": f"msg{i}"} for i in range(1, 3)], "total": 2},
            {"data": [], "total": 2},
        ]

        with patch.object(client, "get_issue_messages", side_effect=responses):
            messages = list(client.iter_all_issue_messages("iss1", batch_size=2))

            assert len(messages) == 2
            assert messages[0]["id"] == "msg1"
            assert messages[1]["id"] == "msg2"

    def test_close(self, mock_config):
        """Test closing the client."""
        client = PylonClient(mock_config)

        with patch.object(client.session, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()

    def test_calculate_backoff_delay_with_retry_after(self, mock_config):
        """Test backoff delay calculation with retry-after header."""
        client = PylonClient(mock_config)

        # Test with retry-after header
        delay = client._calculate_backoff_delay(1, retry_after=120)
        assert delay == 120

        # Test with jitter enabled
        delay_with_jitter = client._calculate_backoff_delay(1, retry_after=100)
        assert 100 <= delay_with_jitter <= 130  # 100 + up to 30% jitter

    def test_calculate_backoff_delay_exponential(self, mock_config):
        """Test exponential backoff delay calculation."""
        client = PylonClient(mock_config)

        # Test exponential backoff without retry-after
        delay1 = client._calculate_backoff_delay(1)
        delay2 = client._calculate_backoff_delay(2)
        delay3 = client._calculate_backoff_delay(3)

        assert delay1 < delay2 < delay3
        assert delay1 >= 1  # Base delay
        assert delay3 <= 300  # Max delay

    def test_make_request_with_rate_limit_success(self, mock_config):
        """Test successful API request with rate limit handling."""
        client = PylonClient(mock_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [], "total": 0}
        mock_response.elapsed.total_seconds.return_value = 0.5

        with patch.object(client.session, "request", return_value=mock_response):
            result = client._make_request_with_rate_limit("GET", "/test")

            assert result == {"data": [], "total": 0}

    def test_make_request_with_rate_limit_429(self, mock_config):
        """Test rate limit handling in enhanced request method."""
        client = PylonClient(mock_config)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.json.return_value = {"message": "Rate limited"}

        with patch.object(client.session, "request", return_value=mock_response):
            with pytest.raises(PylonRateLimitError) as exc_info:
                client._make_request_with_rate_limit("GET", "/test")

            assert exc_info.value.retry_after == 60
            assert exc_info.value.status_code == 429

    def test_handle_pagination_response(self, mock_config):
        """Test pagination response handling."""
        client = PylonClient(mock_config)

        # Test with more data available
        response = {"data": [{"id": "1"}, {"id": "2"}], "total": 10}
        data, has_more, next_offset = client._handle_pagination_response(response, 0, 2)

        assert data == [{"id": "1"}, {"id": "2"}]
        assert has_more is True
        assert next_offset == 2

        # Test with no more data
        response = {"data": [{"id": "9"}, {"id": "10"}], "total": 10}
        data, has_more, next_offset = client._handle_pagination_response(response, 8, 2)

        assert data == [{"id": "9"}, {"id": "10"}]
        assert has_more is False
        assert next_offset == 8

    def test_handle_cursor_pagination_response(self, mock_config):
        """Test cursor pagination response handling."""
        client = PylonClient(mock_config)

        # Test with next cursor
        response = {"data": [{"id": "1"}], "pagination": {"next_cursor": "cursor123"}}
        data, next_cursor = client._handle_cursor_pagination_response(response)

        assert data == [{"id": "1"}]
        assert next_cursor == "cursor123"

        # Test without next cursor
        response = {"data": [{"id": "1"}], "pagination": {}}
        data, next_cursor = client._handle_cursor_pagination_response(response)

        assert data == [{"id": "1"}]
        assert next_cursor is None

    def test_get_accounts_cursor(self, mock_config):
        """Test cursor-based account retrieval."""
        client = PylonClient(mock_config)

        mock_response = {
            "data": [{"id": "acc1"}],
            "pagination": {"next_cursor": "cursor123"},
        }
        with patch.object(
            client, "_make_request_with_rate_limit", return_value=mock_response
        ):
            result = client.get_accounts_cursor(cursor="cursor123", limit=50)

            assert result == mock_response
            client._make_request_with_rate_limit.assert_called_once_with(
                "GET", "/accounts", params={"limit": 50, "cursor": "cursor123"}
            )

    def test_iter_all_accounts_cursor(self, mock_config):
        """Test cursor-based account iteration."""
        client = PylonClient(mock_config)

        responses = [
            {"data": [{"id": "acc1"}], "pagination": {"next_cursor": "cursor2"}},
            {"data": [{"id": "acc2"}], "pagination": {"next_cursor": "cursor3"}},
            {"data": [{"id": "acc3"}], "pagination": {}},
        ]

        with patch.object(client, "get_accounts_cursor", side_effect=responses):
            accounts = list(client.iter_all_accounts_cursor(batch_size=1))

            assert len(accounts) == 3
            assert accounts[0]["id"] == "acc1"
            assert accounts[1]["id"] == "acc2"
            assert accounts[2]["id"] == "acc3"

    def test_iter_all_accounts_with_rate_limit_handling(self, mock_config):
        """Test account iteration with rate limit error handling."""
        client = PylonClient(mock_config)

        # First call raises rate limit error, second call succeeds
        responses = [
            PylonRateLimitError("Rate limited", retry_after=1),
            {"data": [{"id": "acc1"}], "total": 1},
        ]

        with patch.object(client, "get_accounts", side_effect=responses):
            with patch("time.sleep") as mock_sleep:
                accounts = list(client.iter_all_accounts(batch_size=1))

                assert len(accounts) == 1
                assert accounts[0]["id"] == "acc1"
                mock_sleep.assert_called_once_with(1)

    def test_iter_all_accounts_with_api_error(self, mock_config):
        """Test account iteration with API error propagation."""
        client = PylonClient(mock_config)

        api_error = PylonAPIError("API Error", 500)

        with patch.object(client, "get_accounts", side_effect=api_error):
            with pytest.raises(PylonAPIError):
                list(client.iter_all_accounts(batch_size=1))

    def test_pylon_rate_limit_error(self):
        """Test PylonRateLimitError exception."""
        error = PylonRateLimitError("Rate limited", retry_after=60)

        assert error.message == "Rate limited"
        assert error.retry_after == 60
        assert error.status_code == 429
        assert str(error) == "Rate limited"
