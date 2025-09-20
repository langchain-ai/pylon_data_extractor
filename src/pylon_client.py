"""Pylon API client for data extraction."""

import random
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config, get_config

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Simple rate limiter to prevent hitting API rate limits."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        now = datetime.now()

        # Remove requests older than 1 minute
        cutoff = now - timedelta(minutes=1)
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]

        # If we're at the limit, wait until the oldest request is 1 minute old
        if len(self.requests) >= self.requests_per_minute:
            oldest_request = min(self.requests)
            wait_until = oldest_request + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()

            if wait_seconds > 0:
                logger.info("Rate limiting: waiting", wait_seconds=wait_seconds)
                time.sleep(wait_seconds)

        # Record this request
        self.requests.append(now)


class PylonAPIError(Exception):
    """Exception raised for Pylon API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(message)


class PylonRateLimitError(PylonAPIError):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: int = 60,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, 429, response_data)


class PylonClient:
    """Client for interacting with Pylon API."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.session = requests.Session()

        # Set up rate limiter
        self.rate_limiter = RateLimiter(self.config.pylon.requests_per_minute)

        # Set up authentication
        if self.config.pylon.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.config.pylon.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
            )

    def _calculate_backoff_delay(
        self, attempt: int, retry_after: Optional[int] = None
    ) -> float:
        """Calculate backoff delay with exponential backoff and optional jitter."""
        if retry_after:
            # Use server-specified retry-after if available
            base_delay = retry_after
        else:
            # Use exponential backoff
            base_delay = min(
                self.config.pylon.rate_limit_base_delay * (2**attempt),
                self.config.pylon.rate_limit_max_delay,
            )

        if self.config.pylon.rate_limit_jitter:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.1, 0.3) * base_delay
            return base_delay + jitter

        return base_delay

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=60)
    )
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request to the Pylon API with retry logic."""
        url = self._get_full_url(endpoint)

        logger.info("Making API request", method=method, url=url)

        try:
            response = self.session.request(
                method=method, url=url, timeout=self.config.pylon.timeout, **kwargs
            )

            # Log response details
            logger.info(
                "API response received",
                status_code=response.status_code,
                response_time=response.elapsed.total_seconds(),
            )

            if response.status_code == 429:  # Rate limit
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning("Rate limited, waiting", retry_after=retry_after)
                time.sleep(retry_after)
                raise PylonRateLimitError(
                    f"Rate limited. Retry after {retry_after} seconds", retry_after
                )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as e:
            error_data = {}
            try:
                error_data = response.json()
            except:
                pass

            error_msg = f"HTTP {response.status_code}: {str(e)}"
            if error_data.get("message"):
                error_msg += f" - {error_data['message']}"

            logger.error(
                "API request failed", error=error_msg, status_code=response.status_code
            )
            raise PylonAPIError(error_msg, response.status_code, error_data)

        except requests.exceptions.RequestException as e:
            logger.error("Request failed", error=str(e))
            raise PylonAPIError(f"Request failed: {str(e)}")

    def _get_full_url(self, endpoint: str) -> str:
        """Get the full URL for an API endpoint, handling different API versions."""
        base_url = self.config.pylon.api_base_url.rstrip("/")
        endpoint = endpoint.lstrip("/")

        # All endpoints use the base URL without additional prefixes
        return f"{base_url}/{endpoint}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=300),
        retry=retry_if_exception_type(PylonRateLimitError),
    )
    def _make_request_with_rate_limit(
        self, method: str, endpoint: str, **kwargs
    ) -> Dict[str, Any]:
        """Make a request to the Pylon API with enhanced rate limit handling."""
        # Apply rate limiting before making the request
        self.rate_limiter.wait_if_needed()

        url = self._get_full_url(endpoint)

        logger.info(
            "Making API request with rate limit handling", method=method, url=url
        )

        try:
            response = self.session.request(
                method=method, url=url, timeout=self.config.pylon.timeout, **kwargs
            )

            # Log response details
            logger.info(
                "API response received",
                status_code=response.status_code,
                response_time=response.elapsed.total_seconds(),
            )

            if response.status_code == 429:  # Rate limit
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(
                    "Rate limited, will retry with backoff", retry_after=retry_after
                )
                raise PylonRateLimitError(
                    f"Rate limited. Retry after {retry_after} seconds", retry_after
                )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as e:
            error_data = {}
            try:
                error_data = response.json()
            except:
                pass

            error_msg = f"HTTP {response.status_code}: {str(e)}"
            if error_data.get("message"):
                error_msg += f" - {error_data['message']}"

            logger.error(
                "API request failed", error=error_msg, status_code=response.status_code
            )
            raise PylonAPIError(error_msg, response.status_code, error_data)

        except requests.exceptions.RequestException as e:
            logger.error("Request failed", error=str(e))
            raise PylonAPIError(f"Request failed: {str(e)}")

    def get_accounts(
        self,
        limit: int = 100,
        offset: int = 0,
        updated_since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get accounts from Pylon API."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if updated_since:
            # Ensure timezone-aware datetime for API compatibility
            if updated_since.tzinfo is None:
                # Assume UTC if no timezone info
                from datetime import timezone

                updated_since = updated_since.replace(tzinfo=timezone.utc)
            params["updated_since"] = updated_since.isoformat()

        return self._make_request_with_rate_limit("GET", "/accounts", params=params)

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Get a specific account by ID."""
        return self._make_request_with_rate_limit("GET", f"/accounts/{account_id}")

    def get_issues(
        self, start_time: datetime, end_time: datetime, account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get issues from Pylon API.

        Args:
            start_time: The start time (RFC3339) of the time range to get issues for
            end_time: The end time (RFC3339) of the time range to get issues for
            account_id: Optional account ID filter

        Note:
            The duration between start_time and end_time must be less than or equal to 30 days.
        """
        # Ensure timezone-aware datetimes for API compatibility
        if start_time.tzinfo is None:
            from datetime import timezone

            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            from datetime import timezone

            end_time = end_time.replace(tzinfo=timezone.utc)

        params = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

        if account_id:
            params["account_id"] = account_id

        return self._make_request_with_rate_limit("GET", "/issues", params=params)

    def get_issue(self, issue_id: str) -> Dict[str, Any]:
        """Get a specific issue by ID."""
        return self._make_request_with_rate_limit("GET", f"/issues/{issue_id}")

    def get_issue_messages(
        self,
        issue_id: str,
        limit: int = 100,
        offset: int = 0,
        updated_since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get messages for a specific issue."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if updated_since:
            # Ensure timezone-aware datetime for API compatibility
            if updated_since.tzinfo is None:
                # Assume UTC if no timezone info
                from datetime import timezone

                updated_since = updated_since.replace(tzinfo=timezone.utc)
            params["updated_since"] = updated_since.isoformat()

        return self._make_request_with_rate_limit(
            "GET", f"/issues/{issue_id}/messages", params=params
        )

    def get_contacts(
        self,
        limit: int = 100,
        offset: int = 0,
        updated_since: Optional[datetime] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get contacts from Pylon API."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if updated_since:
            # Ensure timezone-aware datetime for API compatibility
            if updated_since.tzinfo is None:
                # Assume UTC if no timezone info
                from datetime import timezone

                updated_since = updated_since.replace(tzinfo=timezone.utc)
            params["updated_since"] = updated_since.isoformat()
        if account_id:
            params["account_id"] = account_id

        return self._make_request_with_rate_limit("GET", "/contacts", params=params)

    def get_contact(self, contact_id: str) -> Dict[str, Any]:
        """Get a specific contact by ID."""
        return self._make_request_with_rate_limit("GET", f"/contacts/{contact_id}")

    def get_users(
        self,
        limit: int = 100,
        offset: int = 0,
        updated_since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get users from Pylon API."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if updated_since:
            # Ensure timezone-aware datetime for API compatibility
            if updated_since.tzinfo is None:
                # Assume UTC if no timezone info
                from datetime import timezone

                updated_since = updated_since.replace(tzinfo=timezone.utc)
            params["updated_since"] = updated_since.isoformat()

        return self._make_request_with_rate_limit("GET", "/users", params=params)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get a specific user by ID."""
        return self._make_request_with_rate_limit("GET", f"/users/{user_id}")

    def get_teams(
        self,
        limit: int = 100,
        offset: int = 0,
        updated_since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get teams from Pylon API."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if updated_since:
            # Ensure timezone-aware datetime for API compatibility
            if updated_since.tzinfo is None:
                # Assume UTC if no timezone info
                from datetime import timezone

                updated_since = updated_since.replace(tzinfo=timezone.utc)
            params["updated_since"] = updated_since.isoformat()

        return self._make_request_with_rate_limit("GET", "/teams", params=params)

    def get_team(self, team_id: str) -> Dict[str, Any]:
        """Get a specific team by ID."""
        return self._make_request_with_rate_limit("GET", f"/teams/{team_id}")

    def iter_all_accounts(
        self, updated_since: Optional[datetime] = None, batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all accounts with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_accounts,
            pagination_type="offset",
            batch_size=batch_size,
            updated_since=updated_since,
        )

    def iter_all_issues(
        self,
        updated_since: Optional[datetime] = None,
        account_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all issues using time-based chunking.

        Args:
            updated_since: Start time for issues retrieval
            account_id: Optional account ID filter
            batch_size: Not used for issues API, kept for compatibility

        Note:
            The Pylon API requires start_time and end_time parameters with max 30-day range.
            This method chunks the time range to retrieve all issues.
        """
        from datetime import timedelta, timezone

        # Default to last 30 days if no updated_since provided
        if updated_since is None:
            updated_since = datetime.now(timezone.utc) - timedelta(days=30)
        elif updated_since.tzinfo is None:
            updated_since = updated_since.replace(tzinfo=timezone.utc)

        # Current time as end_time
        end_time = datetime.now(timezone.utc)

        # Chunk the time range into 30-day periods (API limit)
        current_start = updated_since
        max_chunk_days = 30

        while current_start < end_time:
            # Calculate chunk end time (max 30 days from start)
            chunk_end = min(current_start + timedelta(days=max_chunk_days), end_time)

            logger.info(
                "Fetching issues for time range",
                start_time=current_start.isoformat(),
                end_time=chunk_end.isoformat(),
                account_id=account_id,
            )

            try:
                response = self.get_issues(
                    start_time=current_start, end_time=chunk_end, account_id=account_id
                )

                data = response.get("data", [])
                for item in data:
                    yield item

                # Move to next chunk
                current_start = chunk_end

            except PylonAPIError as e:
                logger.error(
                    "API error during issues iteration",
                    error=e.message,
                    status_code=e.status_code,
                    start_time=current_start.isoformat(),
                    end_time=chunk_end.isoformat(),
                )
                raise

    def iter_all_issue_messages(
        self,
        issue_id: str,
        updated_since: Optional[datetime] = None,
        batch_size: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all messages for an issue with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_issue_messages,
            pagination_type="offset",
            batch_size=batch_size,
            issue_id=issue_id,
            updated_since=updated_since,
        )

    def iter_all_contacts(
        self,
        updated_since: Optional[datetime] = None,
        account_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all contacts with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_contacts,
            pagination_type="offset",
            batch_size=batch_size,
            updated_since=updated_since,
            account_id=account_id,
        )

    def iter_all_users(
        self, updated_since: Optional[datetime] = None, batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all users with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_users,
            pagination_type="offset",
            batch_size=batch_size,
            updated_since=updated_since,
        )

    def iter_all_teams(
        self, updated_since: Optional[datetime] = None, batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all teams with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_teams,
            pagination_type="offset",
            batch_size=batch_size,
            updated_since=updated_since,
        )

    def _handle_pagination_response(
        self, response: Dict[str, Any], current_offset: int, batch_size: int
    ) -> tuple[List[Dict[str, Any]], bool, int]:
        """Handle pagination response and return data, has_more flag, and next offset."""
        data = response.get("data", [])
        pagination = response.get("pagination", {})

        # Check if there's a next page using the pagination info
        has_more = pagination.get("has_next_page", False)

        # For offset-based pagination, calculate next offset
        next_offset = current_offset + len(data) if has_more else current_offset

        return data, has_more, next_offset

    def _handle_cursor_pagination_response(
        self, response: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Handle cursor-based pagination response and return data and next cursor."""
        data = response.get("data", [])
        next_cursor = response.get("pagination", {}).get("next_cursor")

        return data, next_cursor

    def _fetch_page_with_retry(self, fetch_func: callable, **kwargs) -> Dict[str, Any]:
        """Fetch a single page with proper rate limit handling and retry logic."""

        @retry(
            stop=stop_after_attempt(self.config.pylon.rate_limit_retries),
            wait=wait_exponential(
                multiplier=self.config.pylon.rate_limit_base_delay,
                min=self.config.pylon.rate_limit_base_delay,
                max=self.config.pylon.rate_limit_max_delay,
            ),
            retry=retry_if_exception_type(PylonRateLimitError),
        )
        def _make_request_with_retry():
            return fetch_func(**kwargs)

        return _make_request_with_retry()

    def _paginate_all(
        self,
        fetch_func: callable,
        pagination_type: str = "offset",
        batch_size: int = 100,
        **fetch_kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generic pagination method that handles both offset and cursor-based pagination.

        Args:
            fetch_func: Function to call for each page (e.g., self.get_accounts)
            pagination_type: Either "offset" or "cursor"
            batch_size: Number of items per page
            **fetch_kwargs: Additional arguments to pass to fetch_func

        Yields:
            Individual items from all pages
        """
        if pagination_type == "offset":
            offset = 0

            while True:
                logger.info(
                    "Fetching batch",
                    offset=offset,
                    batch_size=batch_size,
                    **fetch_kwargs,
                )

                try:
                    response = self._fetch_page_with_retry(
                        fetch_func, limit=batch_size, offset=offset, **fetch_kwargs
                    )
                    data, has_more, next_offset = self._handle_pagination_response(
                        response, offset, batch_size
                    )

                    if not data:
                        break

                    for item in data:
                        yield item

                    if not has_more:
                        break

                    offset = next_offset

                except PylonAPIError as e:
                    logger.error(
                        "API error during pagination",
                        error=e.message,
                        status_code=e.status_code,
                    )
                    raise

        elif pagination_type == "cursor":
            cursor = None

            while True:
                logger.info(
                    "Fetching batch with cursor",
                    cursor=cursor,
                    batch_size=batch_size,
                    **fetch_kwargs,
                )

                try:
                    response = self._fetch_page_with_retry(
                        fetch_func, cursor=cursor, limit=batch_size, **fetch_kwargs
                    )
                    data, next_cursor = self._handle_cursor_pagination_response(
                        response
                    )

                    if not data:
                        break

                    for item in data:
                        yield item

                    if not next_cursor:
                        break

                    cursor = next_cursor

                except PylonAPIError as e:
                    logger.error(
                        "API error during pagination",
                        error=e.message,
                        status_code=e.status_code,
                    )
                    raise
        else:
            raise ValueError(f"Unsupported pagination type: {pagination_type}")

    def get_accounts_cursor(
        self,
        cursor: Optional[str] = None,
        limit: int = 100,
        updated_since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get accounts using cursor-based pagination."""
        params = {
            "limit": limit,
        }

        if cursor:
            params["cursor"] = cursor
        if updated_since:
            # Ensure timezone-aware datetime for API compatibility
            if updated_since.tzinfo is None:
                # Assume UTC if no timezone info
                from datetime import timezone

                updated_since = updated_since.replace(tzinfo=timezone.utc)
            params["updated_since"] = updated_since.isoformat()

        return self._make_request_with_rate_limit("GET", "/accounts", params=params)

    def iter_all_accounts_cursor(
        self, updated_since: Optional[datetime] = None, batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all accounts using cursor-based pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_accounts_cursor,
            pagination_type="cursor",
            batch_size=batch_size,
            updated_since=updated_since,
        )

    def close(self) -> None:
        """Close the session."""
        if self.session:
            self.session.close()
