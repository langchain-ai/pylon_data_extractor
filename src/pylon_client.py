"""Pylon API client for data extraction."""

import random
import threading
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


class RespectRetryAfterWait:
    """Custom wait strategy that respects API retry_after headers with gentle exponential backoff."""

    def __init__(self, base_delay: int = 60, multiplier: float = 1.1, max_delay: int = 300):
        self.base_delay = base_delay
        self.multiplier = multiplier
        self.max_delay = max_delay

    def __call__(self, retry_state):
        # Check if we have a retry_after value from the exception
        if retry_state.outcome and retry_state.outcome.exception():
            exception = retry_state.outcome.exception()
            if isinstance(exception, PylonRateLimitError) and hasattr(exception, 'retry_after'):
                # Use the server's retry_after value directly
                return exception.retry_after

        # Fall back to gentle exponential backoff starting from base_delay
        attempt = retry_state.attempt_number
        delay = min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        return delay


class ConservativeRateLimiter:
    """Conservative rate limiter that prevents bursts and maintains steady spacing."""

    def __init__(self, requests_per_minute: int = 40):
        # Use a very conservative limit to avoid 429s completely
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # Minimum seconds between requests
        self.last_request_time = 0
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits with steady spacing."""
        with self._lock:
            now = time.time()

            # Calculate time since last request
            time_since_last = now - self.last_request_time

            # If we need to wait, do so
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.debug("Rate limiting: waiting", wait_seconds=wait_time)
                time.sleep(wait_time)
                now = time.time()

            # Record this request time
            self.last_request_time = now


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

        # Set up rate limiter - use conservative limit to avoid 429s
        conservative_limit = min(40, self.config.pylon.requests_per_minute)
        self.rate_limiter = ConservativeRateLimiter(conservative_limit)

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

    def _get_full_url(self, endpoint: str) -> str:
        """Get the full URL for an API endpoint, handling different API versions."""
        base_url = self.config.pylon.api_base_url.rstrip("/")
        endpoint = endpoint.lstrip("/")

        # All endpoints use the base URL without additional prefixes
        return f"{base_url}/{endpoint}"

    @retry(
        stop=stop_after_attempt(5),
        wait=RespectRetryAfterWait(base_delay=60, multiplier=1.1, max_delay=300),
        retry=retry_if_exception_type(PylonRateLimitError),
    )
    def _make_request_with_rate_limit(
        self, method: str, endpoint: str, **kwargs
    ) -> Dict[str, Any]:
        """Make a request to the Pylon API with enhanced rate limit handling."""
        # Apply rate limiting before making the request
        self.rate_limiter.wait_if_needed()

        url = self._get_full_url(endpoint)

        logger.debug(
            "Making API request", method=method, url=url, kwargs=kwargs
        )

        try:
            response = self.session.request(
                method=method, url=url, timeout=self.config.pylon.timeout, **kwargs
            )

            # Log response details
            logger.debug(
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
            logger.debug("Request failed", error=str(e))
            raise PylonAPIError(f"Request failed: {str(e)}")

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Get a specific account by ID."""
        return self._make_request_with_rate_limit("GET", f"/accounts/{account_id}")

    def search_issues(self, body: Dict[str, Any], cursor: Optional[str] = None) -> Dict[str, Any]:
        """Search issues using POST /issues/search with a filter body.

        The body should follow the API format, for example:
        {"filter": {"field": "modified_at", "operator": "time_range", "values": [start, end]}}
        or compound filters like {"filter": {"and": [ ... ]}}.

        Args:
            body: The search filter body
            cursor: Optional cursor for pagination
        """
        request_body = dict(body)
        if cursor:
            request_body["cursor"] = cursor
        return self._make_request_with_rate_limit("POST", "/issues/search", json=request_body)

    def get_issue(self, issue_id: str) -> Dict[str, Any]:
        """Get a specific issue by ID."""
        return self._make_request_with_rate_limit("GET", f"/issues/{issue_id}")

    def get_issue_messages(self, issue_id: str) -> Dict[str, Any]:
        """Get all messages for a specific issue."""
        return self._make_request_with_rate_limit(
            "GET", f"/issues/{issue_id}/messages"
        )

    def _patch_issue(self, issue_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an issue via PATCH request."""
        return self._make_request_with_rate_limit("PATCH", f"/issues/{issue_id}", json=data)

    def get_contacts(
        self,
        limit: int = 100,
        offset: int = 0,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get contacts from Pylon API."""
        params = {
            "limit": limit,
            "offset": offset,
        }

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
    ) -> Dict[str, Any]:
        """Get users from Pylon API."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        return self._make_request_with_rate_limit("GET", "/users", params=params)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get a specific user by ID."""
        return self._make_request_with_rate_limit("GET", f"/users/{user_id}")

    def get_teams(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get teams from Pylon API."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        return self._make_request_with_rate_limit("GET", "/teams", params=params)

    def get_team(self, team_id: str) -> Dict[str, Any]:
        """Get a specific team by ID."""
        return self._make_request_with_rate_limit("GET", f"/teams/{team_id}")

    def iter_all_accounts(
        self, batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all accounts with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_accounts,
            pagination_type="cursor",
            batch_size=batch_size,
        )

    def iter_all_issues(
        self,
        created_start: Optional[datetime] = None,
        created_end: Optional[datetime] = None,
        states: Optional[List[str]] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through issues using POST /issues/search with cursor-based pagination.

        - created_start/created_end build a modified_at filter with operators:
          time_is_after, time_is_before, or time_range
        - states list builds a state filter with operator equals/in
        - extra_filters can be either a full body with "filter"/"filters" or a
          simple mapping which will be transformed into equals filters.
        """
        from datetime import timedelta, timezone

        def build_filter_body(range_start: Optional[datetime], range_end: Optional[datetime]) -> Dict[str, Any]:
            filters: List[Dict[str, Any]] = []

            # modified_at filter
            if range_start or range_end:
                # Ensure timezone-aware datetimes
                rs = range_start
                re = range_end
                if rs and rs.tzinfo is None:
                    rs = rs.replace(tzinfo=timezone.utc)
                if re and re.tzinfo is None:
                    re = re.replace(tzinfo=timezone.utc)

                if rs and re:
                    filters.append(
                        {
                            "field": "modified_at",
                            "operator": "time_range",
                            "values": [rs.isoformat(), re.isoformat()],
                        }
                    )
                elif rs:
                    filters.append(
                        {
                            "field": "modified_at",
                            "operator": "time_is_after",
                            "values": [rs.isoformat()],
                        }
                    )
                elif re:
                    filters.append(
                        {
                            "field": "modified_at",
                            "operator": "time_is_before",
                            "values": [re.isoformat()],
                        }
                    )

            # state filter
            if states:
                op = "equals" if len(states) == 1 else "in"
                filters.append(
                    {
                        "field": "state",
                        "operator": op,
                        "values": states,
                    }
                )

            # extra filters
            if extra_filters:
                # Pass-through if already shaped
                if any(k in extra_filters for k in ("filter", "filters")):
                    body = dict(extra_filters)
                    # Merge constructed filters with provided when possible
                    if filters:
                        constructed = {"and": filters} if len(filters) > 1 else filters[0]
                        if "filter" in body:
                            body["filter"] = {"and": [body["filter"], constructed]}
                        elif "filters" in body:
                            body["filters"].extend(filters)
                        else:
                            body["filter"] = constructed
                    return body

                # Transform mapping into equals filters
                for k, v in extra_filters.items():
                    # Allow simple scalars or lists => equals/in
                    if isinstance(v, list):
                        filters.append({"field": k, "operator": "in", "values": v})
                    else:
                        filters.append({"field": k, "operator": "equals", "values": [v]})

            # Build final body
            if not filters:
                return {}
            if len(filters) == 1:
                return {"filter": filters[0]}
            return {"filter": {"and": filters}}

        def paginate_search_results(body: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
            """Paginate through search results using cursor-based pagination."""
            cursor = None

            while True:
                logger.info(
                    "Searching issues with cursor",
                    cursor=cursor,
                    has_filters=bool(body),
                )

                try:
                    response = self.search_issues(body, cursor=cursor)
                    data = response.get("data", [])
                    pagination = response.get("pagination", {})

                    if not data:
                        break

                    for item in data:
                        yield item

                    # Check if there are more pages
                    if not pagination.get("has_next_page", False):
                        break

                    cursor = pagination.get("cursor")
                    if not cursor:
                        break

                except PylonAPIError as e:
                    logger.error(
                        "API error during issues search iteration",
                        error=e.message,
                        status_code=e.status_code,
                        cursor=cursor,
                    )
                    raise

        # If created range exceeds 30 days, chunk the time range
        if created_start and created_end:
            rs = created_start
            re = created_end
            if rs.tzinfo is None:
                rs = rs.replace(tzinfo=timezone.utc)
            if re.tzinfo is None:
                re = re.replace(tzinfo=timezone.utc)

            current_start = rs
            max_chunk_days = 30
            while current_start < re:
                chunk_end = min(current_start + timedelta(days=max_chunk_days), re)

                body = build_filter_body(current_start, chunk_end)
                logger.info(
                    "Processing time chunk",
                    start_time=current_start.isoformat(),
                    end_time=chunk_end.isoformat(),
                )

                # Use cursor pagination for this time chunk
                yield from paginate_search_results(body)
                current_start = chunk_end
            return

        # Single filter request with cursor pagination
        body = build_filter_body(created_start, created_end)
        logger.info("Processing issues search with cursor pagination", has_filters=bool(body))
        yield from paginate_search_results(body)

    def iter_all_issue_messages(
        self, issue_id: str
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all messages for an issue."""
        try:
            response = self._fetch_page_with_retry(
                self.get_issue_messages, issue_id=issue_id
            )

            # Handle case where response might be None
            if response is None:
                logger.warning("Received None response for issue messages", issue_id=issue_id)
                return

            data = response.get("data", [])

            # Handle case where data might be None
            if data is None:
                logger.warning("Received None data for issue messages", issue_id=issue_id)
                return

            for item in data:
                yield item
        except PylonAPIError as e:
            logger.error(
                "API error during issue messages retrieval",
                error=e.message,
                status_code=e.status_code,
                issue_id=issue_id,
            )
            raise

    def iter_all_contacts(
        self,
        account_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all contacts with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_contacts,
            pagination_type="offset",
            batch_size=batch_size,
            account_id=account_id,
        )

    def iter_all_users(
        self, batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all users with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_users,
            pagination_type="offset",
            batch_size=batch_size,
        )

    def iter_all_teams(
        self, batch_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through all teams with pagination."""
        yield from self._paginate_all(
            fetch_func=self.get_teams,
            pagination_type="offset",
            batch_size=batch_size,
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
        pagination = response.get("pagination", {})
        next_cursor = pagination.get("cursor") if pagination.get("has_next_page", False) else None

        return data, next_cursor

    def _fetch_page_with_retry(self, fetch_func: callable, **kwargs) -> Dict[str, Any]:
        """Fetch a single page with proper rate limit handling and retry logic."""

        @retry(
            stop=stop_after_attempt(self.config.pylon.rate_limit_retries),
            wait=RespectRetryAfterWait(
                base_delay=60,
                multiplier=1.1,
                max_delay=self.config.pylon.rate_limit_max_delay,
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

    def get_accounts(
        self,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get accounts using cursor-based pagination."""
        params = {
            "limit": limit,
        }

        if cursor:
            params["cursor"] = cursor

        return self._make_request_with_rate_limit("GET", "/accounts", params=params)

    

    def close(self) -> None:
        """Close the session."""
        if self.session:
            self.session.close()
