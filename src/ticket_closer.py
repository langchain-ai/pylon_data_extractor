"""Ticket closing module for Pylon issues."""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from langchain_openai import ChatOpenAI
from langsmith import traceable

from .bigquery_utils import BigQueryManager
from .config import Config, get_config
from .pylon_client import PylonClient
from .classifier import PylonClassifier
from .prompts import QUESTION_ANALYSIS_PROMPT

logger = structlog.get_logger(__name__)


class TicketClosingError(Exception):
    """Exception raised for ticket closing errors."""
    pass


class PylonTicketCloser:
    """Ticket closer for Pylon issues."""

    def __init__(self, config: Optional[Config] = None, debug: bool = False):
        self.config = config or get_config()
        self.debug = debug
        self.bigquery_manager = BigQueryManager(self.config)
        self.pylon_client = PylonClient(self.config)
        self.classifier = PylonClassifier(self.config, debug=debug)

        # Initialize OpenAI client with cost-optimized model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=500,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    @traceable(name="close_tickets", tags=["pylon"], resource_tags={"component": "issue_closer"})
    def close_tickets(
        self,
        batch_size: int = 10,
        max_records: Optional[int] = None,
        issue_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Close eligible Slack tickets based on question analysis."""
        logger.info("Starting ticket closing process", batch_size=batch_size, max_records=max_records, issue_id=issue_id)

        # If specific issue_id provided, process only that issue
        if issue_id:
            candidate_issues = [{"issue_id": issue_id}]
            logger.info("Processing specific issue", issue_id=issue_id)
        else:
            # Get candidate issues for closing
            candidate_issues = self._get_candidate_issues(max_records)

            if not candidate_issues:
                logger.info("No candidate issues found for closing")
                return {
                    "total_processed": 0,
                    "total_analyzed": 0,
                    "total_classified": 0,
                    "total_closed": 0,
                    "total_errors": 0,
                    "success": True,
                }

            logger.info("Found candidate issues", count=len(candidate_issues))

        total_processed = 0
        total_analyzed = 0
        total_classified = 0
        total_closed = 0
        total_errors = 0

        # Process issues in batches
        for i in range(0, len(candidate_issues), batch_size):
            batch = candidate_issues[i:i + batch_size]

            for issue in batch:
                try:
                    issue_id = issue["issue_id"]
                    logger.info("Processing issue", issue_id=issue_id, progress=f"{total_processed + 1}/{len(candidate_issues)}")

                    # Check if issue is already closed
                    if self._is_issue_closed(issue_id):
                        logger.info("Issue is already closed, skipping", issue_id=issue_id)
                        total_processed += 1
                        continue

                    # Check if issue has resolution/category, classify if missing
                    needs_classification = self._check_missing_classification(issue_id)
                    if needs_classification:
                        classified = self._ensure_classification(issue_id)
                        if classified:
                            total_classified += 1
                            logger.info("Issue classified", issue_id=issue_id)
                    else:
                        logger.debug("Issue already has resolution and category, skipping classification", issue_id=issue_id)

                    # Check if issue meets closing criteria
                    should_close, analysis = self._should_close_issue(issue_id)

                    if analysis:
                        total_analyzed += 1

                        if should_close:
                            # Close the issue
                            if self._close_issue(issue_id):
                                total_closed += 1
                                logger.info("Issue closed successfully", issue_id=issue_id)
                            else:
                                logger.warning("Issue closure failed", issue_id=issue_id)
                                total_errors += 1
                        else:
                            logger.info("Issue not eligible for closing - questions not all resolved", issue_id=issue_id)
                    else:
                        logger.warning("Question analysis failed", issue_id=issue_id)
                        total_errors += 1

                    total_processed += 1

                except Exception as e:
                    logger.error("Error processing issue", issue_id=issue.get("issue_id", "unknown"), error=str(e))
                    total_errors += 1
                    total_processed += 1
                    continue

        result = {
            "total_processed": total_processed,
            "total_analyzed": total_analyzed,
            "total_classified": total_classified,
            "total_closed": total_closed,
            "total_errors": total_errors,
            "success": total_errors == 0,
        }

        logger.info("Ticket closing completed", **result)
        return result

    def _get_candidate_issues(self, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get issues that are candidates for closing from BigQuery."""
        # Issues must be:
        # 1. In "on customer" state (waiting_on_customer)
        # 2. Last message > 1 week ago
        # 3. From Slack source
        # Note: We'll check resolution/category during processing and classify if missing
        one_week_ago = (datetime.now() - timedelta(weeks=1)).isoformat()

        query = f"""
        WITH issue_last_message AS (
            SELECT
                i.issue_id,
                i.data as issue_data,
                MAX(TIMESTAMP(JSON_VALUE(m.data, '$.timestamp'))) as last_message_time
            FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.int__pylon_issues` i
            LEFT JOIN `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.int__pylon_messages` m
                ON i.issue_id = m.issue_id
            WHERE JSON_VALUE(i.data, '$.state') = 'waiting_on_customer'
            AND JSON_VALUE(i.data, '$.source') = 'slack'
            AND JSON_VALUE(i.data, '$.state') != 'closed'
            GROUP BY i.issue_id, i.data
        )
        SELECT
            issue_id,
            last_message_time
        FROM issue_last_message
        WHERE last_message_time < TIMESTAMP('{one_week_ago}')
        ORDER BY last_message_time ASC
        """

        if max_records:
            query += f" LIMIT {max_records}"

        logger.info("Querying for candidate issues", query=query)
        return self.bigquery_manager.query_to_list(query)

    def _is_issue_closed(self, issue_id: str) -> bool:
        """Check if an issue is already closed."""
        try:
            issue_body = self.pylon_client.get_issue(issue_id)
            issue = issue_body.get('data', {})
            state = issue.get('state')

            is_closed = state == 'closed'
            logger.debug("Issue state check", issue_id=issue_id, state=state, is_closed=is_closed)
            return is_closed

        except Exception as e:
            logger.error("Error checking if issue is closed", issue_id=issue_id, error=str(e))
            # If we can't check the state, assume it's not closed to be safe
            return False

    def _check_missing_classification(self, issue_id: str) -> bool:
        """Check if issue is missing resolution or category fields."""
        try:
            # Get current issue data
            issue_body = self.pylon_client.get_issue(issue_id)
            issue = issue_body.get('data', {})

            resolution = issue.get('custom_fields', {}).get('resolution', {}).get('value')
            category = issue.get('custom_fields', {}).get('category', {}).get('value')

            # Return True if either field is missing
            missing_fields = []
            if not resolution:
                missing_fields.append('resolution')
            if not category:
                missing_fields.append('category')

            if missing_fields:
                logger.debug("Issue missing classification fields", issue_id=issue_id, missing_fields=missing_fields)
                return True
            else:
                logger.debug("Issue has both resolution and category", issue_id=issue_id)
                return False

        except Exception as e:
            logger.error("Error checking classification status", issue_id=issue_id, error=str(e))
            # If we can't check, assume classification is needed to be safe
            return True

    def _ensure_classification(self, issue_id: str) -> bool:
        """Ensure issue has resolution and category, classify if missing."""
        try:
            # Get current issue data
            issue_body = self.pylon_client.get_issue(issue_id)
            issue = issue_body.get('data', {})

            resolution = issue.get('custom_fields', {}).get('resolution', {}).get('value')
            category = issue.get('custom_fields', {}).get('category', {}).get('value')

            # If both are present, no classification needed
            if resolution and category:
                logger.debug("Issue already has resolution and category", issue_id=issue_id)
                return False

            # Determine which fields need classification
            fields_to_classify = []
            if not resolution:
                fields_to_classify.append('resolution')
            if not category:
                fields_to_classify.append('category')

            logger.info("Issue missing classification fields", issue_id=issue_id, missing_fields=fields_to_classify)

            # Run classification for missing fields
            result = self.classifier.classify_issues(
                fields=fields_to_classify,
                batch_size=1,
                max_records=1,
                issue_id=issue_id
            )

            return result.get('total_updated', 0) > 0

        except Exception as e:
            logger.error("Error ensuring classification", issue_id=issue_id, error=str(e))
            return False

    def _should_close_issue(self, issue_id: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Determine if an issue should be closed based on question analysis."""
        try:
            # First verify issue has resolution and category
            issue_body = self.pylon_client.get_issue(issue_id)
            issue = issue_body.get('data', {})

            resolution = issue.get('custom_fields', {}).get('resolution', {}).get('value')
            category = issue.get('custom_fields', {}).get('category', {}).get('value')

            if not resolution or not category:
                logger.warning("Issue still missing resolution or category after classification",
                             issue_id=issue_id, has_resolution=bool(resolution), has_category=bool(category))
                return False, None

            # Get conversation history
            conversation_history = self._get_conversation_history(issue_id)

            if not conversation_history:
                logger.warning("No conversation history found", issue_id=issue_id)
                return False, None

            # Analyze questions in the conversation
            analysis = self._analyze_questions(conversation_history, issue_id)

            if not analysis:
                logger.warning("Question analysis failed", issue_id=issue_id)
                return False, None

            # Check if all questions are resolved
            all_resolved = analysis.get("all_resolved", False)

            logger.info("Question analysis completed",
                       issue_id=issue_id,
                       all_resolved=all_resolved,
                       question_count=len(analysis.get("questions", [])))

            return all_resolved, analysis

        except Exception as e:
            logger.error("Error in should_close_issue", issue_id=issue_id, error=str(e))
            return False, None

    def _get_conversation_history(self, issue_id: str, max_messages: int = 20) -> str:
        """Get conversation history for an issue from BigQuery."""
        query = f"""
        SELECT
            issue_id,
            JSON_VALUE(data, '$.id') as message_id,
            COALESCE(
                JSON_VALUE(data, '$.author.user.email'),
                JSON_VALUE(data, '$.author.contact.email')
            ) as author,
            JSON_VALUE(data, '$.timestamp') as ts,
            JSON_VALUE(data, '$.message_html') as body
        FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.int__pylon_messages`
        WHERE issue_id = '{issue_id}'
        ORDER BY ts ASC
        """

        messages = self.bigquery_manager.query_to_list(query)

        if not messages:
            return ""

        # Limit messages to reduce token usage
        if len(messages) > max_messages:
            # Take first few and last few messages
            messages = messages[:5] + messages[-15:]
            logger.debug(
                "Truncated conversation history",
                issue_id=issue_id,
                original_count=len(messages),
                truncated_to=max_messages
            )

        # Format conversation history
        conversation = []
        for msg in messages:
            author = msg.get("author", "Unknown")
            timestamp = msg.get("ts", "")
            body = msg.get("body", "")

            # Strip HTML tags to minimize tokens
            clean_body = self._strip_html(body)

            conversation.append(f"[{timestamp}] {author}: {clean_body}")

        return "\n".join(conversation)

    def _strip_html(self, html_text: str) -> str:
        """Remove HTML tags and decode entities to minimize token usage."""
        if not html_text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_text)

        # Decode common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    @traceable(name="analyze_questions", tags=["pylon"], resource_tags={"component": "issue_closer"})
    def _analyze_questions(self, conversation_history: str, pylon_issue_id: str) -> Optional[Dict[str, Any]]:
        """Analyze questions in the conversation using OpenAI."""
        try:
            logger.debug("Building question analysis prompt", conversation_length=len(conversation_history))

            # Split prompt into cacheable system message and user message
            system_message = QUESTION_ANALYSIS_PROMPT.replace("{conversation_history}", "").replace("{pylon_issue_id}", "").strip()

            logger.debug("Question analysis prompt built successfully", system_length=len(system_message))

            # Call OpenAI with JSON mode
            logger.debug("Calling LLM for question analysis")
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"<Conversation History>\n{conversation_history}\n</Conversation History>\n\nPylon Issue ID: {pylon_issue_id}"}
            ]
            response = self.llm.invoke(messages)
            logger.debug("LLM response received", response_type=type(response))

            # Parse JSON response
            analysis = json.loads(response.content)
            logger.debug("Question analysis response", analysis=analysis)
            return analysis

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e), response=response.content if hasattr(response, 'content') else "No response")
            return None
        except Exception as e:
            logger.error("Error during question analysis", error=str(e), error_type=type(e).__name__)
            return None

    def _close_issue(self, issue_id: str) -> bool:
        """Close the issue via Pylon PATCH API."""
        try:
            # Get current issue data for tags and logging
            try:
                issue_body = self.pylon_client.get_issue(issue_id)
                issue = issue_body.get('data', {})
                logger.debug("Issue body before closing", issue_id=issue_id, state=issue.get('state'))
            except Exception as e:
                logger.warning("Could not retrieve issue body for debug logging", issue_id=issue_id, error=str(e))
                issue = {}

            # Retain existing tags and add "auto-closed" only if not present
            tags = issue.get('tags', [])
            if "auto-closed" not in tags:
                tags.append("auto-closed")

            # Build payload to close the issue
            payload = {
                "state": "closed",
                "tags": tags
            }

            logger.debug("Closing issue", issue_id=issue_id, payload=payload)

            # Use existing method with PATCH - it will handle rate limiting and retries
            self.pylon_client._patch_issue(issue_id, payload)
            logger.debug("Issue closed successfully", issue_id=issue_id)
            return True

        except Exception as e:
            logger.error("Error closing issue", issue_id=issue_id, error=str(e))
            return False

    def close(self) -> None:
        """Close all connections."""
        if self.bigquery_manager:
            self.bigquery_manager.close()
        if self.pylon_client:
            self.pylon_client.close()
        if self.classifier:
            self.classifier.close()