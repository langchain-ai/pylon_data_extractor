"""Classification module for Pylon issues."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from langchain_openai import ChatOpenAI

from .bigquery_utils import BigQueryManager
from .config import Config, get_config
from .pylon_client import PylonClient
from .prompts import (
    CATEGORY_CLASSIFICATION_PROMPT,
    COMBINED_CLASSIFICATION_PROMPT,
    RESOLUTION_CLASSIFICATION_PROMPT,
)

logger = structlog.get_logger(__name__)


class ClassificationError(Exception):
    """Exception raised for classification errors."""
    pass


class PylonClassifier:
    """Classifier for Pylon issues resolution and category."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.bigquery_manager = BigQueryManager(self.config)
        self.pylon_client = PylonClient(self.config)

        # Initialize OpenAI client
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=200
        )

    def classify_issues(
        self,
        fields: List[str] = ["resolution", "category"],
        batch_size: int = 10,
        max_records: Optional[int] = None,
        issue_id: Optional[str] = None,
        created_start: Optional[str] = None,
        created_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify closed issues with missing resolution or category fields."""
        logger.info("Starting issue classification", fields=fields, batch_size=batch_size, max_records=max_records, issue_id=issue_id, created_start=created_start, created_end=created_end)

        # If specific issue_id provided, process only that issue
        if issue_id:
            unclassified_issues = [{"issue_id": issue_id}]
            logger.info("Processing specific issue", issue_id=issue_id)
        else:
            # Get unclassified issues from BigQuery
            unclassified_issues = self._get_unclassified_issues(fields, max_records, created_start, created_end)

            if not unclassified_issues:
                logger.info("No unclassified issues found")
                return {
                    "total_processed": 0,
                    "total_classified": 0,
                    "total_updated": 0,
                    "total_errors": 0,
                    "success": True,
                }

            logger.info("Found unclassified issues", count=len(unclassified_issues))

        total_processed = 0
        total_classified = 0
        total_updated = 0
        total_errors = 0

        # Process issues in batches
        for i in range(0, len(unclassified_issues), batch_size):
            batch = unclassified_issues[i:i + batch_size]

            for issue in batch:
                try:
                    issue_id = issue["issue_id"]
                    logger.info("Processing issue", issue_id=issue_id, progress=f"{total_processed + 1}/{len(unclassified_issues)}")

                    # Get messages for this issue
                    conversation_history = self._get_conversation_history(issue_id)

                    if not conversation_history:
                        logger.warning("No conversation history found", issue_id=issue_id)
                        total_errors += 1
                        total_processed += 1
                        continue

                    # Classify the issue
                    classification = self._classify_issue(conversation_history, fields)

                    if classification and self._should_update(classification):
                        # Update issue via PATCH API
                        if self._update_issue(issue_id, classification):
                            total_classified += 1
                            total_updated += 1
                            logger.info("Issue classified and updated", issue_id=issue_id, classification=classification)
                        else:
                            total_classified += 1
                            logger.warning("Issue classified but update failed", issue_id=issue_id)
                            total_errors += 1
                    else:
                        logger.info("Issue classification skipped due to low confidence", issue_id=issue_id)

                    total_processed += 1

                except Exception as e:
                    logger.error("Error processing issue", issue_id=issue.get("issue_id", "unknown"), error=str(e))
                    total_errors += 1
                    total_processed += 1
                    continue

        result = {
            "total_processed": total_processed,
            "total_classified": total_classified,
            "total_updated": total_updated,
            "total_errors": total_errors,
            "success": total_errors == 0,
        }

        logger.info("Classification completed", **result)
        return result

    def _get_unclassified_issues(self, fields: List[str], max_records: Optional[int] = None, created_start: Optional[str] = None, created_end: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get closed issues with missing resolution or category fields from BigQuery."""
        conditions = []

        if "resolution" in fields:
            conditions.append("JSON_VALUE(data, '$.custom_fields.resolution.value') IS NULL")
        if "category" in fields:
            conditions.append("JSON_VALUE(data, '$.custom_fields.category.value') IS NULL")

        where_clause = " OR ".join(conditions) if conditions else "1=0"
        
        # Add timerange filtering
        time_conditions = []
        if created_start:
            time_conditions.append(f"TIMESTAMP(JSON_VALUE(data, '$.created_at')) >= TIMESTAMP('{created_start}')")
        if created_end:
            time_conditions.append(f"TIMESTAMP(JSON_VALUE(data, '$.created_at')) <= TIMESTAMP('{created_end}')")
        
        time_filter = ""
        if time_conditions:
            time_filter = f"AND ({' AND '.join(time_conditions)})"

        query = f"""
        SELECT
            JSON_VALUE(data, '$.id') as issue_id,
            TIMESTAMP(JSON_VALUE(data, '$.created_at')) as created_at
        FROM `{self.config.bigquery.project_id}.{self.config.bigquery.dataset_id}.int__pylon_issues`
        WHERE JSON_VALUE(data, '$.state') = 'closed'
        AND ({where_clause})
        {time_filter}
        ORDER BY created_at DESC
        """

        if max_records:
            query += f" LIMIT {max_records}"

        logger.info("Querying for unclassified issues", query=query)
        return self.bigquery_manager.query_to_list(query)

    def _get_conversation_history(self, issue_id: str) -> str:
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

        # Format conversation history
        conversation = []
        for msg in messages:
            author = msg.get("author", "Unknown")
            timestamp = msg.get("ts", "")
            body = msg.get("body", "")
            conversation.append(f"[{timestamp}] {author}: {body}")

        return "\n".join(conversation)

    def _classify_issue(self, conversation_history: str, fields: List[str]) -> Optional[Dict[str, Any]]:
        """Classify an issue using OpenAI."""
        response_text = "No response received"
        try:
            # Choose the appropriate prompt
            logger.debug("Building prompt", fields=fields, conversation_length=len(conversation_history))
            if len(fields) == 1:
                if "resolution" in fields:
                    logger.debug("Using resolution classification prompt")
                    prompt = RESOLUTION_CLASSIFICATION_PROMPT.replace("{conversation_history}", conversation_history)
                else:  # category
                    logger.debug("Using category classification prompt")
                    prompt = CATEGORY_CLASSIFICATION_PROMPT.replace("{conversation_history}", conversation_history)
            else:
                logger.debug("Using combined classification prompt")
                prompt = COMBINED_CLASSIFICATION_PROMPT.replace("{conversation_history}", conversation_history)
            
            logger.debug("Prompt built successfully", prompt_length=len(prompt))

            # Call OpenAI
            logger.debug("Calling LLM")
            response = self.llm.invoke(prompt)
            logger.debug("LLM response received", response_type=type(response))
            response_text = response.content
            logger.debug("Raw LLM response", response=response_text)

            # Clean and parse JSON response
            # Find the first occurrence of '{' and last occurrence of '}' to extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                # Extract just the JSON part
                json_text = response_text[json_start:json_end+1]
                try:
                    classification = json.loads(json_text)
                    logger.debug("Cleaned classification response", classification=classification)
                    return classification
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse extracted JSON", error=str(e), json_text=json_text, full_response=response_text)
                    return None
            else:
                logger.error("No JSON object found in LLM response", response=response_text)
                return None

        except json.JSONDecodeError as e:
            logger.error("Failed to parse classification response", error=str(e), response=response_text)
            return None
        except Exception as e:
            logger.error("Error during LLM classification", error=str(e), response=response_text)
            return None

    def _should_update(self, classification: Dict[str, Any]) -> bool:
        """Check if classification confidence meets threshold for updating."""
        confidence_threshold = 0.8

        # Check resolution confidence if present
        if "resolution_confidence" in classification:
            if classification["resolution_confidence"] < confidence_threshold:
                return False
        elif "confidence" in classification:
            if classification["confidence"] < confidence_threshold:
                return False

        # Check category confidence if present
        if "category_confidence" in classification:
            if classification["category_confidence"] < confidence_threshold:
                return False

        return True

    def _update_issue(self, issue_id: str, classification: Dict[str, Any]) -> bool:
        """Update issue via Pylon PATCH API with classification and auto-classified tag."""
        try:
            # Get full issue data for debug logging
            try:
                issue_body = self.pylon_client.get_issue(issue_id)
                issue = issue_body.get('data', 'No body found')
                logger.debug("Issue body before patching", issue_id=issue_id, issue=issue)
            except Exception as e:
                logger.warning("Could not retrieve issue body for debug logging", issue_id=issue_id, error=str(e))

            # Build update payload
            custom_fields = []

            # Add resolution if present and confident
            if "resolution" in classification and classification.get("resolution_confidence", classification.get("confidence", 0)) >= 0.8:
                custom_fields.append({
                    "slug": "resolution",
                    "value": classification["resolution"]
                })

            # Add category if present and confident
            if "category" in classification and classification.get("category_confidence", classification.get("confidence", 0)) >= 0.8:
                custom_fields.append({
                    "slug": "category",
                    "value": classification["category"]
                })

            if not custom_fields:
                logger.info("No custom fields to update", issue_id=issue_id)
                return True

            # Retain existing tags and add "auto-classified" only if not present
            tags = issue.get('tags', [])
            if "auto-classified" not in tags:
                tags.append("auto-classified")

            payload = {
                "custom_fields": custom_fields,
                "tags": tags
            }

            logger.debug("Patching issue", issue_id=issue_id, payload=payload)

            # Use existing method with PATCH - it will handle rate limiting and retries
            self.pylon_client._patch_issue(issue_id, payload)
            logger.debug("Issue updated successfully", issue_id=issue_id, payload=payload)
            return True

        except Exception as e:
            logger.error("Error updating issue", issue_id=issue_id, error=str(e))
            return False

    def close(self) -> None:
        """Close all connections."""
        if self.bigquery_manager:
            self.bigquery_manager.close()
        if self.pylon_client:
            self.pylon_client.close()