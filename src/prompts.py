"""Classification prompts for Pylon issues.

This module exposes three public constants:
- RESOLUTION_CLASSIFICATION_PROMPT
- CATEGORY_CLASSIFICATION_PROMPT
- COMBINED_CLASSIFICATION_PROMPT

Internally, they are composed from shared sections to keep instructions
consistent and ensure the conversation history is included only once.
"""

# Shared sections (internal)
_SCOPE_RESOLUTION = (
    "Scope: You are an expert customer support analyst. Based on the conversation "
    "history provided, classify the resolution type for this closed support issue."
)

_IMPORTANT_RULES = """
<Important Rules>
Identify LangChain vs customer based on the email address. All langchain emails are from langchain.dev
Weigh the initial few messages and the last few messages higher than the middle messages.
Weigh the LangChain emails higher than the customer emails to determine resolution
Base your classification on the actual resolution described in the conversation, not just the final outcome.
If multiple issues or topics are discussed, prioritize the most important one.
</Important Rules>
""".strip()

_RESOLUTION_CATEGORIES = {
    "bug_fix": {"description": "LangChain Bug fixes or version releases we make", "examples": "we release new version, fixed customer issue"},
    "feature_request": {"description": "New functionality added or requested", "examples": "acknowledged feature request, included in release"},
    "referred_to_public_resource": {"description": "Answered using existing docs", "examples": "pointed to KB article, existing documentation"},
    "referred_with_internal_knowledge": {"description": "Answered but needs new docs", "examples": "internal knowledge shared, should create/update article"},
    "customer_love": {"description": "Customer management activities", "examples": "meetings with DE, sales interactions"},
    "no_action": {"description": "No actionable work required", "examples": "spam, simple questions, no follow-up needed"}
}

_OUTPUT_RESOLUTION = """
Return a JSON object with: resolution (string - must be exact key from options) and confidence (number 0.0-1.0).
""".strip()

_OUTPUT_RESOLUTION_DEBUG = """
Return a JSON object with: resolution (string - must be exact key from options), confidence (number 0.0-1.0), and resolution_reason (one sentance on how the decision was made).
""".strip()

_CONVERSATION_HISTORY = """
<Conversation History>
{conversation_history}
</Conversation History>
""".strip()

# Category specific sections (internal)
_SCOPE_CATEGORY = (
    "Scope: You are an expert customer support analyst. Based on the conversation "
    "history provided, classify the category type for this support issue."
)

_CATEGORY_CATEGORIES = {
    "admin_authentication_access": {"description": "Login/access issues", "examples": "password resets, SSO, MFA problems"},
    "admin_billing_refunds": {"description": "Billing and payment issues", "examples": "subscription changes, refunds, payment problems"},
    "admin_data_deletion": {"description": "Data deletion requests", "examples": "GDPR removal, account closure, DSAR"},
    "admin_general_account_management": {"description": "Account administration", "examples": "adding users, project migration, settings"},
    "admin_security_privacy_and_compliance": {"description": "Security/compliance requests", "examples": "SOC 2, HIPAA, GDPR docs"},
    "langchain_oss_python": {"description": "LangChain Python library", "examples": "bugs, imports, GitHub issues"},
    "anggraph_platform": {"description": "LangGraph cloud platform", "examples": "deployment, agents, scaling issues"},
    "langgraph_studio": {"description": "LangGraph Studio app", "examples": "desktop app, visualization tools"},
    "langsmith_administration": {"description": "LangSmith org settings", "examples": "permissions, admin controls"},
    "langsmith_annotations": {"description": "Annotation workflows", "examples": "annotation queues, feedback"},
    "langsmith_automation_rules": {"description": "Automation features", "examples": "triggers, webhooks, rules"},
    "langsmith_dashboards": {"description": "LangSmith UI issues", "examples": "web UI, search, navigation"},
    "langsmith_datasets_experiments": {"description": "Datasets and experiments", "examples": "evaluations, test runs"},
    "langsmith_evaluation": {"description": "Model evaluation", "examples": "scoring outputs, feedback collection"},
    "langsmith_observability": {"description": "Monitoring and traces", "examples": "trace analysis, metrics, alerts"},
    "langsmith_playground": {"description": "LangSmith Playground", "examples": "prompt testing, model issues"},
    "langsmith_pricing": {"description": "Pricing questions", "examples": "plan details, cost optimization"},
    "langsmith_prompt_hub": {"description": "Prompt management", "examples": "versioning, collaboration"},
    "langsmith_sdk": {"description": "SDK and integrations", "examples": "tracing setup, API usage"},
    "langsmith_insights": {"description": "Automated insights (CLIO)", "examples": "insights features"},
    "langgraph_oss_js": {"description": "LangGraph JS library", "examples": "JS bugs, GitHub issues"},
    "langchain_oss_js": {"description": "LangChain JS library", "examples": "JS installation, usage"},
    "other_partnerships": {"description": "Business partnerships", "examples": "integrations, collaborations"},
    "other_sales": {"description": "Sales inquiries", "examples": "enterprise demo, self-hosting"},
    "other_spam": {"description": "Spam/irrelevant", "examples": "marketing, mass outreach"},
    "other_marketing_promotional": {"description": "Marketing content", "examples": "promotional messages, link sharing"}
}

_OUTPUT_CATEGORY = """
Return a JSON object with: category (string - must be exact key from options) and confidence (number 0.0-1.0).
""".strip()

_OUTPUT_CATEGORY_DEBUG = """
Return a JSON object with: category (string - must be exact key from options), confidence (number 0.0-1.0) and category_reason (one sentance on how the decision was made).
""".strip()

# Public prompts composed from shared sections
RESOLUTION_CLASSIFICATION_PROMPT = f"""
{_SCOPE_RESOLUTION}

{_IMPORTANT_RULES}

<Resolution Categories>
{_RESOLUTION_CATEGORIES}

{_OUTPUT_RESOLUTION}

{_CONVERSATION_HISTORY}
"""

RESOLUTION_CLASSIFICATION_PROMPT_DEBUG = f"""
{_SCOPE_RESOLUTION}

{_IMPORTANT_RULES}

<Resolution Categories>
{_RESOLUTION_CATEGORIES}

{_OUTPUT_RESOLUTION_DEBUG}

{_CONVERSATION_HISTORY}
"""

CATEGORY_CLASSIFICATION_PROMPT = f"""
{_SCOPE_CATEGORY}
Base your classification on the primary nature of the customer's issue or request.
If multiple issues or topics are discussed, prioritize the most important one.

{_IMPORTANT_RULES}

<Categories>
{_CATEGORY_CATEGORIES}
</Categories>


{_OUTPUT_CATEGORY}

{_CONVERSATION_HISTORY}
"""

CATEGORY_CLASSIFICATION_PROMPT_DEBUG = f"""
{_SCOPE_CATEGORY}
Base your classification on the primary nature of the customer's issue or request.
If multiple issues or topics are discussed, prioritize the most important one.

{_IMPORTANT_RULES}

<Categories>
{_CATEGORY_CATEGORIES}
</Categories>


{_OUTPUT_CATEGORY_DEBUG}

{_CONVERSATION_HISTORY}
"""

COMBINED_CLASSIFICATION_PROMPT = f"""
Scope: You are an expert customer support analyst. Based on the conversation history provided, classify both the resolution type and the category for this closed support issue.

{_IMPORTANT_RULES}

<Resolutions>
{_RESOLUTION_CATEGORIES}
</Resolutions>

<Categories>
{_CATEGORY_CATEGORIES}
</Categories>

Return a JSON object with: resolution (string - exact key), resolution_confidence (number 0.0-1.0), category (string - exact key), category_confidence (number 0.0-1.0).

{_CONVERSATION_HISTORY}
"""

COMBINED_CLASSIFICATION_PROMPT_DEBUG = f"""
Scope: You are an expert customer support analyst. Based on the conversation history provided, classify both the resolution type and the category for this closed support issue.

{_IMPORTANT_RULES}

<Resolutions>
{_RESOLUTION_CATEGORIES}
</Resolutions>

<Categories>
{_CATEGORY_CATEGORIES}
</Categories>

Return a JSON object with: resolution (string - exact key), resolution_confidence (number 0.0-1.0), resolution_reason (one sentence), category (string - exact key), category_confidence (number 0.0-1.0), category_reason (one sentence).

{_CONVERSATION_HISTORY}
"""

# Question analysis prompt for ticket closing
QUESTION_ANALYSIS_PROMPT = """
Scope: You are an expert customer support analyst. Based on the conversation history provided, extract all customer questions and determine if they have been resolved.

<Important Rules>
Identify LangChain vs customer based on the email address. All langchain emails are from langchain.dev
Focus only on actual questions asked by the customer (non-LangChain participants).
A question is resolved if there's a clear answer or solution provided by LangChain team members.
Questions can be explicit (with ?) or implicit requests for help/information.
Ignore purely conversational or acknowledgment messages.
Extract a simple summary of each question, not the full text.
</Important Rules>

<Output Format>
Return a JSON object with:
- pylon_issue_id (string): The Pylon issue ID provided
- questions (array): Each question object with:
  - id (number): Sequential ID starting from 1
  - question (string): Simple summary of the customer question
  - resolved (boolean): Whether this question was answered/resolved
- quantity (integer): Number of questions in the array
- all_resolved (boolean): Whether ALL customer questions were resolved

Example:
{
  "pylon_issue_id": "abc-123",
  "questions": [
    {"id": 1, "question": "How to configure LangChain with custom model", "resolved": true},
    {"id": 2, "question": "Why is deployment failing", "resolved": false}
  ],
  "quantity": "2",
  "all_resolved": false
}
</Output Format>

{_CONVERSATION_HISTORY}
"""