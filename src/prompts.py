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

_RESOLUTION_CATEGORIES = """
<Resolution Categories>
Please classify the resolution into one of these categories:
big_fix - any bug fix we need to make. Assign this tag anytime we said we released a new version for the customer to try unless it was explicitly stated as a feature request
feature_request - whenever net new functionality is acknowledged by a LangChain team member to be added or included into a new version that we released
referred_to_public_resource - there’s a doc/KB article with the answer and we can point to it. Ideally this serves as a foundation to deflect cases
referred_with_internal_knowledge - an opportunity to create new articles in the future
customer_love - general customer management activities (meetings w DE, Sales)
no_action - (Does not warrant any real repeatable work docs or process improvement or spam)
</Resolution Categories>
""".strip()

_OUTPUT_RESOLUTION = """
Return a JSON object with: resolution (string) and confidence (number 0.0-1.0).
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

_CATEGORY_CATEGORIES = """
<Category Types>
Please classify the category into one of these types:
admin_authentication_access - Categorize any ticket where the user is reporting login failures, password resets, SSO or MFA problems, or cannot access their account.
admin_billing_refunds - Categorize any ticket where the user is asking about billing issues, subscription changes, plan upgrades or downgrades, adding seats, payment problems, or refund requests.
admin_data_deletion - Categorize any ticket where the user is requesting deletion of their account, GDPR data removal, DSAR, privacy-related erasure, or account closure.
admin_general_account_management - Categorize any ticket about adding or removing users, migrating projects or data, changing account settings, or general account administration.
admin_security_privacy_and_compliance - Categorize any ticket where the user requests SOC 2, HIPAA, GDPR, DPA documentation, asks about compliance policies, or reports a security concern.
langchain_oss_python - Categorize any ticket where the user is asking about the LangGraph open-source Python library, including bugs, import errors, or feature usage. If the issue mentions a GitHub issue or links to the LangGraph Python GitHub repo, and it does not mention LangSmith, Evaluation, or traces, then it belongs here.
langgraph_langgraph_platform - Categorize any ticket where the user reports problems with the LangGraph managed cloud or platform, including runtime issues, scaling, deployment errors, infrastructure usage, or compute configuration. Keywords include Deployment, Agents, Assistants
langgraph_studio - Categorize any ticket where the user reports issues with LangGraph Studio desktop application or SaaS-based visualization tools for agent graphs.
langsmith_administration - Categorize any ticket where the user is asking about organization-wide settings, permissions, or administrative controls in LangSmith.
langsmith_annotations - Categorize any ticket where the user is asking about annotation queues, managing annotation tasks, or attaching feedback to traces or runs.
langsmith_automation_rules - Categorize any ticket where the user is asking about automation rules, setting triggers, adding runs to datasets, creating annotation queues, or firing webhooks.
langsmith_dashboards - Categorize any ticket where the user reports problems with LangSmith’s web UI, search functionality, navigation, or custom dashboard features.
langsmith_datasets_experiments - Categorize any ticket where the user is asking about creating datasets, running experiments, structured evaluations, or debugging LLM test runs.
langsmith_evaluation - Categorize any ticket where the user is asking how to evaluate their application on production traffic, score model outputs, or collect human feedback. 
langsmith_observability - Categorize any ticket where the user is analyzing traces, monitoring metrics, or configuring dashboards and alerts in LangSmith.
langsmith_playground - Categorize any ticket where the user reports issues inside the LangSmith Playground for testing prompts or models.
langsmith_pricing - Categorize any ticket where the user is asking about LangSmith pricing, plan details, feature comparisons, or cost optimization advice.
langsmith_prompt_hub - Categorize any ticket where the user is asking about prompt versioning, collaboration features, or managing prompts inside LangSmith.
langsmith_sdk -  Categorize any ticket where the user is asking about LangSmith SDKs, tracing integrations, API usage, or developer tooling.
other_partnerships - Categorize any ticket where the user is inquiring about business partnerships, integrations, collaborations, or co-marketing opportunities.
other_sales - Categorize any ticket where the user is requesting an enterprise demo, information about self-hosting or on-premise solutions, or is expressing intent to purchase.
other_spam - Categorize any ticket that is irrelevant, unsolicited marketing, mass outreach, or otherwise spammy communication including link sharing
langgraph_oss_js - Categorize any ticket where the user is asking about the LangChain open-source Python library, including bugs, errors, or implementation questions. If the issue mentions a GitHub issue or links to the LangGraph Python GitHub repo, and it does not mention LangGraph, LangSmith, Evaluation, or traces, then it belongs here.
langchain_oss_js - Categorize any ticket where the user is asking about the LangChain open-source JavaScript library, including bugs, installation issues, or usage questions. If the issue mentions a GitHub issue or links to the LangChain JS GitHub repo, and it does not mention LangGraph, LangSmith, Evaluation, or traces, then it belongs here.
other_marketing_promotional - Categorize any ticket that is irrelevant, unsolicited marketing, mass outreach, or otherwise spammy communication including link sharing
langsmith_insights - Categorize any ticket that is about automated insights aka CLIO
</Category Types>
""".strip()

_OUTPUT_CATEGORY = """
Return a JSON object with: category (string) and confidence (number 0.0-1.0).
""".strip()

# Public prompts composed from shared sections
RESOLUTION_CLASSIFICATION_PROMPT = f"""
{_SCOPE_RESOLUTION}

{_IMPORTANT_RULES}

{_RESOLUTION_CATEGORIES}

{_OUTPUT_RESOLUTION}

{_CONVERSATION_HISTORY}
"""

CATEGORY_CLASSIFICATION_PROMPT = f"""
{_SCOPE_CATEGORY}
Base your classification on the primary nature of the customer's issue or request.
If multiple issues or topics are discussed, prioritize the most important one.

{_IMPORTANT_RULES}

{_CATEGORY_CATEGORIES}

{_OUTPUT_CATEGORY}

{_CONVERSATION_HISTORY}
"""

COMBINED_CLASSIFICATION_PROMPT = f"""
Scope: You are an expert customer support analyst. Based on the conversation history provided, classify both the resolution type and the category for this closed support issue.

{_IMPORTANT_RULES}

{_RESOLUTION_CATEGORIES}

{_CATEGORY_CATEGORIES}

Return a JSON object with: resolution (string), resolution_confidence (number 0.0-1.0), category (string), category_confidence (number 0.0-1.0).

{_CONVERSATION_HISTORY}
"""