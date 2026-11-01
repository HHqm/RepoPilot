from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


# v3 adds the maintenance self-correction loop (review revisions, repair
# attempts, multi-query retrieval) and run metrics. All new keys are optional,
# so v2 checkpoints still resume.
STATE_SCHEMA_VERSION = 3


class RepoPilotState(TypedDict, total=False):
    state_schema_version: int
    run_id: str
    workflow_kind: str
    repo_path: str
    issue: str
    constraints: list[str]
    test_command: str
    llm_mode: str
    model_routing: dict[str, Any]
    status: str
    repo_map: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    plan: dict[str, Any]
    proposals: list[dict[str, Any]]
    review: dict[str, Any]
    approved: bool
    approval_comment: str
    apply_result: dict[str, Any]
    test_result: dict[str, Any]
    verification: dict[str, Any]
    reflection: dict[str, Any]
    retrieval: dict[str, Any]
    review_revision: int
    review_feedback: dict[str, Any]
    repair_attempt: int
    attempt_history: list[dict[str, Any]]
    run_metrics: dict[str, Any]
    report: str
    project_name: str
    collaboration_id: str
    conversation_summary: str
    client_messages: list[dict[str, Any]]
    uploaded_documents: list[dict[str, Any]]
    project_knowledge_path: str
    client_brief: str
    target_users: str
    budget_cny: int | None
    delivery_deadline: str
    policy_constraints: list[str]
    business_constraints: list[str]
    preferred_stack: str
    deployment_target: str
    deployment_conditions: str
    requirements_revision: int
    requirements_feedback: str
    product_requirements: dict[str, Any]
    requirements_workshop: dict[str, Any]
    requirements_approved: bool
    requirements_approval_comment: str
    project_plan: dict[str, Any]
    architecture: dict[str, Any]
    generated_project: dict[str, Any]
    development_revision: int
    deployment_guide: dict[str, Any]
    qa_report: dict[str, Any]
    release_review: dict[str, Any]
    release_approved: bool
    release_approval_comment: str
    quality_gate: dict[str, Any]
    project_output_path: str
    delivery_archive: dict[str, Any]
    delivery_archive_path: str
    error: dict[str, Any]
    traces: Annotated[list[dict[str, Any]], operator.add]
