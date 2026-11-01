from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from .state import RepoPilotState
from .studio_agents import SoftwareStudioAgentSuite


# Every source of release feedback (QA defects, release-review findings and a
# client's change request) shares one revision budget, so a project cannot
# loop between development and review indefinitely.
MAX_DEVELOPMENT_REVISIONS = 2


def _has_error(state: RepoPilotState) -> bool:
    return bool(state.get("error"))


def _revision_budget_left(state: RepoPilotState) -> bool:
    return int(state.get("development_revision", 0)) < MAX_DEVELOPMENT_REVISIONS


def _continue_or_failure(state: RepoPilotState) -> str:
    return "failure" if _has_error(state) else "continue"


def _after_requirements_approval(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    if state.get("requirements_approved"):
        return "plan"
    if state.get("status") == "cancelled":
        return "report"
    if int(state.get("requirements_revision", 0)) >= 3:
        return "report"
    return "revise"


def _after_release_review(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    if state["release_review"]["approved"]:
        return "approval"
    return "revise" if _revision_budget_left(state) else "report"


def _after_qa(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    if state["qa_report"]["passed"]:
        return "review"
    return "revise" if _revision_budget_left(state) else "report"


def _after_release_approval(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    if state.get("release_approved"):
        return "write"
    # A rejection that explains what to change is a change request; one
    # without a reason (or with no budget left) closes the project.
    if state.get("status") == "release_changes_requested" and (
        _revision_budget_left(state)
    ):
        return "revise"
    return "report"


def _after_report(state: RepoPilotState) -> str:
    return "failure" if _has_error(state) else "end"


def _after_quality(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    return "package" if state["quality_gate"]["success"] else "report"


def _guarded_nodes(
    suite: SoftwareStudioAgentSuite,
) -> dict[str, Callable[[RepoPilotState], dict]]:
    return {
        "client_intake": suite.guarded(
            "ClientLiaisonAgent", suite.client_intake_agent
        ),
        "knowledge": suite.guarded("KnowledgeAgent", suite.knowledge_agent),
        "product_manager": suite.guarded(
            "ProductManagerAgent", suite.product_manager_agent
        ),
        "requirements_workshop": suite.guarded(
            "DeliveryDirectorAgent",
            suite.requirements_workshop_agent,
        ),
        "requirements_approval": suite.guarded(
            "RequirementsApprovalAgent",
            suite.requirements_approval_agent,
        ),
        "requirements_revision": suite.guarded(
            "ProductManagerAgent",
            suite.requirements_revision_agent,
        ),
        "project_manager": suite.guarded(
            "ProjectManagerAgent", suite.project_manager_agent
        ),
        "architect": suite.guarded("ArchitectAgent", suite.architect_agent),
        "developer": suite.guarded(
            "DevelopmentTeamAgent", suite.developer_agent
        ),
        "developer_revision": suite.guarded(
            "DevelopmentTeamAgent",
            suite.developer_revision_agent,
        ),
        "deployment": suite.guarded("DevOpsAgent", suite.deployment_agent),
        "qa": suite.guarded("QualityAssuranceAgent", suite.qa_agent),
        "release_review": suite.guarded(
            "ReleaseReviewerAgent", suite.reviewer_agent
        ),
        "release_approval": suite.guarded(
            "ReleaseApprovalAgent", suite.release_approval_agent
        ),
        "write_project": suite.guarded(
            "DeliveryEngineerAgent", suite.executor_agent
        ),
        "quality": suite.guarded(
            "QualityEngineeringAgent", suite.quality_agent
        ),
        "package": suite.guarded(
            "DeliveryPackagingAgent", suite.packager_agent
        ),
        "report": suite.guarded(
            "DeliveryManagerAgent", suite.reporter_agent
        ),
    }


def build_studio_workflow(suite: SoftwareStudioAgentSuite, checkpointer):
    graph = StateGraph(RepoPilotState)
    for name, node in _guarded_nodes(suite).items():
        graph.add_node(name, node)
    graph.add_node("failure", suite.failure_reporter_agent)

    graph.add_edge(START, "client_intake")
    for current, following in (
        ("client_intake", "knowledge"),
        ("knowledge", "product_manager"),
        ("product_manager", "requirements_workshop"),
        ("requirements_workshop", "requirements_approval"),
        ("requirements_revision", "requirements_workshop"),
        ("project_manager", "architect"),
        ("architect", "developer"),
        ("developer", "deployment"),
        ("developer_revision", "deployment"),
        ("deployment", "qa"),
        ("write_project", "quality"),
        ("package", "report"),
    ):
        graph.add_conditional_edges(
            current,
            _continue_or_failure,
            {"continue": following, "failure": "failure"},
        )

    graph.add_conditional_edges(
        "requirements_approval",
        _after_requirements_approval,
        {
            "plan": "project_manager",
            "revise": "requirements_revision",
            "report": "report",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "qa",
        _after_qa,
        {
            "review": "release_review",
            "revise": "developer_revision",
            "report": "report",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "release_review",
        _after_release_review,
        {
            "approval": "release_approval",
            "revise": "developer_revision",
            "report": "report",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "release_approval",
        _after_release_approval,
        {
            "write": "write_project",
            "revise": "developer_revision",
            "report": "report",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "quality",
        _after_quality,
        {
            "package": "package",
            "report": "report",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "report",
        _after_report,
        {"end": END, "failure": "failure"},
    )
    graph.add_edge("failure", END)
    return graph.compile(checkpointer=checkpointer)
