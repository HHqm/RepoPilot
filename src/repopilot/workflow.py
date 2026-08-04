from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from .agents import AgentSuite
from .state import RepoPilotState


def _has_error(state: RepoPilotState) -> bool:
    return bool(state.get("error"))


def _continue_or_failure(state: RepoPilotState) -> str:
    return "failure" if _has_error(state) else "continue"


def _after_review(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    return "approval" if state["review"]["approved"] else "report"


def _after_approval(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    return "execute" if state.get("approved") else "report"


def _after_verification(state: RepoPilotState) -> str:
    if _has_error(state):
        return "failure"
    return "report" if state["verification"]["success"] else "reflect"


def _after_report(state: RepoPilotState) -> str:
    return "failure" if _has_error(state) else "end"


def _guarded_nodes(
    suite: AgentSuite,
) -> dict[str, Callable[[RepoPilotState], dict]]:
    return {
        "intake": suite.guarded("IntakeAgent", suite.intake_agent),
        "repo_mapper": suite.guarded(
            "RepoMapperAgent", suite.repo_mapper_agent
        ),
        "retrieve": suite.guarded("RetrievalAgent", suite.retrieval_agent),
        "plan": suite.guarded("PlannerAgent", suite.planner_agent),
        "code": suite.guarded("CoderAgent", suite.coder_agent),
        "review": suite.guarded("ReviewerAgent", suite.reviewer_agent),
        "approval": suite.guarded("ApprovalAgent", suite.approval_agent),
        "execute": suite.guarded("ExecutorAgent", suite.executor_agent),
        "test": suite.guarded("TestAgent", suite.test_agent),
        "verify": suite.guarded("VerifierAgent", suite.verifier_agent),
        "reflect": suite.guarded("ReflectionAgent", suite.reflection_agent),
        "report": suite.guarded("ReporterAgent", suite.reporter_agent),
    }


def build_workflow(suite: AgentSuite, checkpointer):
    graph = StateGraph(RepoPilotState)
    for name, node in _guarded_nodes(suite).items():
        graph.add_node(name, node)
    graph.add_node("failure", suite.failure_reporter_agent)

    graph.add_edge(START, "intake")
    for current, following in (
        ("intake", "repo_mapper"),
        ("repo_mapper", "retrieve"),
        ("retrieve", "plan"),
        ("plan", "code"),
        ("code", "review"),
        ("execute", "test"),
        ("test", "verify"),
        ("reflect", "report"),
    ):
        graph.add_conditional_edges(
            current,
            _continue_or_failure,
            {"continue": following, "failure": "failure"},
        )

    graph.add_conditional_edges(
        "review",
        _after_review,
        {"approval": "approval", "report": "report", "failure": "failure"},
    )
    graph.add_conditional_edges(
        "approval",
        _after_approval,
        {"execute": "execute", "report": "report", "failure": "failure"},
    )
    graph.add_conditional_edges(
        "verify",
        _after_verification,
        {"report": "report", "reflect": "reflect", "failure": "failure"},
    )
    graph.add_conditional_edges(
        "report",
        _after_report,
        {"end": END, "failure": "failure"},
    )
    graph.add_edge("failure", END)
    return graph.compile(checkpointer=checkpointer)
