from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


LLMMode = Literal["mock", "deepseek", "routed"]


class AgentModelRoute(BaseModel):
    """A secret-free route stored in project state and checkpoints."""

    model_config = ConfigDict(extra="forbid")

    credential_id: str = Field(min_length=3, max_length=80)
    model: str = Field(min_length=1, max_length=200)

    @field_validator("credential_id", "model")
    @classmethod
    def route_value_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ModelRoutingConfig(BaseModel):
    """Run-level immutable routing contract.

    Only credential identifiers are serialised. API keys remain inside the
    server-side credential vault and never enter LangGraph state.
    """

    model_config = ConfigDict(extra="forbid")

    preset: Literal["single", "balanced", "cross_review", "custom"] = "custom"
    default_route: AgentModelRoute
    agent_routes: dict[str, AgentModelRoute] = Field(default_factory=dict)

    @field_validator("agent_routes")
    @classmethod
    def bounded_agent_routes(
        cls, value: dict[str, AgentModelRoute]
    ) -> dict[str, AgentModelRoute]:
        if len(value) > 32:
            raise ValueError("agent_routes may contain at most 32 agents")
        return value


class ProviderCredentialCreate(BaseModel):
    """Bring-your-own-key input. The secret is never used in a response model."""

    model_config = ConfigDict(extra="forbid")

    preset_id: str = Field(default="custom", min_length=2, max_length=60)
    display_name: str = Field(min_length=2, max_length=80)
    base_url: str = Field(min_length=8, max_length=500)
    api_key: SecretStr
    manual_model: str | None = Field(default=None, max_length=200)
    supports_json_object: bool = True

    @field_validator("preset_id", "display_name", "base_url")
    @classmethod
    def provider_value_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("manual_model")
    @classmethod
    def optional_model_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ProviderModelView(BaseModel):
    id: str
    owned_by: str = ""


class ProviderCredentialView(BaseModel):
    credential_id: str
    preset_id: str
    provider_name: str
    base_url: str
    secret_hint: str
    source: Literal["builtin", "environment", "session"]
    status: Literal["ready", "manual", "unavailable"]
    supports_json_object: bool
    models: list[ProviderModelView] = Field(default_factory=list)
    note: str = ""


class ModelRoutingValidation(BaseModel):
    valid: bool
    routing: ModelRoutingConfig
    warnings: list[str] = Field(default_factory=list)


class IssueRequest(BaseModel):
    repo_path: str
    issue: str = Field(min_length=5, max_length=10_000)
    constraints: list[str] = Field(default_factory=list)
    test_command: str = "python -m unittest discover -s tests"
    llm_mode: LLMMode | None = None
    model_routing: ModelRoutingConfig | None = None

    @field_validator("repo_path", "test_command")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ProjectRequest(BaseModel):
    """Client brief for the software-company project factory workflow."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=2, max_length=120)
    client_brief: str = Field(min_length=20, max_length=20_000)
    target_users: str = Field(min_length=2, max_length=5_000)
    budget_cny: int | None = Field(default=None, ge=0, le=1_000_000_000)
    delivery_deadline: str = Field(default="待产品经理评估", max_length=200)
    policy_constraints: list[str] = Field(default_factory=list, max_length=20)
    business_constraints: list[str] = Field(default_factory=list, max_length=20)
    preferred_stack: str = Field(default="由架构团队评估", max_length=200)
    deployment_target: Literal[
        "local",
        "intranet",
        "docker",
        "kubernetes",
        "public_cloud",
        "edge",
        "undecided",
    ] = "undecided"
    deployment_conditions: str = Field(
        default="由 DevOps 团队根据方案给出运行、上线与回滚条件",
        max_length=5_000,
    )
    llm_mode: LLMMode | None = None
    model_routing: ModelRoutingConfig | None = None

    @field_validator(
        "project_name",
        "client_brief",
        "target_users",
        "delivery_deadline",
        "preferred_stack",
        "deployment_conditions",
    )
    @classmethod
    def project_text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ProjectLaunchRequest(ProjectRequest):
    """Server-created launch contract for a frozen collaboration room.

    These fields are intentionally absent from the public ``ProjectRequest`` so
    callers cannot choose storage paths or impersonate uploaded project assets.
    """

    collaboration_id: str | None = Field(default=None, max_length=64)
    conversation_summary: str = Field(default="", max_length=30_000)
    uploaded_documents: list[dict] = Field(default_factory=list, max_length=50)


class RequirementItem(BaseModel):
    requirement_id: str
    title: str
    description: str
    priority: Literal["must", "should", "could", "wont"] = "must"
    acceptance_criteria: list[str] = Field(min_length=1, max_length=12)


class ProductRequirementsDocument(BaseModel):
    product_name: str
    vision: str
    problem_statement: str
    target_users: list[str] = Field(min_length=1, max_length=12)
    user_journeys: list[str] = Field(min_length=1, max_length=20)
    functional_requirements: list[RequirementItem] = Field(
        min_length=1, max_length=30
    )
    non_functional_requirements: list[str] = Field(
        default_factory=list, max_length=30
    )
    in_scope: list[str] = Field(default_factory=list, max_length=30)
    out_of_scope: list[str] = Field(default_factory=list, max_length=30)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    clarification_questions: list[str] = Field(default_factory=list, max_length=12)
    success_metrics: list[str] = Field(default_factory=list, max_length=20)


class ProjectPhase(BaseModel):
    phase_id: str
    name: str
    objective: str
    deliverables: list[str] = Field(min_length=1, max_length=15)
    estimated_days: int = Field(ge=1, le=3650)
    budget_cny: int | None = Field(default=None, ge=0)
    dependencies: list[str] = Field(default_factory=list, max_length=12)
    exit_criteria: list[str] = Field(min_length=1, max_length=12)


class ProjectExecutionPlan(BaseModel):
    methodology: str
    executive_summary: str
    phases: list[ProjectPhase] = Field(min_length=2, max_length=20)
    team_roles: list[str] = Field(min_length=1, max_length=30)
    budget_summary: str
    schedule_summary: str
    risks: list[str] = Field(default_factory=list, max_length=30)
    compliance_actions: list[str] = Field(default_factory=list, max_length=30)
    communication_plan: list[str] = Field(default_factory=list, max_length=20)


class ArchitectureComponent(BaseModel):
    name: str
    responsibility: str
    technology: str
    interfaces: list[str] = Field(default_factory=list, max_length=15)


class ArchitectureBlueprint(BaseModel):
    architecture_style: str
    primary_language: str
    recommended_stack: list[str] = Field(min_length=1, max_length=20)
    rationale: list[str] = Field(min_length=1, max_length=20)
    components: list[ArchitectureComponent] = Field(min_length=1, max_length=24)
    data_design: list[str] = Field(default_factory=list, max_length=24)
    api_design: list[str] = Field(default_factory=list, max_length=30)
    security_design: list[str] = Field(default_factory=list, max_length=30)
    observability: list[str] = Field(default_factory=list, max_length=20)
    deployment_topology: list[str] = Field(default_factory=list, max_length=20)


class GeneratedFileDraft(BaseModel):
    path: str = Field(min_length=1, max_length=240)
    purpose: str = Field(min_length=1, max_length=1_000)
    content: str = Field(max_length=200_000)


class GeneratedProject(BaseModel):
    summary: str
    files: list[GeneratedFileDraft] = Field(min_length=1, max_length=36)


class ProjectRevision(BaseModel):
    summary: str
    changed_files: list[GeneratedFileDraft] = Field(min_length=1, max_length=24)
    delete_paths: list[str] = Field(default_factory=list, max_length=12)
    addressed_defects: list[str] = Field(min_length=1, max_length=40)


class DeploymentGuide(BaseModel):
    target: str
    prerequisites: list[str] = Field(min_length=1, max_length=30)
    environment_variables: list[str] = Field(default_factory=list, max_length=30)
    local_run_steps: list[str] = Field(min_length=1, max_length=30)
    production_steps: list[str] = Field(min_length=1, max_length=40)
    health_checks: list[str] = Field(min_length=1, max_length=20)
    release_conditions: list[str] = Field(min_length=1, max_length=30)
    rollback_steps: list[str] = Field(min_length=1, max_length=20)
    usage_guide: list[str] = Field(min_length=1, max_length=30)
    environment_specification: list[str] = Field(
        default_factory=list, max_length=30
    )
    build_steps: list[str] = Field(default_factory=list, max_length=30)
    test_steps: list[str] = Field(default_factory=list, max_length=30)
    boundary_conditions: list[str] = Field(default_factory=list, max_length=30)
    troubleshooting: list[str] = Field(default_factory=list, max_length=40)
    backup_restore: list[str] = Field(default_factory=list, max_length=20)
    observability: list[str] = Field(default_factory=list, max_length=20)
    security_checklist: list[str] = Field(default_factory=list, max_length=30)
    handover_checklist: list[str] = Field(default_factory=list, max_length=30)


class ProjectSessionCreate(BaseModel):
    """Create a persistent client/product-manager collaboration room."""

    project_name: str = Field(min_length=2, max_length=120)
    target_users: str = Field(default="待通过对话确认", max_length=5_000)
    budget_cny: int | None = Field(default=None, ge=0, le=1_000_000_000)
    delivery_deadline: str = Field(default="待通过对话确认", max_length=200)
    policy_constraints: list[str] = Field(default_factory=list, max_length=20)
    business_constraints: list[str] = Field(default_factory=list, max_length=20)
    preferred_stack: str = Field(default="由架构团队评估", max_length=200)
    deployment_target: Literal[
        "local",
        "intranet",
        "docker",
        "kubernetes",
        "public_cloud",
        "edge",
        "undecided",
    ] = "undecided"
    deployment_conditions: str = Field(
        default="由 DevOps 团队与客户共同确认",
        max_length=5_000,
    )
    llm_mode: LLMMode | None = None
    model_routing: ModelRoutingConfig | None = None


class ProjectChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12_000)

    @field_validator("content")
    @classmethod
    def chat_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ProjectMessage(BaseModel):
    message_id: str
    role: Literal["client", "product_manager", "system"]
    author: str
    content: str
    created_at: str
    evidence_ids: list[str] = Field(default_factory=list)


class ProjectAsset(BaseModel):
    asset_id: str
    original_name: str
    stored_name: str
    media_type: str
    size: int = Field(ge=0)
    parser: str
    chunk_count: int = Field(ge=0)
    redaction_count: int = Field(ge=0)
    uploaded_at: str


class ProductUnderstanding(BaseModel):
    summary: str
    confirmed_goals: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=30)
    open_questions: list[str] = Field(default_factory=list, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    ready_for_workshop: bool = False


class ProductChatResponse(BaseModel):
    reply: str = Field(min_length=1)
    understanding: ProductUnderstanding


class ProjectSessionView(BaseModel):
    session_id: str
    status: str
    project_name: str
    llm_mode: str
    model_routing: ModelRoutingConfig | None = None
    messages: list[ProjectMessage] = Field(default_factory=list)
    assets: list[ProjectAsset] = Field(default_factory=list)
    understanding: ProductUnderstanding
    run_id: str | None = None


class WorkshopRisk(BaseModel):
    risk_id: str
    category: str
    scenario: str
    impact: str
    avoidance_strategy: str
    early_warning: str
    owner: str


class RequirementDecomposition(BaseModel):
    item_id: str
    requirement_id: str
    workstream: str
    deliverable: str
    dependencies: list[str] = Field(default_factory=list, max_length=12)
    done_definition: list[str] = Field(default_factory=list, max_length=12)


class RequirementsWorkshop(BaseModel):
    meeting_title: str
    participants: list[str] = Field(min_length=4, max_length=20)
    client_intent_summary: str
    decisions: list[str] = Field(min_length=1, max_length=30)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
    risks: list[WorkshopRisk] = Field(default_factory=list, max_length=30)
    decomposition: list[RequirementDecomposition] = Field(
        min_length=1, max_length=50
    )
    testing_strategy: list[str] = Field(default_factory=list, max_length=30)
    delivery_boundaries: list[str] = Field(default_factory=list, max_length=30)
    readiness_recommendation: str


class TestCaseResult(BaseModel):
    case_id: str
    requirement_id: str
    name: str
    method: str
    status: Literal["passed", "failed", "blocked", "runtime_required"]
    evidence: str


class QualityAssuranceReport(BaseModel):
    passed: bool
    summary: str
    test_cases: list[TestCaseResult] = Field(min_length=1, max_length=80)
    defects: list[str] = Field(default_factory=list, max_length=40)
    acceptance_coverage: list[str] = Field(default_factory=list, max_length=40)
    recommended_commands: list[str] = Field(default_factory=list, max_length=30)
    runtime_validation_required: bool = True
    validation_level: Literal["static_preflight", "sandbox_runtime"] = (
        "static_preflight"
    )
    runtime_verified: bool = False
    environment_limitations: list[str] = Field(default_factory=list, max_length=20)


class Evidence(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    path: str
    text: str
    source_type: Literal["code", "document"]
    trust_level: str = "repository_source"
    document_version: str | None = None
    language: str
    start_line: int | None = None
    end_line: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    symbol: str | None = None
    fusion_score: float
    rerank_score: float
    score: float


class PlanStep(BaseModel):
    step_id: str
    description: str
    target_files: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"] = "medium"
    verification: str


class ChangePlan(BaseModel):
    summary: str
    assumptions: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(min_length=1, max_length=12)


class CodeEditDraft(BaseModel):
    path: str
    reasoning: str
    search: str = Field(min_length=1)
    replace: str


class CodeEditSet(BaseModel):
    edits: list[CodeEditDraft] = Field(min_length=1, max_length=24)


class ChangeProposal(BaseModel):
    path: str
    reasoning: str
    original_sha256: str
    new_content: str
    diff: str


class ReviewFinding(BaseModel):
    severity: Literal["info", "low", "medium", "high", "critical"]
    message: str
    path: str | None = None


class ReviewDecision(BaseModel):
    approved: bool
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)


class Reflection(BaseModel):
    root_cause: str
    next_action: str
    retry_recommended: bool = False


class ApprovalDecision(BaseModel):
    approved: bool
    comment: str = Field(default="", max_length=2000)
    actor_id: str = Field(default="local-user", min_length=1, max_length=200)
    artifact_digest: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )


class RunCreated(BaseModel):
    run_id: str
    status: str
    state: dict
    interrupt: dict | None = None
