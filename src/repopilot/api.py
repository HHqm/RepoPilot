from __future__ import annotations

import atexit
import hmac
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .models import (
    ApprovalDecision,
    ModelRoutingConfig,
    ModelRoutingValidation,
    ProviderCredentialCreate,
    ProviderCredentialView,
    IssueRequest,
    ProjectChatRequest,
    ProjectRequest,
    ProjectSessionCreate,
    ProjectSessionView,
    RunCreated,
)
from .model_routing import LLM_AGENT_PROFILES
from .service import RepoPilotService
from .town import (
    AGENT_CATALOG,
    PHASE_LABELS,
    TERMINAL_UI_STATUSES,
    TOWN_LOCATIONS,
)


service = RepoPilotService(Settings.from_env())
atexit.register(service.close)
FRONTEND_VERSION = "0.6.0"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    service.close()


app = FastAPI(
    title="RepoPilot API",
    version="0.6.0",
    description=(
        "Human-governed multi-agent software company: client brief to "
        "reviewed, deployable project."
    ),
    lifespan=lifespan,
)
WEB_ROOT = Path(__file__).with_name("web")
app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")


@app.middleware("http")
async def prevent_stale_frontend(request: Request, call_next):
    path = request.url.path
    public_path = (
        path == "/"
        or path == "/health"
        or path == "/ui-config"
        or path == "/model-providers/catalog"
        or path == "/openapi.json"
        or path.startswith("/assets/")
        or path.startswith("/docs")
        or path.startswith("/redoc")
    )
    if not public_path:
        configured_token = service.settings.api_token
        if configured_token:
            authorization = request.headers.get("authorization", "")
            supplied = (
                authorization[7:]
                if authorization.lower().startswith("bearer ")
                else ""
            )
            if not hmac.compare_digest(supplied, configured_token):
                return JSONResponse(
                    {"detail": "Missing or invalid RepoPilot API token"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        else:
            client_host = request.client.host if request.client else ""
            if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
                return JSONResponse(
                    {
                        "detail": (
                            "Remote API access requires REPOPILOT_API_TOKEN"
                        )
                    },
                    status_code=403,
                )
    response = await call_next(request)
    if (
        path == "/"
        or path.startswith("/assets/")
        or path == "/ui-config"
        or path == "/model-providers/catalog"
    ):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-RepoPilot-Frontend"] = FRONTEND_VERSION
    return response


@app.get("/", include_in_schema=False)
def town_console() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str | int]:
    rag = service.suite.rag
    return {
        "status": "ok",
        "llm_mode": service.settings.llm_mode,
        "model": service.settings.deepseek_model,
        "rag": "qdrant_dense_sparse_rrf_rerank",
        "embedding_provider": service.settings.rag_embedding_provider,
        "embedding_model": rag.store.embedding.model_id,
        "sparse_model": rag.store.embedding.sparse_model_id,
        "reranker_provider": service.settings.rag_reranker_provider,
        "reranker_model": rag.store.reranker.model_id,
        "vector_store": "qdrant-local" if rag.store.local_mode else "qdrant",
        "collection": rag.store.collection_name,
        "document_parser": service.settings.rag_document_parser,
        "context_compression": "deterministic_schema_aware",
        "context_max_input_tokens": service.settings.context_max_input_tokens,
        "execution": "background",
        "events": "live",
        "workflow": "client_chat_files_workshop_to_zip",
        "model_routing": "agent_level_multi_provider",
    }


@app.get("/ui-config")
def ui_config() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    sample = project_root / "examples" / "buggy_calculator"
    return {
        "sample_repo_path": str(sample) if sample.is_dir() else "",
        "default_llm_mode": service.settings.llm_mode,
        "model": service.settings.deepseek_model,
        "frontend_version": FRONTEND_VERSION,
        "context": {
            "compression": "deterministic_schema_aware",
            "max_input_tokens": service.settings.context_max_input_tokens,
            "planner_tokens": service.settings.context_planner_tokens,
            "coder_tokens": service.settings.context_coder_tokens,
            "reviewer_tokens": service.settings.context_reviewer_tokens,
            "reflection_tokens": service.settings.context_reflection_tokens,
        },
        "agents": [
            agent for agent in AGENT_CATALOG if not agent.get("hidden", False)
        ],
        "system_agents": [
            agent for agent in AGENT_CATALOG if agent.get("hidden", False)
        ],
        "locations": TOWN_LOCATIONS,
        "phases": PHASE_LABELS,
        "terminal_statuses": TERMINAL_UI_STATUSES,
        "workflow": "collaborative_project_factory",
        "model_providers": service.model_registry.catalog(),
        "model_agents": list(LLM_AGENT_PROFILES),
        "private_model_endpoints": (
            service.settings.allow_private_model_endpoints
        ),
        "supported_languages": [
            "auto",
            "Python",
            "Java",
            "Go",
            "C++",
            "TypeScript/JavaScript",
        ],
    }


@app.get("/model-providers/catalog")
def model_provider_catalog() -> list[dict]:
    return service.model_registry.catalog()


@app.get(
    "/model-providers/credentials",
    response_model=list[ProviderCredentialView],
)
def list_model_credentials() -> list[ProviderCredentialView]:
    return service.model_registry.list_credentials()


@app.post(
    "/model-providers/credentials",
    response_model=ProviderCredentialView,
    status_code=201,
)
def create_model_credential(
    request: ProviderCredentialCreate,
) -> ProviderCredentialView:
    try:
        return service.model_registry.add_credential(request)
    except Exception as exc:
        detail = service.suite.redact_secrets(str(exc))
        raise HTTPException(status_code=400, detail=detail) from exc


@app.post(
    "/model-providers/credentials/{credential_id}/refresh",
    response_model=ProviderCredentialView,
)
def refresh_model_credential(
    credential_id: str,
) -> ProviderCredentialView:
    try:
        return service.model_registry.refresh_models(credential_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        detail = service.suite.redact_secrets(str(exc))
        raise HTTPException(status_code=400, detail=detail) from exc


@app.delete(
    "/model-providers/credentials/{credential_id}",
    status_code=204,
)
def delete_model_credential(credential_id: str) -> Response:
    try:
        service.model_registry.delete_credential(credential_id)
        service.suite.invalidate_credential(credential_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@app.post(
    "/model-routing/validate",
    response_model=ModelRoutingValidation,
)
def validate_model_routing(
    routing: ModelRoutingConfig,
) -> ModelRoutingValidation:
    try:
        return service.model_registry.validate_routing(routing)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/project-sessions",
    response_model=ProjectSessionView,
    status_code=201,
)
def create_project_session(
    request: ProjectSessionCreate,
) -> ProjectSessionView:
    try:
        return service.collaboration.create(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/project-sessions/{session_id}",
    response_model=ProjectSessionView,
)
def get_project_session(session_id: str) -> ProjectSessionView:
    try:
        return service.collaboration.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/project-sessions/{session_id}/messages",
    response_model=ProjectSessionView,
)
def chat_with_product_manager(
    session_id: str,
    request: ProjectChatRequest,
) -> ProjectSessionView:
    try:
        return service.collaboration.chat(session_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/project-sessions/{session_id}/files",
    response_model=ProjectSessionView,
)
async def upload_project_file(
    session_id: str,
    file: UploadFile = File(...),
) -> ProjectSessionView:
    try:
        limit = service.settings.rag_max_document_bytes
        content = bytearray()
        while True:
            chunk = await file.read(min(1024 * 1024, limit + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > limit:
                raise ValueError(f"文件超过 {limit} 字节限制")
        return service.collaboration.add_document(
            session_id,
            filename=file.filename or "document",
            media_type=file.content_type or "application/octet-stream",
            content=bytes(content),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post(
    "/project-sessions/{session_id}/launch",
    response_model=RunCreated,
    status_code=202,
)
def launch_project_session(session_id: str) -> RunCreated:
    try:
        run, _ = service.launch_collaboration(session_id)
        return run
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/demo-workspace")
def create_demo_workspace() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    source = project_root / "examples" / "buggy_calculator"
    if not source.is_dir():
        raise HTTPException(status_code=404, detail="Bundled demo repository not found")
    target = service.settings.runtime_dir / "demos" / uuid.uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return {"repo_path": str(target)}


@app.post("/runs", response_model=RunCreated, status_code=202)
def create_run(request: IssueRequest) -> RunCreated:
    try:
        return service.start(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/projects", response_model=RunCreated, status_code=202)
def create_project(request: ProjectRequest) -> RunCreated:
    try:
        return service.start_project(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/runs/{run_id}/decision", response_model=RunCreated, status_code=202)
def decide_run(run_id: str, decision: ApprovalDecision) -> RunCreated:
    try:
        return service.decide(run_id, decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/runs/{run_id}", response_model=RunCreated)
def get_run(run_id: str) -> RunCreated:
    try:
        return service.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/download")
def download_delivery(run_id: str) -> FileResponse:
    try:
        run = service.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if run.status != "completed":
        raise HTTPException(status_code=409, detail="项目尚未完成，交付包不可下载")
    archive_value = str(run.state.get("delivery_archive_path") or "")
    if not archive_value:
        raise HTTPException(status_code=404, detail="交付包尚未生成")
    archive = Path(archive_value).resolve()
    projects_root = (service.settings.runtime_dir / "projects").resolve()
    try:
        archive.relative_to(projects_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="交付包路径不安全") from exc
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise HTTPException(status_code=404, detail="交付包不存在")
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=archive.name,
    )


@app.get("/runs/{run_id}/events")
def get_events(
    run_id: str,
    request: Request,
    after: int = 0,
) -> StreamingResponse:
    try:
        service.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def stream():
        cursor = max(0, after)
        header_cursor = request.headers.get("last-event-id", "").strip()
        if header_cursor.isdigit():
            cursor = max(cursor, int(header_cursor))
        yield "retry: 1000\n\n"
        for event in service.iter_events(run_id, after_sequence=cursor):
            if event is None:
                yield ": heartbeat\n\n"
                continue
            payload = json.dumps(
                {
                    "sequence": event["sequence"],
                    "occurred_at": event["occurred_at"],
                    **event["data"],
                },
                ensure_ascii=False,
            )
            yield (
                f"id: {event['sequence']}\n"
                f"event: {event['type']}\n"
                f"data: {payload}\n\n"
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
