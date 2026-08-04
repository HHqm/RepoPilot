const elements = {
  systemText: document.querySelector("#systemText"),
  engineBadge: document.querySelector("#engineBadge"),
  agentCount: document.querySelector("#agentCount"),
  documentCount: document.querySelector("#documentCount"),
  runStatus: document.querySelector("#runStatus"),
  phaseMetric: document.querySelector("#phaseMetric"),
  themeToggle: document.querySelector("#themeToggle"),
  setupCard: document.querySelector("#setupCard"),
  sessionForm: document.querySelector("#sessionForm"),
  projectName: document.querySelector("#projectName"),
  targetUsers: document.querySelector("#targetUsers"),
  budgetCny: document.querySelector("#budgetCny"),
  deliveryDeadline: document.querySelector("#deliveryDeadline"),
  preferredStack: document.querySelector("#preferredStack"),
  deploymentTarget: document.querySelector("#deploymentTarget"),
  constraints: document.querySelector("#constraints"),
  llmMode: document.querySelector("#llmMode"),
  routeSummary: document.querySelector("#routeSummary"),
  openModelCenter: document.querySelector("#openModelCenter"),
  configureModelsButton: document.querySelector("#configureModelsButton"),
  modelCenter: document.querySelector("#modelCenter"),
  closeModelCenter: document.querySelector("#closeModelCenter"),
  providerForm: document.querySelector("#providerForm"),
  providerPreset: document.querySelector("#providerPreset"),
  providerName: document.querySelector("#providerName"),
  providerBaseUrl: document.querySelector("#providerBaseUrl"),
  providerApiKey: document.querySelector("#providerApiKey"),
  providerManualModel: document.querySelector("#providerManualModel"),
  providerJsonMode: document.querySelector("#providerJsonMode"),
  connectProviderButton: document.querySelector("#connectProviderButton"),
  credentialCount: document.querySelector("#credentialCount"),
  credentialList: document.querySelector("#credentialList"),
  routingPreset: document.querySelector("#routingPreset"),
  defaultCredential: document.querySelector("#defaultCredential"),
  defaultModel: document.querySelector("#defaultModel"),
  applyRoutingPreset: document.querySelector("#applyRoutingPreset"),
  agentRoutingList: document.querySelector("#agentRoutingList"),
  routingWarnings: document.querySelector("#routingWarnings"),
  saveRoutingButton: document.querySelector("#saveRoutingButton"),
  fillDemo: document.querySelector("#fillDemo"),
  createSessionButton: document.querySelector("#createSessionButton"),
  currentProjectName: document.querySelector("#currentProjectName"),
  sessionId: document.querySelector("#sessionId"),
  sessionBadge: document.querySelector("#sessionBadge"),
  fileInput: document.querySelector("#fileInput"),
  assetList: document.querySelector("#assetList"),
  chatMessages: document.querySelector("#chatMessages"),
  pmStatus: document.querySelector("#pmStatus"),
  chatForm: document.querySelector("#chatForm"),
  chatInput: document.querySelector("#chatInput"),
  sendMessageButton: document.querySelector("#sendMessageButton"),
  newProjectButton: document.querySelector("#newProjectButton"),
  promptChips: document.querySelector("#promptChips"),
  readinessBadge: document.querySelector("#readinessBadge"),
  understandingSummary: document.querySelector("#understandingSummary"),
  confirmedGoals: document.querySelector("#confirmedGoals"),
  understoodConstraints: document.querySelector("#understoodConstraints"),
  openQuestions: document.querySelector("#openQuestions"),
  launchWorkshopButton: document.querySelector("#launchWorkshopButton"),
  meetingSection: document.querySelector("#meetingSection"),
  meetingStatus: document.querySelector("#meetingStatus"),
  meetingScreenTitle: document.querySelector("#meetingScreenTitle"),
  meetingScreenText: document.querySelector("#meetingScreenText"),
  meetingOutput: document.querySelector("#meetingOutput"),
  pipeline: document.querySelector("#pipeline"),
  approvalCard: document.querySelector("#approvalCard"),
  approvalState: document.querySelector("#approvalState"),
  approvalSummary: document.querySelector("#approvalSummary"),
  approvalDetails: document.querySelector("#approvalDetails"),
  approvalMeta: document.querySelector("#approvalMeta"),
  approvalMaterial: document.querySelector("#approvalMaterial"),
  approvalComment: document.querySelector("#approvalComment"),
  rejectButton: document.querySelector("#rejectButton"),
  approveButton: document.querySelector("#approveButton"),
  artifactPanel: document.querySelector("#artifactPanel"),
  deliverySummary: document.querySelector("#deliverySummary"),
  downloadButton: document.querySelector("#downloadButton"),
  downloadMeta: document.querySelector("#downloadMeta"),
  agentRoster: document.querySelector("#agentRoster"),
  activityList: document.querySelector("#activityList"),
  toast: document.querySelector("#toast"),
};

let uiConfig = null;
let projectSession = null;
let currentRun = null;
let currentCheckpoint = null;
let eventSource = null;
let activeMeetingTab = "decisions";
let activeArtifact = "prd";
let toastTimer = null;
let latestTraceAgent = null;
let providerCredentials = [];
let modelRoutingDraft = null;

const phaseLabels = {};
const terminalStatuses = new Set([
  "completed",
  "release_rejected",
  "delivery_failed",
  "failed",
  "cancelled",
]);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function splitLines(value) {
  return String(value || "")
    .split(/\r?\n|；|;/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(value = new Date().toISOString()) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "NOW";
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function showToast(message, error = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.className = `toast show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => {
    elements.toast.className = "toast";
    elements.toast.textContent = "";
  }, 4200);
}

async function parseResponse(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail || `请求失败（HTTP ${response.status}）`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function setBusy(button, busy, busyText) {
  if (!button.dataset.originalText) {
    button.dataset.originalText = button.textContent.trim();
  }
  button.disabled = busy;
  button.textContent = busy ? busyText : button.dataset.originalText;
}

function providerById(credentialId) {
  return providerCredentials.find(
    (item) => item.credential_id === credentialId,
  );
}

function modelOptions(credentialId, selectedModel = "") {
  const credential = providerById(credentialId);
  const models = credential?.models || [];
  if (!models.length) {
    return '<option value="">请先连接并发现模型</option>';
  }
  return models
    .map(
      (item) =>
        `<option value="${escapeHtml(item.id)}"${item.id === selectedModel ? " selected" : ""}>${escapeHtml(item.id)}</option>`,
    )
    .join("");
}

function credentialOptions(selectedCredential = "") {
  return providerCredentials
    .map(
      (item) =>
        `<option value="${escapeHtml(item.credential_id)}"${item.credential_id === selectedCredential ? " selected" : ""}>${escapeHtml(item.provider_name)} · ${escapeHtml(item.status)}</option>`,
    )
    .join("");
}

function firstModel(credentialId) {
  return providerById(credentialId)?.models?.[0]?.id || "";
}

function preferredInitialCredential() {
  if (uiConfig?.default_llm_mode === "deepseek") {
    const environment = providerCredentials.find(
      (item) => item.credential_id === "env:deepseek",
    );
    if (environment) return environment.credential_id;
  }
  return providerCredentials[0]?.credential_id || "builtin:mock";
}

function ensureRoutingDraft() {
  if (modelRoutingDraft) return;
  const credentialId = preferredInitialCredential();
  const route = {
    credential_id: credentialId,
    model: firstModel(credentialId) || "mock-deterministic",
  };
  modelRoutingDraft = {
    preset: "single",
    default_route: { ...route },
    agent_routes: Object.fromEntries(
      (uiConfig?.model_agents || []).map((agent) => [
        agent.agent_id,
        { ...route },
      ]),
    ),
  };
}

function updateRouteSummary() {
  const mode = elements.llmMode.value;
  if (mode === "mock") {
    elements.routeSummary.textContent =
      "当前使用确定性 Mock，不会产生 API 费用";
    elements.engineBadge.textContent = "MOCK";
    return;
  }
  if (mode === "deepseek") {
    elements.routeSummary.textContent =
      "全部 LLM Agent 使用服务端托管的 DeepSeek 配置";
    elements.engineBadge.textContent = "DEEPSEEK";
    return;
  }
  ensureRoutingDraft();
  const routes = [
    modelRoutingDraft.default_route,
    ...Object.values(modelRoutingDraft.agent_routes),
  ];
  const providers = new Set(
    routes.map((route) => providerById(route.credential_id)?.provider_name),
  );
  providers.delete(undefined);
  elements.routeSummary.textContent = `${uiConfig?.model_agents?.length || 0} 个 LLM Agent · ${providers.size || 1} 个供应商 · 运行时冻结快照`;
  elements.engineBadge.textContent = `MULTI × ${providers.size || 1}`;
}

function renderProviderPresetOptions() {
  const catalog = uiConfig?.model_providers || [];
  elements.providerPreset.innerHTML = catalog
    .map(
      (item) =>
        `<option value="${escapeHtml(item.preset_id)}">${escapeHtml(item.name)}</option>`,
    )
    .join("");
  if (catalog.length) {
    elements.providerPreset.value =
      catalog.some((item) => item.preset_id === "zzshu")
        ? "zzshu"
        : catalog[0].preset_id;
    applyProviderPreset();
  }
}

function applyProviderPreset() {
  const preset = (uiConfig?.model_providers || []).find(
    (item) => item.preset_id === elements.providerPreset.value,
  );
  if (!preset) return;
  elements.providerName.value = preset.name;
  elements.providerBaseUrl.value =
    preset.base_url === "https://" ? "" : preset.base_url;
  elements.providerJsonMode.checked = Boolean(preset.supports_json_object);
}

function renderCredentials() {
  elements.credentialCount.textContent = `${providerCredentials.length} 个`;
  elements.credentialList.innerHTML = providerCredentials
    .map(
      (credential) => `
        <article class="credential-card" data-credential-card="${escapeHtml(credential.credential_id)}">
          <div class="credential-icon">${escapeHtml(credential.provider_name.slice(0, 1))}</div>
          <div>
            <strong>${escapeHtml(credential.provider_name)}</strong>
            <small>${escapeHtml(credential.base_url)}</small>
            <span>${escapeHtml(credential.secret_hint)} · ${credential.models.length} 个模型 · ${escapeHtml(credential.note)}</span>
          </div>
          <div class="credential-actions">
            <button type="button" data-refresh-credential="${escapeHtml(credential.credential_id)}" aria-label="刷新模型列表">↻</button>
            ${
              credential.source === "session"
                ? `<button type="button" data-delete-credential="${escapeHtml(credential.credential_id)}" aria-label="删除凭证">×</button>`
                : '<em>托管</em>'
            }
          </div>
        </article>
      `,
    )
    .join("");
}

function renderRouting() {
  ensureRoutingDraft();
  const fallbackCredential = preferredInitialCredential();
  if (!providerById(modelRoutingDraft.default_route.credential_id)) {
    modelRoutingDraft.default_route = {
      credential_id: fallbackCredential,
      model: firstModel(fallbackCredential),
    };
  }
  elements.routingPreset.value = modelRoutingDraft.preset;
  elements.defaultCredential.innerHTML = credentialOptions(
    modelRoutingDraft.default_route.credential_id,
  );
  elements.defaultModel.innerHTML = modelOptions(
    modelRoutingDraft.default_route.credential_id,
    modelRoutingDraft.default_route.model,
  );
  if (!modelRoutingDraft.default_route.model) {
    modelRoutingDraft.default_route.model = elements.defaultModel.value;
  }
  elements.agentRoutingList.innerHTML = (uiConfig?.model_agents || [])
    .map((agent) => {
      let route =
        modelRoutingDraft.agent_routes[agent.agent_id] ||
        modelRoutingDraft.default_route;
      if (!providerById(route.credential_id)) {
        route = { ...modelRoutingDraft.default_route };
        modelRoutingDraft.agent_routes[agent.agent_id] = route;
      }
      return `
        <article class="agent-route" data-agent-route="${escapeHtml(agent.agent_id)}">
          <div class="route-agent-identity">
            <span>${escapeHtml(agent.name.slice(0, 1))}</span>
            <div>
              <strong>${escapeHtml(agent.name)}</strong>
              <small>${escapeHtml(agent.profile)}</small>
            </div>
          </div>
          <label>
            <span>供应商凭证</span>
            <select data-route-credential="${escapeHtml(agent.agent_id)}">
              ${credentialOptions(route.credential_id)}
            </select>
          </label>
          <label>
            <span>模型</span>
            <select data-route-model="${escapeHtml(agent.agent_id)}">
              ${modelOptions(route.credential_id, route.model)}
            </select>
          </label>
        </article>
      `;
    })
    .join("");
  updateRouteSummary();
}

async function refreshCredentials() {
  const response = await fetch("/model-providers/credentials", {
    cache: "no-store",
  });
  providerCredentials = await parseResponse(response);
  renderCredentials();
  renderRouting();
}

async function connectProvider(event) {
  event.preventDefault();
  setBusy(elements.connectProviderButton, true, "正在验证连接…");
  elements.routingWarnings.innerHTML = "";
  try {
    const response = await fetch("/model-providers/credentials", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset_id: elements.providerPreset.value,
        display_name: elements.providerName.value.trim(),
        base_url: elements.providerBaseUrl.value.trim(),
        api_key: elements.providerApiKey.value,
        manual_model: elements.providerManualModel.value.trim() || null,
        supports_json_object: elements.providerJsonMode.checked,
      }),
    });
    const credential = await parseResponse(response);
    elements.providerApiKey.value = "";
    elements.providerManualModel.value = "";
    await refreshCredentials();
    modelRoutingDraft.default_route = {
      credential_id: credential.credential_id,
      model: credential.models[0]?.id || "",
    };
    applyRoutingPresetToDraft();
    renderRouting();
    showToast(`${credential.provider_name} 已连接，API Key 未写入浏览器存储`);
  } catch (error) {
    elements.routingWarnings.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    showToast(error.message, true);
  } finally {
    setBusy(elements.connectProviderButton, false, "");
  }
}

function applyRoutingPresetToDraft() {
  ensureRoutingDraft();
  const agents = uiConfig?.model_agents || [];
  const realCredentials = providerCredentials.filter(
    (item) => item.credential_id !== "builtin:mock",
  );
  const primary =
    providerById(modelRoutingDraft.default_route.credential_id) ||
    realCredentials[0] ||
    providerCredentials[0];
  const secondary =
    realCredentials.find(
      (item) => item.credential_id !== primary?.credential_id,
    ) || primary;
  const routeFor = (credential) => ({
    credential_id: credential.credential_id,
    model: credential.models[0]?.id || "",
  });
  if (!primary) return;
  modelRoutingDraft.default_route = routeFor(primary);
  if (modelRoutingDraft.preset === "custom") return;
  modelRoutingDraft.agent_routes = Object.fromEntries(
    agents.map((agent, index) => {
      let credential = primary;
      if (modelRoutingDraft.preset === "balanced" && realCredentials.length) {
        credential = realCredentials[index % realCredentials.length];
      }
      if (
        modelRoutingDraft.preset === "cross_review" &&
        ["QualityAssuranceAgent", "ReleaseReviewerAgent"].includes(
          agent.agent_id,
        )
      ) {
        credential = secondary;
      }
      return [agent.agent_id, routeFor(credential)];
    }),
  );
}

function applyRoutingPreset() {
  ensureRoutingDraft();
  modelRoutingDraft.preset = elements.routingPreset.value;
  applyRoutingPresetToDraft();
  renderRouting();
}

async function saveRouting() {
  ensureRoutingDraft();
  setBusy(elements.saveRoutingButton, true, "正在校验路由…");
  try {
    const response = await fetch("/model-routing/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(modelRoutingDraft),
    });
    const result = await parseResponse(response);
    modelRoutingDraft = result.routing;
    elements.routingWarnings.innerHTML =
      '<p class="routing-ok">路由校验通过；项目创建时将冻结当前配置。</p>' +
      result.warnings
        .map((warning) => `<p>建议：${escapeHtml(warning)}</p>`)
        .join("");
    elements.llmMode.value = "routed";
    updateRouteSummary();
    showToast("Agent 级多供应商路由已启用");
  } catch (error) {
    elements.routingWarnings.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    showToast(error.message, true);
    return;
  } finally {
    setBusy(elements.saveRoutingButton, false, "");
  }
}

async function handleCredentialAction(event) {
  const refreshButton = event.target.closest("[data-refresh-credential]");
  const deleteButton = event.target.closest("[data-delete-credential]");
  if (!refreshButton && !deleteButton) return;
  const credentialId =
    refreshButton?.dataset.refreshCredential ||
    deleteButton?.dataset.deleteCredential;
  try {
    if (refreshButton) {
      const response = await fetch(
        `/model-providers/credentials/${encodeURIComponent(credentialId)}/refresh`,
        { method: "POST" },
      );
      await parseResponse(response);
      showToast("模型列表已刷新");
    } else {
      const response = await fetch(
        `/model-providers/credentials/${encodeURIComponent(credentialId)}`,
        { method: "DELETE" },
      );
      if (!response.ok) await parseResponse(response);
      if (modelRoutingDraft) {
        if (modelRoutingDraft.default_route.credential_id === credentialId) {
          modelRoutingDraft.default_route = {
            credential_id: "builtin:mock",
            model: "mock-deterministic",
          };
        }
        Object.values(modelRoutingDraft.agent_routes).forEach((route) => {
          if (route.credential_id === credentialId) {
            route.credential_id = "builtin:mock";
            route.model = "mock-deterministic";
          }
        });
      }
      showToast("凭证已从服务进程中清除");
    }
    await refreshCredentials();
  } catch (error) {
    showToast(error.message, true);
  }
}

function handleAgentRouteChange(event) {
  const agentId =
    event.target.dataset.routeCredential || event.target.dataset.routeModel;
  if (!agentId) return;
  ensureRoutingDraft();
  modelRoutingDraft.preset = "custom";
  elements.routingPreset.value = "custom";
  const row = event.target.closest("[data-agent-route]");
  const credentialSelect = row.querySelector("[data-route-credential]");
  const modelSelect = row.querySelector("[data-route-model]");
  if (event.target.dataset.routeCredential) {
    modelSelect.innerHTML = modelOptions(credentialSelect.value);
  }
  modelRoutingDraft.agent_routes[agentId] = {
    credential_id: credentialSelect.value,
    model: modelSelect.value,
  };
  updateRouteSummary();
}

function listHtml(values, emptyText = "暂无") {
  const items = Array.isArray(values) ? values : [];
  if (!items.length) return `<li>${escapeHtml(emptyText)}</li>`;
  return items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function cardsHtml(values, renderer) {
  const items = Array.isArray(values) ? values : [];
  if (!items.length) return `<p class="empty-state">暂无内容</p>`;
  return items.map(renderer).join("");
}

function renderAgentRoster() {
  const agents = (uiConfig?.agents || []).filter((agent) => !agent.hidden);
  elements.agentCount.textContent = String(agents.length || 16);
  elements.agentRoster.innerHTML = agents
    .map(
      (agent) => `
        <article class="agent-card${latestTraceAgent === agent.id ? " active" : ""}" data-agent="${escapeHtml(agent.id)}">
          <i style="--agent-color:${escapeHtml(agent.color)}">${escapeHtml(agent.badge)}</i>
          <div>
            <strong>${escapeHtml(agent.name)} · ${escapeHtml(agent.role)}</strong>
            <small>${escapeHtml(agent.intro)}</small>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderMessages(messages) {
  const received = Array.isArray(messages) ? messages : [];
  const entries = received.length
    ? received
    : [
        {
          role: "product_manager",
          author: "产品经理·言澈",
          content:
            "你好，我已经在线。你可以直接描述想做的软件，也可以先上传需求文件；我会逐步追问业务目标、用户流程、范围边界、预算和部署条件。",
          evidence_ids: [],
        },
      ];
  elements.chatMessages.innerHTML = entries
    .map((message) => {
      const roleClass =
        message.role === "client"
          ? "message-client"
          : message.role === "system"
            ? "message-system"
            : "message-product";
      const evidence = (message.evidence_ids || [])
        .map((id) => `<i>${escapeHtml(id)}</i>`)
        .join("");
      return `
        <article class="message ${roleClass}">
          <span>${escapeHtml(message.author)}</span>
          <p>${escapeHtml(message.content)}</p>
          ${evidence ? `<div class="message-evidence">${evidence}</div>` : ""}
        </article>
      `;
    })
    .join("");
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function renderAssets(assets) {
  const items = Array.isArray(assets) ? assets : [];
  elements.documentCount.textContent = String(items.length);
  if (!items.length) {
    elements.assetList.innerHTML = `<p class="empty-state">还没有上传资料</p>`;
    return;
  }
  elements.assetList.innerHTML = items
    .map((asset) => {
      const extension = asset.original_name.split(".").pop()?.toUpperCase() || "DOC";
      return `
        <article class="asset-item">
          <span>${escapeHtml(extension.slice(0, 4))}</span>
          <div>
            <strong title="${escapeHtml(asset.original_name)}">${escapeHtml(asset.original_name)}</strong>
            <small>${formatBytes(asset.size)} · ${asset.chunk_count} chunks · ${escapeHtml(asset.parser)}</small>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderUnderstanding(understanding) {
  const value = understanding || {
    summary: "等待客户需求对话。",
    confirmed_goals: [],
    constraints: [],
    open_questions: ["请先描述业务问题"],
    ready_for_workshop: false,
  };
  elements.understandingSummary.textContent = value.summary || "等待客户需求对话。";
  elements.confirmedGoals.innerHTML = listHtml(value.confirmed_goals, "尚未确认");
  elements.understoodConstraints.innerHTML = listHtml(value.constraints, "尚未确认");
  elements.openQuestions.innerHTML = listHtml(value.open_questions, "关键问题已澄清");

  const goals = value.confirmed_goals?.length || 0;
  const questions = value.open_questions?.length || 0;
  const percentage = !projectSession
    ? 0
    : value.ready_for_workshop
      ? 100
      : Math.min(85, goals * 22 + (questions ? 10 : 25));
  elements.readinessBadge.textContent = `${percentage}%`;
  const locked = Boolean(projectSession?.run_id);
  elements.launchWorkshopButton.disabled =
    !projectSession || !value.ready_for_workshop || locked;
  elements.pmStatus.textContent = !projectSession
    ? "在线 · 可以直接开始沟通"
    : locked
      ? "需求已冻结，团队正在执行"
      : value.ready_for_workshop
        ? "需求理解完成，可召开评审会"
        : "正在与你澄清需求";
}

function renderSession(session) {
  projectSession = session;
  const exists = Boolean(session);
  elements.currentProjectName.textContent = exists
    ? session.project_name
    : "尚未命名的软件项目";
  elements.sessionId.textContent = exists
    ? session.session_id
    : "AUTO-CREATE ON FIRST MESSAGE";
  elements.sessionBadge.textContent = exists
    ? session.status.toUpperCase()
    : "首次操作自动创建";
  elements.fileInput.disabled = Boolean(session?.run_id);
  elements.chatInput.disabled = Boolean(session?.run_id);
  elements.sendMessageButton.disabled = Boolean(session?.run_id);
  elements.chatInput.placeholder = exists
    ? session.run_id
      ? "需求已冻结；请在客户决策台处理后续确认。"
      : "继续告诉产品经理业务目标、流程、边界和部署条件…"
    : "例如：我想做一个社区预约系统，居民可以选择事项和时间，工作人员可以确认或改期……";
  renderMessages(session?.messages || []);
  renderAssets(session?.assets || []);
  renderUnderstanding(session?.understanding);
  if (exists) {
    elements.engineBadge.textContent =
      session.llm_mode === "routed"
        ? "MULTI ROUTER"
        : session.llm_mode.toUpperCase();
  }
}

function appendActivity(agent, summary, occurredAt = null) {
  const article = document.createElement("article");
  const short = String(agent || "系统").slice(0, 2);
  article.innerHTML = `
    <span>${escapeHtml(short)}</span>
    <p><strong>${escapeHtml(agent || "系统")}</strong><br>${escapeHtml(summary || "")}</p>
    <time>${formatTime(occurredAt || new Date().toISOString())}</time>
  `;
  elements.activityList.prepend(article);
  while (elements.activityList.children.length > 28) {
    elements.activityList.lastElementChild.remove();
  }
}

function stageForStatus(status) {
  const map = {
    queued: "meeting",
    running: "meeting",
    retrieving_product_knowledge: "product",
    drafting_product_requirements: "product",
    holding_requirements_workshop: "meeting",
    awaiting_requirements_approval: "approval",
    revising_requirements: "product",
    planning_project: "plan",
    designing_architecture: "plan",
    generating_project: "development",
    preparing_deployment: "development",
    testing_project: "qa",
    qa_failed: "development",
    reviewing_release: "qa",
    awaiting_release_approval: "release",
    resuming: "release",
    writing_project: "delivery",
    running_quality_gate: "delivery",
    preparing_delivery: "delivery",
    completed: "delivery",
  };
  return map[status] || "product";
}

function renderPipeline(status) {
  const order = [
    "product",
    "meeting",
    "approval",
    "plan",
    "development",
    "qa",
    "release",
    "delivery",
  ];
  const active = stageForStatus(status);
  const activeIndex = order.indexOf(active);
  elements.pipeline.querySelectorAll("[data-stage]").forEach((card) => {
    const index = order.indexOf(card.dataset.stage);
    card.classList.toggle("active", index === activeIndex && status !== "completed");
    card.classList.toggle("done", status === "completed" || index < activeIndex);
  });
}

function renderMeeting(workshop, status) {
  const hasWorkshop = Boolean(workshop);
  elements.meetingSection.classList.toggle(
    "is-live",
    ["holding_requirements_workshop", "planning_project"].includes(status),
  );
  elements.meetingStatus.textContent = hasWorkshop
    ? "会议纪要已形成"
    : status === "holding_requirements_workshop"
      ? "会议进行中"
      : "等待产品经理准备";
  elements.meetingScreenTitle.textContent = hasWorkshop
    ? workshop.meeting_title
    : "CLIENT INTENT";
  elements.meetingScreenText.textContent = hasWorkshop
    ? workshop.client_intent_summary
    : status === "holding_requirements_workshop"
      ? "交付总监正在组织跨职能评审…"
      : "会议尚未开始";
  renderMeetingTab(workshop);
}

function renderMeetingTab(workshop = currentRun?.state?.requirements_workshop) {
  if (!workshop) {
    elements.meetingOutput.innerHTML = `<p class="empty-state">会议开始后，这里会展示真实的服务端会议纪要。</p>`;
    return;
  }
  if (activeMeetingTab === "risks") {
    elements.meetingOutput.innerHTML = cardsHtml(
      workshop.risks,
      (risk) => `
        <article class="risk-card">
          <strong>${escapeHtml(risk.risk_id)} · ${escapeHtml(risk.category)} · ${escapeHtml(risk.scenario)}</strong>
          <p><b>影响：</b>${escapeHtml(risk.impact)}<br><b>规避：</b>${escapeHtml(risk.avoidance_strategy)}</p>
          <small>预警：${escapeHtml(risk.early_warning)} · Owner：${escapeHtml(risk.owner)}</small>
        </article>
      `,
    );
    return;
  }
  if (activeMeetingTab === "breakdown") {
    elements.meetingOutput.innerHTML = cardsHtml(
      workshop.decomposition,
      (item) => `
        <article class="document-card">
          <strong>${escapeHtml(item.item_id)} · ${escapeHtml(item.requirement_id)} · ${escapeHtml(item.workstream)}</strong>
          <p>${escapeHtml(item.deliverable)}</p>
          <small>${escapeHtml((item.done_definition || []).join("；"))}</small>
        </article>
      `,
    );
    return;
  }
  if (activeMeetingTab === "boundaries") {
    elements.meetingOutput.innerHTML = `
      <article class="document-card">
        <strong>交付边界</strong>
        <p>${escapeHtml((workshop.delivery_boundaries || []).join("\n"))}</p>
      </article>
      <article class="document-card">
        <strong>测试策略</strong>
        <p>${escapeHtml((workshop.testing_strategy || []).join("\n"))}</p>
      </article>
      <article class="document-card">
        <strong>仍未解决</strong>
        <p>${escapeHtml((workshop.unresolved_questions || []).join("\n") || "无")}</p>
      </article>
    `;
    return;
  }
  elements.meetingOutput.innerHTML = `
    ${cardsHtml(workshop.decisions, (decision, index) => `
      <article class="document-card"><strong>DECISION ${String(index + 1).padStart(2, "0")}</strong><p>${escapeHtml(decision)}</p></article>
    `)}
    <article class="document-card"><strong>会议结论</strong><p>${escapeHtml(workshop.readiness_recommendation)}</p></article>
  `;
}

function renderPrd(prd) {
  if (!prd) return `<p class="empty-state">PRD 尚未生成</p>`;
  return `
    <article class="document-card">
      <strong>${escapeHtml(prd.product_name)} · 产品愿景</strong>
      <p>${escapeHtml(prd.vision)}</p>
    </article>
    <article class="document-card">
      <strong>业务问题</strong>
      <p>${escapeHtml(prd.problem_statement)}</p>
    </article>
    ${cardsHtml(prd.functional_requirements, (item) => `
      <article class="document-card">
        <strong>${escapeHtml(item.requirement_id)} · ${escapeHtml(item.title)} · ${escapeHtml(item.priority)}</strong>
        <p>${escapeHtml(item.description)}\n\n验收：${escapeHtml((item.acceptance_criteria || []).join("；"))}</p>
      </article>
    `)}
    <article class="document-card">
      <strong>范围边界</strong>
      <p>范围内：${escapeHtml((prd.in_scope || []).join("；"))}\n\n范围外：${escapeHtml((prd.out_of_scope || []).join("；"))}</p>
    </article>
  `;
}

function renderQa(report) {
  if (!report) return `<p class="empty-state">测试报告尚未生成</p>`;
  return `
    <article class="test-card">
      <strong>STATIC QA GATE · ${report.passed ? "PASS" : "FAIL"}</strong>
      <p>${escapeHtml(report.summary)}</p>
      <small>隔离运行验证：${report.runtime_verified ? "已完成" : "未完成"} · 目标环境：${report.runtime_validation_required ? "仍需执行" : "无需补充"}</small>
    </article>
    ${cardsHtml(report.test_cases, (test) => `
      <article class="test-card">
        <strong>${escapeHtml(test.case_id)} · ${escapeHtml(test.requirement_id)} · ${escapeHtml(test.status)}</strong>
        <p>${escapeHtml(test.name)}\n${escapeHtml(test.evidence)}</p>
        <small>${escapeHtml(test.method)}</small>
      </article>
    `)}
    <article class="document-card">
      <strong>环境限制与建议命令</strong>
      <p>${escapeHtml([...(report.environment_limitations || []), ...(report.recommended_commands || [])].join("\n"))}</p>
    </article>
  `;
}

function renderGuide(guide) {
  if (!guide) return `<p class="empty-state">部署指南尚未生成</p>`;
  const sections = [
    ["代码与运行环境", guide.environment_specification],
    ["前置条件", guide.prerequisites],
    ["环境变量", guide.environment_variables],
    ["本地运行", guide.local_run_steps],
    ["构建步骤", guide.build_steps],
    ["测试步骤", guide.test_steps],
    ["生产部署", guide.production_steps],
    ["健康检查", guide.health_checks],
    ["上线条件", guide.release_conditions],
    ["使用指南", guide.usage_guide],
    ["故障排查", guide.troubleshooting],
    ["备份恢复", guide.backup_restore],
    ["可观测性", guide.observability],
    ["安全清单", guide.security_checklist],
    ["回滚步骤", guide.rollback_steps],
    ["交付边界", guide.boundary_conditions],
    ["移交清单", guide.handover_checklist],
  ];
  return `
    <article class="document-card"><strong>目标环境</strong><p>${escapeHtml(guide.target)}</p></article>
    ${sections
      .filter(([, values]) => values?.length)
      .map(
        ([title, values]) => `
          <h3>${escapeHtml(title)}</h3>
          <ol>${values.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ol>
        `,
      )
      .join("")}
  `;
}

function renderArtifact() {
  const state = currentRun?.state || {};
  const content = {
    prd: renderPrd(state.product_requirements),
    workshop: state.requirements_workshop
      ? `<article class="document-card"><strong>${escapeHtml(state.requirements_workshop.meeting_title)}</strong><p>${escapeHtml(state.requirements_workshop.client_intent_summary)}</p></article>${cardsHtml(state.requirements_workshop.risks, (risk) => `<article class="risk-card"><strong>${escapeHtml(risk.risk_id)} · ${escapeHtml(risk.scenario)}</strong><p>${escapeHtml(risk.avoidance_strategy)}</p><small>${escapeHtml(risk.owner)}</small></article>`)}`
      : `<p class="empty-state">会议纪要尚未生成</p>`,
    plan: state.project_plan
      ? `<article class="document-card"><strong>${escapeHtml(state.project_plan.methodology)}</strong><p>${escapeHtml(state.project_plan.executive_summary)}</p></article>${cardsHtml(state.project_plan.phases, (phase) => `<article class="document-card"><strong>${escapeHtml(phase.phase_id)} · ${escapeHtml(phase.name)} · ${phase.estimated_days} 天</strong><p>${escapeHtml(phase.objective)}\n${escapeHtml((phase.deliverables || []).join("；"))}</p><small>预算：${phase.budget_cny ?? "待确认"} CNY</small></article>`)}`
      : `<p class="empty-state">项目规划尚未生成</p>`,
    architecture: state.architecture
      ? `<article class="document-card"><strong>${escapeHtml(state.architecture.primary_language)} · ${escapeHtml(state.architecture.architecture_style)}</strong><p>${escapeHtml((state.architecture.rationale || []).join("\n"))}</p></article>${cardsHtml(state.architecture.components, (component) => `<article class="document-card"><strong>${escapeHtml(component.name)} · ${escapeHtml(component.technology)}</strong><p>${escapeHtml(component.responsibility)}</p><small>${escapeHtml((component.interfaces || []).join("；"))}</small></article>`)}`
      : `<p class="empty-state">架构尚未生成</p>`,
    code: state.generated_project
      ? `<article class="document-card"><strong>代码摘要</strong><p>${escapeHtml(state.generated_project.summary)}</p></article>${cardsHtml(state.generated_project.files, (file) => `<article class="file-card"><strong>${escapeHtml(file.path)}</strong><p>${escapeHtml(file.purpose)}</p><pre>${escapeHtml(file.content)}</pre></article>`)}`
      : `<p class="empty-state">代码尚未生成</p>`,
    qa: renderQa(state.qa_report),
    guide: renderGuide(state.deployment_guide),
  };
  elements.artifactPanel.innerHTML = content[activeArtifact] || content.prd;
}

function renderApproval(run) {
  const state = run?.state || {};
  const status = run?.status || "idle";
  const awaitingRequirements = status === "awaiting_requirements_approval";
  const awaitingRelease = status === "awaiting_release_approval";
  const awaiting = awaitingRequirements || awaitingRelease;
  currentCheckpoint = awaitingRequirements
    ? "requirements_approval"
    : awaitingRelease
      ? "release_approval"
      : null;

  elements.approvalComment.disabled = !awaiting;
  elements.rejectButton.disabled = !awaiting;
  elements.approveButton.disabled = !awaiting;
  elements.approvalDetails.open = awaiting;
  elements.approvalCard.dataset.state =
    status === "completed" ? "approved" : awaiting ? "awaiting" : "idle";

  if (awaitingRequirements) {
    const workshop = state.requirements_workshop || {};
    elements.approvalState.textContent = "等待客户确认 PRD";
    elements.approvalSummary.textContent =
      "请同时检查产品经理的 PRD 与交付总监会议结论。批准前，架构和开发不会开工。";
    elements.approvalMeta.textContent = `${workshop.risks?.length || 0} 项风险 · ${workshop.decomposition?.length || 0} 个工作包`;
    elements.approvalMaterial.innerHTML =
      renderPrd(state.product_requirements) +
      `<h3>会议避坑清单</h3>` +
      cardsHtml(
        workshop.risks,
        (risk) => `<article class="risk-card"><strong>${escapeHtml(risk.risk_id)} · ${escapeHtml(risk.scenario)}</strong><p>${escapeHtml(risk.avoidance_strategy)}</p><small>Owner：${escapeHtml(risk.owner)}</small></article>`,
      );
    elements.rejectButton.textContent = "退回产品与会议组";
    elements.approveButton.textContent = "确认需求，进入研发";
    return;
  }

  if (awaitingRelease) {
    const files = state.generated_project?.files || [];
    elements.approvalState.textContent = "等待客户最终验收";
    elements.approvalSummary.textContent =
      "测试团队和独立 Reviewer 已完成发布前检查。请核对源码、测试限制、部署环境与交付边界后决定。";
    elements.approvalMeta.textContent = `${files.length} 个源码文件 · 静态 QA ${state.qa_report?.passed ? "PASS" : "FAIL"} · 运行验证${state.qa_report?.runtime_verified ? "已完成" : "未完成"} · ${state.architecture?.primary_language || ""}`;
    elements.approvalMaterial.innerHTML =
      renderQa(state.qa_report) +
      `<h3>待交付源码</h3>` +
      cardsHtml(
        files,
        (file) => `<article class="file-card"><strong>${escapeHtml(file.path)}</strong><p>${escapeHtml(file.purpose)}</p></article>`,
      ) +
      `<h3>部署边界</h3>` +
      `<article class="document-card"><p>${escapeHtml((state.deployment_guide?.boundary_conditions || []).join("\n"))}</p></article>`;
    elements.rejectButton.textContent = "退回开发与测试团队";
    elements.approveButton.textContent = "验收通过，生成 ZIP";
    return;
  }

  if (status === "completed") {
    elements.approvalState.textContent = "客户已验收 · ZIP 已生成";
    elements.approvalSummary.textContent =
      "源码、项目文档、测试报告、部署指南与清单已经完成打包。";
    elements.approvalMeta.textContent = currentRun.state.delivery_archive?.archive_name || "ZIP READY";
    elements.approvalMaterial.innerHTML = `<article class="document-card"><strong>最终客户决定</strong><p>${escapeHtml(state.release_approval_comment || "批准交付")}</p></article>`;
    return;
  }

  if (["release_rejected", "delivery_failed", "failed"].includes(status)) {
    elements.approvalState.textContent =
      status === "release_rejected" ? "客户已退回" : "交付被安全门禁阻止";
    elements.approvalSummary.textContent =
      status === "release_rejected"
        ? "没有生成 ZIP；请根据客户意见重新建立变更任务。"
        : state.error?.message || "请检查测试、质量门禁或工作流错误。";
    return;
  }

  elements.approvalState.textContent =
    status === "idle" ? "等待工作流" : "团队执行中";
  elements.approvalSummary.textContent =
    "PRD 与会议纪要完成后进行第一次确认；测试完成后进行最终交付验收。";
  elements.approvalMeta.textContent = "尚无待决策材料";
  elements.approvalMaterial.innerHTML = `<p class="empty-state">等待产品与研发团队交付材料</p>`;
}

function renderDelivery(run) {
  const packageInfo = run?.state?.delivery_archive;
  if (run?.status !== "completed" || !packageInfo) {
    elements.downloadButton.classList.add("disabled");
    elements.downloadButton.removeAttribute("href");
    elements.downloadButton.setAttribute("aria-disabled", "true");
    elements.downloadMeta.textContent = "等待项目完成";
    return;
  }
  elements.downloadButton.classList.remove("disabled");
  elements.downloadButton.href = `/runs/${encodeURIComponent(run.run_id)}/download`;
  elements.downloadButton.removeAttribute("aria-disabled");
  elements.downloadButton.setAttribute("download", packageInfo.archive_name);
  elements.downloadMeta.textContent = `${packageInfo.file_count} 个文件 · ${formatBytes(packageInfo.size)} · SHA-256 ${packageInfo.sha256.slice(0, 12)}…`;
  elements.deliverySummary.textContent =
    "交付包已完成：包含源码、PRD、需求避坑会议纪要、项目规划、架构、测试报告、详细部署指南和完整性清单。";
}

function renderRun(run) {
  currentRun = run;
  const status = run?.status || "idle";
  elements.runStatus.textContent = status.toUpperCase();
  elements.phaseMetric.textContent = phaseLabels[status] || status;
  elements.systemText.textContent =
    terminalStatuses.has(status) ? "本次软件公司工作流已结束" : "真实事件流已连接";
  renderPipeline(status);
  renderMeeting(run?.state?.requirements_workshop, status);
  renderApproval(run);
  renderArtifact();
  renderDelivery(run);
}

async function refreshRun() {
  if (!currentRun?.run_id) return;
  const response = await fetch(`/runs/${encodeURIComponent(currentRun.run_id)}`);
  renderRun(await parseResponse(response));
}

function connectEvents(runId) {
  eventSource?.close();
  eventSource = new EventSource(`/runs/${encodeURIComponent(runId)}/events`);
  eventSource.addEventListener("agent_trace", (event) => {
    const payload = JSON.parse(event.data);
    latestTraceAgent = payload.agent;
    renderAgentRoster();
    appendActivity(payload.agent, payload.summary, payload.occurred_at);
  });
  eventSource.addEventListener("human_decision", (event) => {
    const payload = JSON.parse(event.data);
    appendActivity(
      "客户",
      `${payload.approved ? "批准" : "退回"}：${payload.comment || "未填写说明"}`,
      payload.occurred_at,
    );
  });
  eventSource.addEventListener("run_error", (event) => {
    const payload = JSON.parse(event.data);
    showToast(payload.message || "工作流发生安全错误", true);
  });
  eventSource.addEventListener("run_status", async (event) => {
    const payload = JSON.parse(event.data);
    elements.runStatus.textContent = String(payload.status || "").toUpperCase();
    elements.phaseMetric.textContent = phaseLabels[payload.status] || payload.status;
    try {
      await refreshRun();
    } catch (error) {
      showToast(error.message, true);
    }
    if (terminalStatuses.has(payload.status)) {
      eventSource?.close();
    }
  });
  eventSource.onerror = () => {
    if (!terminalStatuses.has(currentRun?.status)) {
      elements.systemText.textContent = "事件流正在重连…";
    }
  };
}

function inferProjectName(seedText = "") {
  const explicitName = elements.projectName.value.trim();
  if (explicitName) return explicitName;
  const compact = String(seedText)
    .replace(/\.[a-z0-9]{1,8}$/i, "")
    .replace(/\s+/g, " ")
    .replace(/^(我想做|我们想做|我需要|我们需要|需求是|项目是)[：:\s]*/u, "")
    .trim();
  if (compact.length >= 2) {
    return compact.slice(0, 28);
  }
  return "客户新软件项目";
}

function projectSessionPayload(seedText = "") {
  const constraints = splitLines(elements.constraints.value);
  const routed = elements.llmMode.value === "routed";
  if (routed) ensureRoutingDraft();
  return {
    project_name: inferProjectName(seedText),
    target_users: elements.targetUsers.value.trim() || "待通过对话确认",
    budget_cny: elements.budgetCny.value
      ? Number(elements.budgetCny.value)
      : null,
    delivery_deadline:
      elements.deliveryDeadline.value.trim() || "待通过对话确认",
    policy_constraints: constraints,
    business_constraints: [],
    preferred_stack: elements.preferredStack.value,
    deployment_target: elements.deploymentTarget.value,
    deployment_conditions:
      "由产品经理在对话中澄清，并由 DevOps 在会议后形成详细指南",
    llm_mode: elements.llmMode.value,
    model_routing: routed ? modelRoutingDraft : null,
  };
}

async function ensureProjectSession(seedText = "") {
  if (projectSession) return projectSession;
  elements.pmStatus.textContent = "正在建立项目会话…";
  const response = await fetch("/project-sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(projectSessionPayload(seedText)),
  });
  const session = await parseResponse(response);
  renderSession(session);
  elements.setupCard.style.display = "none";
  appendActivity(
    "系统",
    `已自动建立「${session.project_name}」项目会话，产品经理开始接洽。`,
  );
  return session;
}

async function createSession(event) {
  event.preventDefault();
  setBusy(elements.createSessionButton, true, "正在保存并建立会话…");
  try {
    const session = await ensureProjectSession(elements.projectName.value);
    elements.chatInput.focus();
    showToast(`「${session.project_name}」已建立，请直接与产品经理沟通`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(elements.createSessionButton, false, "");
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const content = elements.chatInput.value.trim();
  if (!content) return;
  elements.sendMessageButton.disabled = true;
  try {
    await ensureProjectSession(content);
    elements.chatInput.value = "";
    elements.pmStatus.textContent = "正在理解需求并检索项目资料…";
    const optimistic = {
      ...projectSession,
      messages: [
        ...projectSession.messages,
        {
          role: "client",
          author: "客户",
          content,
          evidence_ids: [],
        },
      ],
    };
    renderMessages(optimistic.messages);
    const response = await fetch(
      `/project-sessions/${encodeURIComponent(projectSession.session_id)}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      },
    );
    renderSession(await parseResponse(response));
  } catch (error) {
    elements.chatInput.value = content;
    showToast(error.message, true);
  } finally {
    elements.sendMessageButton.disabled = Boolean(projectSession?.run_id);
  }
}

async function uploadFiles(event) {
  const files = [...event.target.files];
  event.target.value = "";
  if (!files.length) return;
  elements.fileInput.disabled = true;
  try {
    await ensureProjectSession(files[0].name);
  } catch (error) {
    elements.fileInput.disabled = false;
    showToast(`无法建立项目会话：${error.message}`, true);
    return;
  }
  for (const file of files) {
    appendActivity("项目资料库", `正在解析 ${file.name}…`);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(
        `/project-sessions/${encodeURIComponent(projectSession.session_id)}/files`,
        { method: "POST", body: form },
      );
      renderSession(await parseResponse(response));
      appendActivity("项目资料库", `${file.name} 已进入项目 RAG`);
      showToast(`${file.name} 已解析并进入项目知识库`);
    } catch (error) {
      appendActivity("项目资料库", `${file.name} 上传失败：${error.message}`);
      showToast(`${file.name}：${error.message}`, true);
    }
  }
  elements.fileInput.disabled = Boolean(projectSession?.run_id);
}

async function launchWorkshop() {
  if (!projectSession) return;
  setBusy(elements.launchWorkshopButton, true, "正在召集团队…");
  try {
    const response = await fetch(
      `/project-sessions/${encodeURIComponent(projectSession.session_id)}/launch`,
      { method: "POST" },
    );
    const run = await parseResponse(response);
    projectSession = { ...projectSession, run_id: run.run_id, status: "workshop_running" };
    renderSession(projectSession);
    renderRun(run);
    connectEvents(run.run_id);
    elements.meetingSection.scrollIntoView({ behavior: "smooth", block: "start" });
    appendActivity("交付总监·北辰", "已召集产品、架构、开发、测试、安全和运维参加需求评审。");
    showToast("需求评审会已开始，工作流会在客户确认点暂停");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.launchWorkshopButton.disabled = Boolean(projectSession?.run_id);
    elements.launchWorkshopButton.textContent = projectSession?.run_id
      ? "需求评审会已启动"
      : "召集需求评审会";
  }
}

async function submitDecision(approved) {
  if (!currentRun || !currentCheckpoint) return;
  const comment = elements.approvalComment.value.trim();
  if (!approved && !comment) {
    elements.approvalComment.focus();
    showToast("退回时必须说明需要修改什么", true);
    return;
  }
  elements.approveButton.disabled = true;
  elements.rejectButton.disabled = true;
  elements.approvalState.textContent = approved ? "正在提交确认…" : "正在提交退回…";
  try {
    const response = await fetch(
      `/runs/${encodeURIComponent(currentRun.run_id)}/decision`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          approved,
          comment,
          actor_id: "web-client",
          artifact_digest: currentRun.interrupt?.artifact_digest || null,
        }),
      },
    );
    renderRun(await parseResponse(response));
    elements.approvalComment.value = "";
    showToast(approved ? "客户决定已确认，团队继续工作" : "已退回相关团队修改");
  } catch (error) {
    showToast(error.message, true);
    await refreshRun().catch(() => {});
  }
}

function fillDemo() {
  elements.projectName.value = "社区公共服务预约平台";
  elements.targetUsers.value = "社区居民、社区服务人员和客户内网运维人员";
  elements.budgetCny.value = "180000";
  elements.deliveryDeadline.value = "8 周内完成可部署 MVP";
  elements.preferredStack.value = "Go";
  elements.deploymentTarget.value = "docker";
  elements.constraints.value =
    "居民敏感信息不得进入代码和日志\n数据不得离开客户内网\n首期不采购昂贵商业中间件";
  showToast("演示项目边界已填入；智能引擎选择保持不变");
}

function resetProject() {
  eventSource?.close();
  projectSession = null;
  currentRun = null;
  currentCheckpoint = null;
  latestTraceAgent = null;
  elements.setupCard.style.display = "";
  elements.sessionForm.reset();
  elements.llmMode.value = uiConfig?.default_llm_mode || "mock";
  modelRoutingDraft = null;
  ensureRoutingDraft();
  renderRouting();
  updateRouteSummary();
  renderSession(null);
  renderRun(null);
  renderAgentRoster();
  document
    .querySelector(".chat-panel")
    .scrollIntoView({ behavior: "smooth", block: "start" });
  elements.chatInput.focus();
}

function setupInteractions() {
  elements.sessionForm.addEventListener("submit", createSession);
  elements.chatForm.addEventListener("submit", sendMessage);
  elements.fileInput.addEventListener("change", uploadFiles);
  elements.launchWorkshopButton.addEventListener("click", launchWorkshop);
  elements.approveButton.addEventListener("click", () => submitDecision(true));
  elements.rejectButton.addEventListener("click", () => submitDecision(false));
  elements.fillDemo.addEventListener("click", fillDemo);
  elements.newProjectButton.addEventListener("click", resetProject);
  elements.providerForm.addEventListener("submit", connectProvider);
  elements.providerPreset.addEventListener("change", applyProviderPreset);
  elements.credentialList.addEventListener("click", handleCredentialAction);
  elements.agentRoutingList.addEventListener(
    "change",
    handleAgentRouteChange,
  );
  elements.routingPreset.addEventListener("change", () => {
    if (modelRoutingDraft) {
      modelRoutingDraft.preset = elements.routingPreset.value;
    }
  });
  elements.applyRoutingPreset.addEventListener(
    "click",
    applyRoutingPreset,
  );
  elements.defaultCredential.addEventListener("change", () => {
    ensureRoutingDraft();
    modelRoutingDraft.default_route = {
      credential_id: elements.defaultCredential.value,
      model: firstModel(elements.defaultCredential.value),
    };
    elements.defaultModel.innerHTML = modelOptions(
      elements.defaultCredential.value,
      modelRoutingDraft.default_route.model,
    );
    modelRoutingDraft.default_route.model = elements.defaultModel.value;
    modelRoutingDraft.preset = "custom";
    elements.routingPreset.value = "custom";
    updateRouteSummary();
  });
  elements.defaultModel.addEventListener("change", () => {
    ensureRoutingDraft();
    modelRoutingDraft.default_route.model = elements.defaultModel.value;
    modelRoutingDraft.preset = "custom";
    elements.routingPreset.value = "custom";
  });
  elements.saveRoutingButton.addEventListener("click", saveRouting);
  const showModelCenter = async () => {
    if (!elements.modelCenter.open) {
      elements.modelCenter.showModal();
    }
    try {
      await refreshCredentials();
    } catch (error) {
      elements.routingWarnings.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    }
  };
  elements.openModelCenter.addEventListener("click", showModelCenter);
  elements.configureModelsButton.addEventListener(
    "click",
    showModelCenter,
  );
  const closeModelCenterDialog = () => {
    elements.providerApiKey.value = "";
    elements.modelCenter.close();
  };
  elements.closeModelCenter.addEventListener(
    "click",
    closeModelCenterDialog,
  );
  elements.modelCenter.addEventListener("click", (event) => {
    if (event.target === elements.modelCenter) {
      closeModelCenterDialog();
    }
  });
  elements.llmMode.addEventListener("change", () => {
    if (elements.llmMode.value === "routed") {
      showModelCenter();
    }
    updateRouteSummary();
  });
  elements.chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!elements.sendMessageButton.disabled) {
        elements.chatForm.requestSubmit();
      }
    }
  });
  elements.promptChips.addEventListener("click", (event) => {
    const button = event.target.closest("[data-prompt]");
    if (!button || elements.chatInput.disabled) return;
    elements.chatInput.value = button.dataset.prompt;
    elements.chatInput.focus();
  });
  document.querySelectorAll(".meeting-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".meeting-tab").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      activeMeetingTab = button.dataset.meetingTab;
      renderMeetingTab();
    });
  });
  document.querySelectorAll(".artifact-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".artifact-tab").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      activeArtifact = button.dataset.artifact;
      renderArtifact();
    });
  });
  elements.themeToggle.addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    localStorage.setItem("repopilot-theme", next);
    elements.themeToggle.textContent = next === "light" ? "☾" : "☼";
  });
  elements.downloadButton.addEventListener("click", (event) => {
    if (elements.downloadButton.classList.contains("disabled")) {
      event.preventDefault();
      showToast("项目完成并通过客户验收后才能下载 ZIP", true);
    }
  });
}

async function boot() {
  const savedTheme = localStorage.getItem("repopilot-theme");
  if (savedTheme === "light" || savedTheme === "dark") {
    document.documentElement.dataset.theme = savedTheme;
    elements.themeToggle.textContent = savedTheme === "light" ? "☾" : "☼";
  }
  try {
    const [configResponse, healthResponse] = await Promise.all([
      fetch("/ui-config", { cache: "no-store" }),
      fetch("/health", { cache: "no-store" }),
    ]);
    uiConfig = await parseResponse(configResponse);
    const health = await parseResponse(healthResponse);
    Object.assign(phaseLabels, uiConfig.phases || {});
    renderProviderPresetOptions();
    await refreshCredentials();
    const hasManagedDeepSeek = providerCredentials.some(
      (item) => item.credential_id === "env:deepseek",
    );
    const deepseekOption = elements.llmMode.querySelector(
      'option[value="deepseek"]',
    );
    deepseekOption.disabled = !hasManagedDeepSeek;
    if (deepseekOption.disabled) {
      deepseekOption.textContent = "DeepSeek · 服务端未配置";
    }
    const defaultMode = uiConfig.default_llm_mode || "mock";
    elements.llmMode.value =
      defaultMode === "deepseek" && !hasManagedDeepSeek
        ? "mock"
        : defaultMode;
    updateRouteSummary();
    elements.systemText.textContent =
      `Agent 模型路由已就绪 · UI ${uiConfig.frontend_version || "0.6.0"} · 项目 RAG 独立运行`;
    renderAgentRoster();
  } catch (error) {
    elements.systemText.textContent = "后端连接失败";
    showToast(error.message, true);
  }
}

setupInteractions();
renderSession(null);
renderRun(null);
boot();
