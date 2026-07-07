"use strict";
/* Achilles web UI — a thin client on the engine/UI boundary (docs/protocol.md).
   One WebSocket per session (v1: one run per connection). The transcript renders
   the event stream; the composer + gate buttons produce the commands. */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};

// ---- persistent state: projects + their sessions (localStorage) -----------
const STORE_KEY = "achilles.projects";
let projects = load();
let selectedProjectId = null;

function load() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
  catch { return []; }
}
function save() { localStorage.setItem(STORE_KEY, JSON.stringify(projects)); }
function uid() { return Math.random().toString(36).slice(2, 10); }
function projectById(id) { return projects.find((p) => p.id === id); }

// ---- live session state ---------------------------------------------------
let ws = null;
let session = null;        // { id, projectId, mode, model, goal, result }
let pending = null;        // { kind: "goal"|"answer"|"approval", replyTo, subject }

// ---- model dropdown -------------------------------------------------------
async function loadModels() {
  const sel = $("#model");
  try {
    const r = await fetch("/api/models");
    const { models, default: def } = await r.json();
    sel.innerHTML = "";
    const ids = models && models.length ? models : (def ? [def] : []);
    for (const id of ids) {
      const o = el("option", null, id);
      o.value = id;
      if (id === def) o.selected = true;
      sel.appendChild(o);
    }
    if (!ids.length) sel.appendChild(el("option", null, "(kein Modellserver)"));
  } catch {
    sel.innerHTML = "";
    sel.appendChild(el("option", null, "(kein Modellserver)"));
  }
}

// ---- sidebar rendering ----------------------------------------------------
function renderProjects() {
  const box = $("#projects");
  box.innerHTML = "";
  if (!projects.length) {
    box.appendChild(el("div", "muted", "Noch keine Projekte."));
    return;
  }
  for (const p of projects) {
    const wrap = el("div", "project" + (p.id === selectedProjectId ? " selected" : ""));
    const head = el("div", "project-head");
    head.appendChild(el("span", "project-twisty", p.open ? "▾" : "▸"));
    head.appendChild(el("span", "project-name", p.name));
    head.title = p.path;
    head.onclick = () => { selectProject(p.id); p.open = !p.open; save(); renderProjects(); };
    wrap.appendChild(head);
    if (p.open) {
      const list = el("div", "sessions");
      if (!p.sessions.length) list.appendChild(el("div", "session-item", "keine Sessions"));
      for (const s of p.sessions.slice().reverse()) {
        const item = el("div", "session-item");
        const dot = el("span", "dot " + (s.result || "running"), "●");
        item.appendChild(dot);
        item.appendChild(document.createTextNode(`${labelMode(s.mode)} · ${s.goal || "…"}`));
        item.title = `${s.goal}\n${new Date(s.ts).toLocaleString()} — ${s.result || "running"}`;
        list.appendChild(item);
      }
      wrap.appendChild(list);
    }
    box.appendChild(wrap);
  }
}

function labelMode(m) { return m === "interview" ? "Planungsmodus" : "Autopilot"; }

function selectProject(id) {
  selectedProjectId = id;
  const p = projectById(id);
  $("#project-path").textContent = p ? p.path : "Kein Projekt gewählt";
  $("#new-session").disabled = !p;
  $("#new-session-caret").disabled = !p;
  renderProjects();
}

// ---- actions: new project / new session -----------------------------------
$("#new-project").onclick = () => {
  const path = prompt("Projektverzeichnis (absoluter Pfad):");
  if (!path) return;
  const name = path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || path;
  const p = { id: uid(), path: path.trim(), name, open: true, sessions: [] };
  projects.push(p);
  save();
  selectProject(p.id);
};

const menu = $("#mode-menu");
$("#new-session-caret").onclick = (e) => { e.stopPropagation(); menu.classList.toggle("hidden"); };
$("#new-session").onclick = () => startSession("interview");   // default click = Planungsmodus
document.addEventListener("click", () => menu.classList.add("hidden"));
menu.querySelectorAll(".menu-item").forEach((b) => {
  b.onclick = (e) => { e.stopPropagation(); menu.classList.add("hidden"); startSession(b.dataset.mode); };
});

function startSession(mode) {
  const project = projectById(selectedProjectId);
  if (!project) return;
  if (ws) { try { ws.close(); } catch {} }
  clearTranscript();
  session = { id: uid(), projectId: project.id, mode, model: $("#model").value, goal: null, result: null };
  connect();
  systemLine(`Neue Session — ${labelMode(mode)} · ${project.path}`);
  setPending({ kind: "goal" });
}

// ---- WebSocket ------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}`);
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); };
  ws.onerror = () => setConn(false);
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    handleEvent(msg.type, msg.data || {}, msg.id);
  };
}

function setConn(on) {
  const c = $("#conn");
  c.classList.toggle("on", on);
  c.title = on ? "verbunden" : "getrennt";
}

function sendCmd(type, data, extra) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(Object.assign({ v: 1, type, data: data || {} }, extra || {})));
}

// ---- event handling -------------------------------------------------------
function handleEvent(type, data, id) {
  switch (type) {
    case "run.started":
      setStatus("läuft"); break;
    case "interview.question":
      assistant(data.prompt || "?", data.default ? `Default: ${data.default}` : null);
      setPending({ kind: "answer", replyTo: id, canSkip: true });
      break;
    case "spec.ready":
      block("Spec", data.spec_md, true); break;
    case "plan.ready":
      listBlock("Plan", (data.steps || []).map((s) => s.text || s), true); break;
    case "dod.ready":
      listBlock("Definition of Done", (data.criteria || []).map((c) => `[${c.kind}] ${c.text}`)); break;
    case "approval.request":
      approvalBubble(data.subject || "plan", data.content || "", id); break;
    case "step.started":
      setStatus(`Step ${data.index}/${data.total}`);
      systemLine(`▶ Step ${data.index}/${data.total}: ${data.text || ""}`); break;
    case "step.finished":
      systemLine(`✔ Step ${data.index} — ${data.status}`); break;
    case "verify.result":
      verifyLine(data.passed, data.command, data.output); break;
    case "commit.made":
      logLine(`commit: ${data.message}`, "muted"); break;
    case "accept.round":
      systemLine(`Definition of Done — Runde ${data.round}/${data.max}`); break;
    case "accept.failures":
      listBlock("Noch offen", (data.failures || []).map((f) => `[${f.kind}] ${f.text} — ${f.reason}`)); break;
    case "log":
      logLine(data.text || "", data.level || "info"); break;
    case "run.finished":
      finishRun(data.result || "success", data.reason); break;
    case "error":
      logLine("✖ " + (data.message || "Fehler"), "error"); break;
    default: break;
  }
}

// ---- composer + gates -----------------------------------------------------
const input = $("#input");
const sendBtn = $("#send");
const skipBtn = $("#skip");

function setPending(p) {
  pending = p;
  const active = !!p;
  input.disabled = !active;
  sendBtn.disabled = !active || p.kind === "approval";
  skipBtn.classList.toggle("hidden", !(p && p.kind === "answer" && p.canSkip));
  if (active && p.kind !== "approval") { input.focus(); }
  input.placeholder = !active ? "…"
    : p.kind === "goal" ? "Was soll gebaut werden? (Enter senden)"
    : p.kind === "answer" ? "Antwort … (Enter senden, oder überspringen)"
    : "…";
}

function submitComposer() {
  if (!pending) return;
  const text = input.value.trim();
  if (pending.kind === "goal") {
    if (!text) return;
    session.goal = text;
    recordSession();
    user(text);
    sendCmd("run.start", {
      goal: text, mode: session.mode,
      cwd: projectById(session.projectId).path,
      config_overrides: { model: session.model },
    });
    input.value = ""; autosize();
    setPending(null); setStatus("läuft");
  } else if (pending.kind === "answer") {
    user(text || "(überspringen)");
    sendCmd("answer", text ? { value: text } : { skip: true }, { reply_to: pending.replyTo });
    input.value = ""; autosize();
    setPending(null);
  }
}

skipBtn.onclick = () => {
  if (!pending || pending.kind !== "answer") return;
  user("(überspringen)");
  sendCmd("answer", { skip: true }, { reply_to: pending.replyTo });
  setPending(null);
};

sendBtn.onclick = submitComposer;
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitComposer(); }
});
input.addEventListener("input", autosize);
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; }

// ---- transcript builders --------------------------------------------------
const T = () => $("#transcript");
function clearTranscript() { T().innerHTML = ""; }
function push(node) { const t = T(); t.appendChild(node); t.scrollTop = t.scrollHeight; return node; }

function user(text) { return push(el("div", "bubble user", text)); }
function assistant(text, sub) {
  const b = el("div", "bubble assistant");
  b.appendChild(el("div", null, text));
  if (sub) b.appendChild(el("div", "muted", sub));
  return push(b);
}
function systemLine(text) { return push(el("div", "bubble system", text)); }
function logLine(text, level) {
  const lv = level === "warn" ? "warn" : level === "error" ? "error" : "";
  return push(el("div", "bubble log " + lv, text));
}
function verifyLine(passed, cmd, output) {
  const b = el("div", "bubble log");
  const s = el("span", "verify " + (passed ? "pass" : "fail"), (passed ? "✔ verify grün" : "✖ verify rot"));
  b.appendChild(s);
  if (output) { const pre = el("pre", null, output); b.appendChild(pre); }
  return push(b);
}
function block(title, body, pre) {
  const b = el("div", "bubble assistant");
  b.appendChild(el("div", "title", title));
  b.appendChild(pre ? el("pre", null, body) : el("div", null, body));
  return push(b);
}
function listBlock(title, items, ordered) {
  const b = el("div", "bubble assistant");
  b.appendChild(el("div", "title", title));
  const list = el(ordered ? "ol" : "ul");
  for (const it of items) list.appendChild(el("li", null, it));
  b.appendChild(list);
  return push(b);
}

function approvalBubble(subject, content, replyTo) {
  const b = el("div", "bubble assistant");
  b.appendChild(el("div", "title", subject === "spec" ? "Spec — freigeben?" : "Plan — freigeben?"));
  if (content) b.appendChild(el("pre", null, content));
  const row = el("div", "gate-actions");
  const approve = el("button", "btn btn-primary", "Freigeben");
  const reject = el("button", "btn", "Ablehnen");
  const edit = el("button", "btn ghost", "Ändern …");
  approve.onclick = () => decide(b, row, replyTo, { decision: "approve" }, "Freigegeben");
  reject.onclick = () => decide(b, row, replyTo, { decision: "reject" }, "Abgelehnt");
  edit.onclick = () => {
    const instruction = prompt("Was soll geändert werden? (Klartext)");
    if (instruction == null) return;
    decide(b, row, replyTo, { decision: "edit", instruction }, "Änderung gesendet");
  };
  row.append(approve, reject, edit);
  b.appendChild(row);
  setPending({ kind: "approval", replyTo, subject });
  return push(b);
}

function decide(bubble, row, replyTo, data, note) {
  sendCmd("approval", data, { reply_to: replyTo });
  row.remove();
  bubble.appendChild(el("div", "muted", "→ " + note));
  setPending(null);
}

// ---- run lifecycle / recents ----------------------------------------------
function setStatus(text) { $("#run-status").textContent = text || ""; }

function recordSession() {
  const p = projectById(session.projectId);
  if (!p) return;
  session.ts = Date.now();
  p.sessions.push({ id: session.id, goal: session.goal, mode: session.mode,
                    model: session.model, ts: session.ts, result: null });
  save(); renderProjects();
}

function finishRun(result, reason) {
  setStatus(result === "success" ? "fertig ✔" : result === "halted" ? "angehalten" : "fehlgeschlagen");
  systemLine(`Run ${result}${reason ? " — " + reason : ""}`);
  if (session) {
    session.result = result;
    const p = projectById(session.projectId);
    const rec = p && p.sessions.find((s) => s.id === session.id);
    if (rec) { rec.result = result; save(); renderProjects(); }
  }
  setPending(null);
}

// ---- boot -----------------------------------------------------------------
loadModels();
renderProjects();
setConn(false);
