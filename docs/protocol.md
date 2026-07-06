# Achilles Event/Command-Protokoll

Status: **Entwurf v1** · betrifft die Entkopplung von Engine und Oberfläche.

Dieses Dokument spezifiziert die Nachrichten-Grenze zwischen dem headless
**Achilles-Core** (Python: `harness`, `planner`, `acceptance`, `llm`) und einer
**Oberfläche** (Terminal-Client, Web-UI, später Electron/Tauri). Ziel ist, dass
die Engine keinen UI-Code enthält und jede Oberfläche ein austauschbarer Client
auf demselben Protokoll ist.

Das Protokoll legt **Nachrichten-Shapes** fest, nicht den Transport. Dieselben
Shapes gelten für JSON-Lines über stdin/stdout (Subprozess) und für WebSocket
(Web-UI).

---

## 1. Designprinzipien

1. **Zwei Richtungen.** *Events* (Engine → UI, fire-and-forget) und *Commands*
   (UI → Engine). Ein blockierendes Gate ist ein *Request*: ein Event, das auf
   ein korreliertes Command wartet.
2. **Die Engine bleibt synchron.** Ein `request()`-Helper blockiert intern auf
   einer Inbox-Queue; `run()` bleibt linearer Code. Kein async-Umbau des
   Harness.
3. **Transport-agnostisch.** Die Nachrichten-Shapes sind für alle Transporte
   identisch. Die Transportwahl ist eine spätere, separate Entscheidung.
4. **`log` ist der Auffangkanal, nicht die Struktur.** Jeder semantische
   Zustandswechsel bekommt einen eigenen Event-Typ. `log` trägt nur noch Prosa
   fürs Terminal-Gefühl. **Regel:** kein neuer semantischer Zustand geht über
   `log` — sonst erodiert die Struktur wieder zu Prosa-Parsing.
5. **Chatbot-Modell.** Aus UI-Sicht ist der gesamte Verlauf ein Chat.
   `interview.question` und `approval.request` sind beide „Assistant fragt,
   wartet auf User-Turn" und werden identisch als Chat-Bubbles gerendert.

---

## 2. Envelope

Jede Nachricht — Event wie Command — trägt denselben Rahmen:

```json
{
  "v": 1,
  "run": "run-a1b2",
  "seq": 42,
  "ts": 1720000000123,
  "type": "approval.request",
  "id": "req-7",
  "data": { }
}
```

| Feld   | Pflicht | Bedeutung |
|--------|---------|-----------|
| `v`    | ja      | Protokoll-Version (aktuell `1`). |
| `run`  | ja      | Run-Identität. v1: ein Run pro Engine-Prozess. |
| `seq`  | Events  | Monoton steigend pro Run. Ordering, Replay, Lücken-Erkennung. |
| `ts`   | ja      | Zeitstempel (epoch ms). |
| `type` | ja      | Nachrichtentyp (siehe Kataloge). |
| `id`   | Requests| Nur auf Events, die eine Antwort erwarten. |
| `data` | ja      | Typ-spezifische Nutzlast (kann `{}` sein). |

**Korrelation.** Ein Request-Event trägt `id`. Das antwortende Command trägt
`reply_to` mit demselben Wert. Das ersetzt die heute synchronen
`input()`-Aufrufe durch ein korreliertes Paar.

---

## 3. Events (Engine → UI)

Requests (erwarten eine Antwort) sind mit **(req)** markiert.

| Typ | `data` | Bedeutung |
|-----|--------|-----------|
| `run.started` | `{goal, mode, resumed}` | Beginn. `mode: "autopilot" \| "interview"`. |
| `interview.question` **(req)** | `{field, prompt, default, kind, options?}` | Ein Slot des festen Fragenkatalogs. Erwartet `answer`. |
| `spec.ready` | `{spec_md, sections}` | Normalisiertes Spec geschrieben (`.achilles/spec.md`). |
| `plan.ready` | `{steps: [{text}]}` | Plan erstellt/aktualisiert. |
| `dod.ready` | `{criteria: [{kind, text}]}` | Definition of Done erstellt. |
| `approval.request` **(req)** | `{subject, content}` | `subject: "spec" \| "plan"`. Erwartet `approval`. |
| `step.started` | `{index, total, text}` | Ein Plan-Schritt beginnt. |
| `step.finished` | `{index, status}` | `status: "done" \| "unfinished"`. |
| `verify.result` | `{command, passed, output}` | Ergebnis des Verifikations-Orakels. |
| `commit.made` | `{message, sha?}` | Ein Git-Commit wurde erzeugt. |
| `accept.round` | `{round, max}` | Eine Akzeptanz-Runde beginnt. |
| `accept.failures` | `{failures: [{kind, text, reason}]}` | Noch unerfüllte Kriterien. |
| `log` | `{level, text}` | Freie Narration. `level: "info" \| "warn" \| "error" \| "muted"`. |
| `run.finished` | `{result, reason?}` | `result: "success" \| "halted" \| "failed"`. |
| `error` | `{fatal, message}` | Infrastrukturfehler (LLM/Judge nicht erreichbar). |

Die `prompt`-Zeile von `interview.question` **ist** die Assistant-Bubble; es
braucht keinen separaten Nachrichten-Typ für den Interview-Text.

---

## 4. Commands (UI → Engine)

| Typ | `data` | Antwortet auf |
|-----|--------|---------------|
| `run.start` | `{goal, mode, cwd, config_overrides?}` | — (startet einen Run) |
| `answer` | `{reply_to, value}` **oder** `{reply_to, skip: true}` | `interview.question` |
| `approval` | `{reply_to, decision, instruction?}` | `approval.request` |
| `resume` | `{run}` | — (setzt gespeicherten Spec/Plan fort) |
| `cancel` | `{run}` | — (bricht den laufenden Run ab) |

`approval.decision: "approve" \| "reject" \| "edit"`. Bei `"edit"` trägt
`instruction` die Klartext-Änderung. Das faltet die heute getrennten
`input()`-Aufrufe von `_approve()` und `_ask_edit_instruction()` in **ein**
Command.

---

## 5. Das Request/Response-Muster

Der einzige strukturelle Umbau am Harness. Heute blockiert ein `input()` den
Thread:

```python
ans = input("Proceed with this plan? [Y/n/edit] ")
```

Nachher blockiert ein `request()` auf der Inbox-Queue, gibt aber die Kontrolle
im selben linearen Fluss zurück:

```python
reply = self.request("approval.request",
                     {"subject": "plan", "content": plan_md})
# reply == {"decision": "approve"}
#       |  {"decision": "edit", "instruction": "..."}
#       |  {"decision": "reject"}
```

Die Engine-Grenze ist damit ein **`Channel`-Objekt** mit zwei Methoden, injiziert
statt des heutigen `log`-Callbacks in `Harness.__init__`:

- `emit(type, data)` — Event ohne Antwort (fire-and-forget).
- `request(type, data) -> data` — Event mit `id`, blockiert bis ein Command mit
  passendem `reply_to` eintrifft, gibt dessen `data` zurück.

`self.log(x)` wird zu `self.emit("log", {"text": x, "level": ...})` — ein
mechanisches Ersetzen, das die vorhandenen Log-Stellen im Kontrollfluss
unverändert lässt.

**Client-Implementierungen derselben Grenze:**

- **Terminal-Client:** `emit` druckt, `request` beantwortet via `input()`.
- **Web-Client:** `Channel` über WebSocket; `request` sendet das Event und wartet
  auf die eingehende Command-Nachricht.

Gleiche Engine, kein `if terminal:` im Harness.

---

## 6. Interview: Stream, festes Rückgrat, minimaler Router

Das Interview läuft als **Stream** (eine `interview.question` pro Slot), nicht
als Sammelformular — die UI präsentiert Achilles wie einen gewöhnlichen Chatbot.

Der Fragenkatalog ist **fest und deterministisch** im Harness verankert (kein
schlaues Modell nötig, um Fragen zu erzeugen). Startkatalog, je ein Slot:

| `field` | Frage | Leerantwort-Default |
|---------|-------|---------------------|
| `purpose` | Zweck des Projekts? | aus `goal` ableiten (1 Modell-Call) |
| `audience` | Zielgruppe? | „general audience" |
| `features` | Kernfunktionen? | aus `goal` ableiten |
| `scope` | Prototyp-Scope? | „minimal working prototype" |
| `ui_ux` | UI/UX (Aussehen, Bedienung)? | Design-Default / weglassen |

Da ein Chatbot einlädt, frei zu tippen, kollidiert reines Slot-Filling mit dem
Chat-Gefühl. Auflösung: die freie Antwort läuft durch einen **rein heuristischen
Router** mit genau zwei Absichten — **kein Modell-Call im Interview-Loop**:

```
skip   → leer / „egal" / „weiß nicht" → Default, nächste Frage
answer → alles andere → Literal-Wert als Slot-Antwort, nächste Frage
```

Es gibt **kein** `back`-Intent und keine Klassifikation von Meta-Fragen. Der
Grund: die Korrektur läuft nicht über den Stream, sondern über das
**Spec-Approval-Gate** (`approval.request {subject:"spec"}`) — der User sieht das
fertige Spec und kann `decision:"edit"`. Ein perfekter Inline-Router ist damit
unnötig, und der Determinismus bleibt maximal (der einzige Modell-Kontakt im
Planungsmodus ist die `normalize()`-Runde, die das Spec zusammensetzt). Auf
langsamer, lokaler Hardware spart das pro Interview mehrere Modell-Aufrufe.

Non-interaktiv (kein TTY / Headless) nehmen alle Slots ihren Default → leeres
Interview → Autopilot-äquivalent; der CI-/Test-Pfad bleibt grün.

Die Detail-Entscheidungen zu Fragenkatalog, `normalize()` und dem
Verbatim→DoD-Mapping stehen in [spec-artifact.md](spec-artifact.md).

---

## 7. Zwei Einstiege: „Neues Projekt" ▾

Die UI startet Runs über einen Dropdown. Er wählt nichts weiter als den
`mode`-Wert im `run.start`-Command:

```
"Neues Projekt" ▾
 ├─ Autopilot      → run.start {mode: "autopilot", goal}
 └─ Planungsmodus  → run.start {mode: "interview",  goal}
```

Beide Modi teilen sich den ersten Turn (das Prompt-Eingabefeld) und divergieren
erst danach im `run()`:

- **Autopilot** (`mode: "autopilot"`): `goal` → `make_plan` → Plan-Approval →
  Execute. Heutiges Verhalten, unverändert. Entspricht einem leeren Interview.
- **Planungsmodus** (`mode: "interview"`): `goal` ist der Eröffnungs-Turn →
  Interview-Stream → `spec.ready` → Spec-Approval → `make_plan` →
  Plan-Approval → Execute.

**Namenshinweis:** Das UI-Label heißt „Planungsmodus", der Command-Wert aber
`"interview"` (bzw. `"spec"`), **nicht** `"planning"` — beide Modi erzeugen einen
Plan (`make_plan`), das Unterscheidende dieses Modus ist das vorgelagerte
Interview/Spec, nicht „Planung" an sich. Label für den Menschen, präziser Wert
für den Code.

**„Neues Projekt" ≠ Resume.** Der Dropdown startet immer frisch. Das Fortsetzen
eines Runs mit gespeichertem Spec/Plan (`resume`-Command) ist ein separater
Einstieg pro Projekt, nicht Teil dieses Dropdowns.

---

## 8. Sequenzen

### 8.1 Autopilot

```
UI → run.start {mode:"autopilot", goal:"<prompt>"}
E  → run.started {mode:"autopilot", resumed:false}
E  → plan.ready {steps:[…]}
E  → dod.ready  {criteria:[…]}
E  → approval.request {id:a1, subject:"plan", content:…}
UI → approval {reply_to:a1, decision:"approve"}
E  → step.started {index:1, total:5, text:"…"}
E  → log {text:"write_file index.html"}
E  → verify.result {passed:true}
E  → commit.made {message:"achilles: step 1 — …"}
E  → step.finished {index:1, status:"done"}
   … Steps 2–5 …
E  → accept.round {round:1, max:3}
E  → run.finished {result:"success"}
```

### 8.2 Planungsmodus (mit Interview + zwei Gates)

```
UI → run.start {mode:"interview", goal:"<roher 300-Zeilen-Prompt>"}
E  → run.started {mode:"interview", resumed:false}
E  → interview.question {id:q1, field:"purpose",  prompt:"Zweck?", default:"…"}
UI → answer {reply_to:q1, value:"Bäckerei-Landingpage"}
E  → interview.question {id:q2, field:"audience", prompt:"Zielgruppe?", default:"…"}
UI → answer {reply_to:q2, skip:true}                    # Default greift
   … (restliche Slots) …
E  → spec.ready {spec_md, sections}
E  → approval.request {id:a1, subject:"spec", content:<spec_md>}
UI → approval {reply_to:a1, decision:"edit", instruction:"Öffnungszeiten raus"}
E  → spec.ready {…}                                      # neu normalisiert
E  → approval.request {id:a2, subject:"spec", content:…}
UI → approval {reply_to:a2, decision:"approve"}
E  → plan.ready {steps:[…]}
E  → dod.ready  {criteria:[…]}
E  → approval.request {id:a3, subject:"plan", content:…}
UI → approval {reply_to:a3, decision:"approve"}
   … Execute wie in 8.1 …
E  → run.finished {result:"success"}
```

Autopilot ist derselbe Stream ohne den `interview.*`- und den
`spec`-`approval`-Block — das heutige Verhalten ist ein Spezialfall.

---

## 9. Bewusst NICHT in v1

- **Transport** (stdin/stdout vs WebSocket) — separate Entscheidung, Shapes sind
  identisch.
- **Step-Diffs.** Falls die Web-UI Diffs pro Schritt zeigen soll, kommt später
  ein `step.diff {unified}`-Event dazu. Kein Bruch am bestehenden Protokoll.
- **Interjektion während der Ausführung.** v1 ist **strikt**: Eingabe nur, wenn
  ein Request offen ist; während `_work` läuft, ist das Eingabefeld beschäftigt.
  v2 kann Nachrichten in eine Queue legen und am nächsten Step-Grenzpunkt
  einspielen (deckt sich mit dem `cancel`-Checkpoint). Nachrüstbar ohne
  Protokollbruch.
- **`cancel` mitten in `_work`.** Sofortiges Abbrechen braucht Checkpoints
  zwischen Tool-Calls im act-loop. v1: Abbruch wirkt am nächsten
  Step-Grenzpunkt.
- **Mehrere parallele Runs / Auth.** Das `run`-Feld ist vorgesehen, aber v1 =
  ein Run pro Engine-Prozess.

---

## 10. Bezug zum heutigen Code

| Protokoll-Element | Heutige Stelle |
|-------------------|----------------|
| `Channel.emit`/`request` | ersetzt den `log`-Callback in `Harness.__init__` |
| `approval.request` + `approval` | `_approve()` / `_approve_loop()` (`harness.py`) |
| `edit`-Instruktion | `_ask_edit_instruction()` → in `approval` gefaltet |
| `plan.ready` | `_print_plan` nach `make_plan` |
| `dod.ready` | `_make_and_save_dod` |
| `step.started/finished` | `_work_through_plan` |
| `verify.result` | `_verify` |
| `commit.made` | `_commit` |
| `accept.round`/`accept.failures` | `_acceptance_phase` |
| Resume-Logik | goal-verkettetes `plan.md` (und künftig `spec.md`) |

Der Interview-Teil (`interview.*`, `spec.ready`, `spec`-Approval) ist neu und
setzt vor `make_plan` an; siehe auch die Spec-Artefakt-Skizze (`spec.md` →
Planner + Definition of Done).
