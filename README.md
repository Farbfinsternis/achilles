# Achilles

Ein **minimaler Harness für agentic coding mit kleinen, lokalen Modellen.**

Achilles ist das Gegenstück zu Odysseus: Odysseus ist ein Generalist (E-Mail,
Kalender, Recherche …) und steckt seine ganze Intelligenz in *Tool-Auswahl*.
Achilles macht nur **eines** — Code (und andere Artefakte) in kleinen, getesteten
Schritten schreiben — und stützt sich auf ein **Verifikations-Orakel**
(Tests/Compiler) plus eine **Definition of Done**, damit ein schwaches Modell
nicht beim ersten Versuch recht haben muss. Es muss nur **konvergieren**.

## Die Idee in einem Satz

> Das Modell ist klug, aber vergesslich. Der Harness ist dumm, aber zuverlässig.
> Also verlagern wir jede Last, mit der das Modell schlecht umgeht, **aus dem
> Modell heraus.**

| Schwäche kleiner Modelle | Achilles' Gegenmittel |
|---|---|
| vergisst die Aufgabe | Plan lebt in `.achilles/plan.md`, wird pro Schritt neu gelesen |
| prüft seine Arbeit nicht | **der Harness** führt nach jedem Schritt die Tests aus |
| füllt den Kontext bei großen Aufgaben | jeder Schritt läuft in **frischem** Kontext (`act → verify → commit → reset`) |
| baut Fehler aufeinander | jeder grüne Schritt wird per Git committet (billiger Rollback-Punkt) |
| vergeigt das Ausgabeformat | **Constrained Decoding** erzwingt das Format an der Wurzel (`act_protocol = "json"`) |
| erfindet generischen Inhalt | Ziel-Fakten und geforderte Dateipfade werden in **jeden** Schritt injiziert |

Nichts davon braucht ein kluges Modell. Es braucht eine kluge **Schleife**.

## Aufbau (jede Datei = ein Konzept)

```
achilles/
  protocol.py      Der Format-Vertrag: parst ```act-Blöcke aus dem Modell-Text.
  tools.py         Die Hände: read_file, write_file, list_dir, run_command (erweiterbare Registry).
  llm.py           Der Motor-Anschluss: OpenAI-kompatibel, abhängigkeitsfrei.
  planner.py       Ziel → Checkliste (die "dumme Entscheidung").
  acceptance.py    Die Definition of Done: exists/contains/judge — die "Decke" über dem Orakel.
  harness.py       Der Orchestrator: Zustandsmaschine + act/verify/commit-Schleife.
  repl.py          Die interaktive Sitzung: reicht Ziele an den Harness, steuert mit :befehlen.
  config.py        Alles Einstellbare an einem Ort.
  lmstudio.py      Modell in den VRAM laden/entladen/wechseln (LM-Studio-CLI).
  comfy.py         Optionales generate_image-Tool (Bildgenerierung über ComfyUI).
  comfy_client.py  Der ComfyUI-HTTP-Client.
  workflows.py     Registrierung/Anwendung markierter ComfyUI-Workflows.
```

## Abhängigkeiten

- **Python 3.11+** (für `tomllib` aus der Standardbibliothek).
- **Keine Runtime-Abhängigkeiten.** Achilles nutzt ausschließlich die stdlib
  (`urllib` für HTTP, `tomllib` für Config). Das ist Absicht — ein minimaler
  Harness soll nichts zu installieren haben.
- **Ein OpenAI-kompatibler LLM-Server**, den du selbst betreibst:
  [LM Studio](https://lmstudio.ai), [llama.cpp](https://github.com/ggml-org/llama.cpp),
  [Ollama](https://ollama.com) oder [vLLM](https://github.com/vllm-project/vllm).
- Optional **`pytest`** — nur für die Testsuite und das mitgelieferte Beispiel.
- Optional für Bildgenerierung: **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**
  und die **LM-Studio-CLI (`lms`)** (zum VRAM-Swap auf kleinen Karten).

## Installation / Build

Es gibt nichts zu bauen. Zum direkten Ausführen genügt Python 3.11+:

```
python -m achilles "…dein ziel…" -w pfad/zum/projekt
```

Um `achilles` als echten Befehl von überall aufrufbar zu machen, einmalig im
Repo-Root:

```
pip install -e .
```

`-e` (editable) heißt: der Befehl folgt deinen Quell-Änderungen, ohne dass du neu
installieren musst. Danach liegt `achilles` im PATH und der Workspace-Default ist
das **aktuelle Verzeichnis** — genau der Ablauf „in ein Projekt wechseln,
`achilles` tippen".

## Schnellstart

**1. Ein lokales Modell servieren**, das einen OpenAI-kompatiblen Endpunkt
anbietet. Beispiel llama.cpp:

```
./llama-server -m ornith-9b.gguf -c 65536 --port 8080
```

**2. `achilles.toml` anpassen** — vor allem `model`, `base_url` und das
**`verify_command`** (das Orakel). Ohne Orakel rät Achilles nur.

> **Kontextfenster statt Token-Cap:** `max_tokens = 0` (Default) sendet *kein*
> eigenes Limit — die Engine füllt das Kontextfenster des geladenen Modells. Lade
> das Modell darum mit ausreichend Kontext (llama.cpp `-c 65536`, in LM Studio die
> Context-Length hochsetzen), sonst brechen ganze Datei-Schreibvorgänge mittendrin
> ab. Einen festen Deckel nur setzen, wenn du Output bewusst drosseln willst.

**3. Das mitgelieferte Beispiel ausprobieren** (zeigt die Schleife end-to-end):

```
pip install pytest
cd examples/kata
python ../../__main__.py "make the failing tests pass" --verify "python -m pytest -q" -y
```

Du solltest sehen, wie Achilles plant, `mathx.py` schreibt, die Tests laufen lässt
und den Balken grün macht — jeder Schritt committet.

**4. An einem echten Projekt:**

```
python -m achilles "füge einen zweiten Gegnertyp hinzu, der springt" -w pfad/zum/projekt
```

Du **erzählst**, was du willst. Die Zerlegung in kleine Schritte macht der
Planungs-Pass — genau das, was ein Frontier-Modell sonst still im Kopf erledigt.

### CLI-Optionen

```
python -m achilles [ziel …] [optionen]

  -w, --workspace <pfad>   Projektverzeichnis (Default: aktuelles).
  -y, --yes                Den erzeugten Plan automatisch bestätigen.
      --verify <cmd>       Orakel-Befehl setzen/überschreiben.
      --no-acceptance      Definition-of-Done-Phase überspringen (nur das Orakel).
      --no-git             Git-Checkpoints deaktivieren.
```

Ohne Ziel startet Achilles die interaktive Sitzung (REPL, siehe unten).

## Das Act-Protokoll & Constrained Decoding

Wie das Modell seine Werkzeuge bedient, steuert `act_protocol` in `achilles.toml`:

- **`"native"`** (Default) — OpenAI-Tool-Calling: die Tools gehen als JSON-Schema
  raus, strukturierte `tool_calls` kommen zurück. Ideal für tool-getunte Modelle.
- **`"json"`** — **Constrained Decoding**: das Modell gibt pro Zug **ein** JSON-
  Objekt aus, dessen Form der Server per `response_format` grammatikalisch
  **erzwingt**. Der zuverlässige Weg für ein schwaches 4B, dessen Format sonst
  bricht. Derselbe Schalter erzwingt auch Planner, Definition of Done und Judge.
- **`"text"`** — das textbasierte ```` ```act ````-Protokoll; der universelle
  Fallback.

Egal welche Wahl: Achilles fällt zur Laufzeit auf `"text"` zurück, wenn der Server
die reichere Anfrage ablehnt — es bricht nie hart ab.

## Definition of Done (die „Decke")

Das `verify_command` ist der **Boden**: es beweist, dass nichts *kaputt* ist. Es
sagt nichts darüber, ob das **Ziel** erreicht wurde — für generative Aufgaben
(„baue eine Landingpage für …") ist der Boden trivial erfüllbar.

Darum erzeugt ein zweiter Planungs-Pass eine `.achilles/done.md` mit
Akzeptanzkriterien, je nach Prüfart getaggt:

```
- [ ] exists:   <pfad>              (der Harness prüft os.path)
- [ ] contains: <pfad> :: <text>    (der Harness prüft Teilstring)
- [ ] judge:    <klartext>          (das Modell als strenger, kontext-isolierter Prüfer)
```

Der Harness prüft die Kriterien und schickt gezielte Fixes, bis sie erfüllt sind
(oder es bei ausbleibendem Fortschritt sauber abbricht und das klemmende Kriterium
benennt). Die `done.md` ist editierbar — du kannst Kriterien entschärfen und
erneut starten. Mit `--no-acceptance` bzw. `use_acceptance = false` bleibt es beim
Orakel-Boden.

## Bildgenerierung mit ComfyUI (optional)

Setzt du `comfy_url` in `achilles.toml`, bekommt das Modell eine `generate_image`-
Hand. Der Ablauf ist atomar: Achilles entlädt das LM-Studio-Modell, rendert über
ComfyUI, holt den VRAM zurück und lädt das Modell wieder (auch wenn ein Render
scheitert). Der knappe Bild-Brief des Modells wird zuvor von einer **Prompt-
Engineer-Persona** in einen dichten Diffusion-Prompt verfeinert.

Workflows verwaltest du in der REPL (ein Node im API-Export wird als
`achilles:prompt` bzw. `achilles:aspect` markiert):

```
achilles> :workflow register <pfad-zum-api-export.json>   # markierten Workflow prüfen + speichern
achilles> :workflow try <pfad>                            # Modell die Nodes selbst finden lassen
achilles> :workflow default <name>                        # Standard-Workflow setzen
achilles> :workflow list                                  # registrierte Workflows (★ = Default)
```

## Interaktive Sitzung (REPL)

Statt eines einzelnen Ziels kannst du eine laufende Sitzung starten — lass das
Ziel einfach weg:

```
python -m achilles -w pfad/zum/projekt
```

Jede Klartext-Zeile ist ein Ziel und läuft durch dieselbe `act → verify →
commit`-Schleife. Was über mehrere Ziele **erhalten bleibt**, sind Config,
Workspace und das Git-Repo — du kannst also mehrere Aufgaben hintereinander
aufgeben, das Modell wechseln oder den Workspace umstellen, ohne den Prozess neu
zu starten. Ein mehrzeiliges Ziel kannst du direkt **einfügen** (es wird als ein
Ziel erkannt) oder eine Zeile mit `\` fortsetzen. Zeilen, die mit `:` beginnen,
steuern die Sitzung und gehen **nie** ans Modell:

```
achilles> :verify python -m pytest -q     # das Orakel zur Laufzeit setzen
achilles> make the failing tests pass     # ein Ziel — Plan, Schleife, grün, commit
achilles> :status                         # wie viele Plan-Schritte sind erledigt?
achilles> :model ornith-1.0-9b            # Modell für die Sitzung wechseln (lädt/schaltet um)
achilles> :quit
```

| Befehl | Wirkung |
|---|---|
| `:help` | Befehlsübersicht |
| `:config` | aktuelle Sitzungs-Config zeigen |
| `:model [<id>]` | Modell der Sitzung zeigen / wechseln |
| `:verify [<cmd>]` | Orakel-Befehl zeigen / setzen |
| `:workspace [<pfad>]` | Arbeitsverzeichnis zeigen / wechseln |
| `:tools` | die Werkzeugliste zeigen, die das Modell erhält |
| `:workflow …` | ComfyUI-Bild-Workflows verwalten (`:workflow help`) |
| `:plan` | aktuellen `.achilles/plan.md` ausgeben |
| `:status` | erledigte vs. offene Plan-Schritte |
| `:quit` (oder Strg-D) | Sitzung beenden |

Der Plan ist **ans Ziel gebunden**: Gibst du dasselbe Ziel erneut ein (oder
startest neu), wird ein unterbrochener Lauf *fortgesetzt*; bei einem neuen Ziel
wird der alte Plan nach `.achilles/plan.<n>.md` archiviert und frisch geplant. Das
gilt auch für die one-shot-CLI.

## Konfiguration (`achilles.toml`)

Config wird geschichtet, von global nach spezifisch (später gewinnt): die
mitgelieferte `achilles.toml` neben dem Paket → `~/.achilles.toml` →
`<workspace>/achilles.toml`. Jedes Feld ist zusätzlich per Umgebungsvariable
überschreibbar (`ACHILLES_MODEL=…`, `ACHILLES_VERIFY_COMMAND=…`, …).

Die wichtigsten Felder:

| Feld | Bedeutung |
|---|---|
| `base_url`, `api_key`, `model` | der OpenAI-kompatible Endpunkt und das Modell |
| `verify_command` | das Orakel, nach jedem Schritt ausgeführt (leer = blind) |
| `act_protocol` | `"native"` \| `"json"` \| `"text"` (siehe oben) |
| `use_acceptance`, `max_accept_rounds` | Definition of Done an/aus, Fix-Runden |
| `judge_model` | optional ein stärkeres Modell nur für den Judge |
| `use_git` | grüne Schritte committen (billiger Rollback) |
| `max_tokens` | `0` = Kontextfenster des Modells füllen (empfohlen) |
| `temperature` | Sampling-Temperatur |
| `comfy_url`, `lms_command`, `workflows_dir` | Bildgenerierung (optional) |

## Tests

```
pip install pytest
python -m pytest tests/ -q
```

## Was bewusst (noch) fehlt

Bewusst weggelassen, weil später dran:

- **Repo-Map / Code-Retrieval** (tree-sitter + Abhängigkeitsgraph) statt das
  Modell selbst suchen zu lassen.
- **Semantische Output-Kürzung** (Stacktrace behalten, grünes Rauschen wegwerfen).
- **Echter Kontext-/Token-Budgetierer** (`n_ctx` messen, trimmen).

Die Architektur lässt für jedes davon bewusst Platz.
