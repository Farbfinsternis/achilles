# Achilles

Ein **minimaler Harness für agentic coding mit kleinen, lokalen Modellen.**

Achilles ist das Gegenstück zu Odysseus: Odysseus ist ein Generalist (E-Mail,
Kalender, Recherche …) und steckt seine ganze Intelligenz in *Tool-Auswahl*.
Achilles macht nur **eines** — Code (und andere Artefakte) in kleinen, getesteten
Schritten schreiben — und stützt sich auf ein **Verifikations-Orakel**
(Tests/Compiler) plus eine **Definition of Done**, damit ein schwaches Modell
nicht beim ersten Versuch recht haben muss. Es muss nur **konvergieren**.

---

## Quickstart

Du willst Achilles einfach nur schnell ausprobieren? Hier ist der kürzeste Weg:

### 1. Vorbereitung
Für diesen Quickstart benötigen wir:
- **Python 3.11+**: Bitte stelle sicher, dass eine aktuelle Python-Version installiert ist.
- **[LM Studio](https://lmstudio.ai/)**: Lade es herunter, installiere es und starte den lokalen Server. Lade dir dort ein Modell deiner Wahl herunter.

### 2. Achilles herunterladen und installieren
Öffne ein Terminal und lade das Projekt herunter:
```bash
git clone https://github.com/Farbfinsternis/achilles.git
```

Wechsle nun in das heruntergeladene Verzeichnis. 
*(Windows-Tipp: Öffne den Ordner `achilles` im Windows Explorer, klicke oben in die Adresszeile, tippe `cmd` ein und drücke Enter. So öffnet sich das Terminal direkt im richtigen Pfad!)*

Mache Achilles nun global auf deinem System verfügbar, indem du folgenden Befehl ausführst:
```bash
pip install -e .
```
*(Das `-e` sorgt dafür, dass Achilles sich automatisch aktualisiert, falls du später den Code im Ordner veränderst.)*

### 3. Achilles starten und Modell verbinden
Starte Achilles, indem du einfach Folgendes in dein Terminal eingibst:
```bash
achilles
```

Wechsle kurz zu LM Studio. Gehe in die Liste deiner Modelle, klicke auf dein geladenes Modell und wähle **"Copy Model Identifier"** (oder kopiere den genauen Namen). 
Gehe zurück in dein Terminal zu Achilles und tippe:
```bash
:model <Dein_kopierter_Identifier>
```
*(z.B.: `:model lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF`)*

### 4. Loslegen!
Jetzt kannst du deinen ersten Prompt eingeben, z.B.:
> "Schreibe mir ein kleines Python-Skript, das das aktuelle Datum ausgibt."

**Wichtiger Hinweis:** Achilles arbeitet mit *lokalen* Modellen. Diese brauchen für ihre Denk- und Programmierschritte deutlich länger als riesige Frontier-Modelle (wie GPT-4 oder Claude 3.5 Sonnet) in der Cloud. Hab also ein wenig Geduld, während Achilles arbeitet!

### Bonus: Bilder generieren mit ComfyUI
Wenn du [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installiert hast, kannst du dem Modell erlauben, Bilder für dich zu generieren!
1. Erstelle in ComfyUI deinen gewünschten Workflow.
2. Speichere ihn als **API Workflow** (als `.json` Datei exportieren).
3. Ziehe diese JSON-Datei per **Drag & Drop** einfach direkt in das Terminal, in dem du gerade mit Achilles schreibst!

---

## Die Idee hinter Achilles

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

```text
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

## Abhängigkeiten für Entwickler

- **Python 3.11+** (für `tomllib` aus der Standardbibliothek).
- **Keine Runtime-Abhängigkeiten.** Achilles nutzt ausschließlich die stdlib.
- **Ein OpenAI-kompatibler LLM-Server**, den du selbst betreibst (z.B. LM Studio, llama.cpp, Ollama).
- Optional **`pytest`** — nur für die Testsuite und das mitgelieferte Beispiel.

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

Das `verify_command` (das Orakel) in der `achilles.toml` ist der **Boden**: es beweist, dass nichts *kaputt* ist. Darum erzeugt ein zweiter Planungs-Pass eine `.achilles/done.md` mit Akzeptanzkriterien:

```text
- [ ] exists:   <pfad>              (der Harness prüft os.path)
- [ ] contains: <pfad> :: <text>    (der Harness prüft Teilstring)
- [ ] judge:    <klartext>          (das Modell als strenger, kontext-isolierter Prüfer)
```

Der Harness prüft die Kriterien und schickt gezielte Fixes, bis sie erfüllt sind.

## Interaktive Sitzung (REPL)

Du kannst Achilles jederzeit ohne festes Ziel starten:
```bash
achilles -w pfad/zum/projekt
```

Jede Zeile ist ein neues Ziel. Was über mehrere Ziele erhalten bleibt, sind Config, Workspace und das Git-Repo. Zeilen, die mit `:` beginnen, steuern die Sitzung und gehen nie ans Modell:

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

## Konfiguration (`achilles.toml`)

Die Config wird geschichtet geladen (Paket → `~/.achilles.toml` → `<workspace>/achilles.toml`). 
Die wichtigsten Felder:

| Feld | Bedeutung |
|---|---|
| `base_url`, `api_key`, `model` | der OpenAI-kompatible Endpunkt und das Modell |
| `verify_command` | das Orakel, nach jedem Schritt ausgeführt (leer = blind) |
| `act_protocol` | `"native"` \| `"json"` \| `"text"` |
| `use_acceptance` | Definition of Done an/aus |
| `use_git` | grüne Schritte committen |

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## Was bewusst (noch) fehlt

- **Repo-Map / Code-Retrieval** (tree-sitter + Abhängigkeitsgraph)
- **Semantische Output-Kürzung** (Stacktrace behalten, grünes Rauschen wegwerfen).
- **Echter Kontext-/Token-Budgetierer** (`n_ctx` messen, trimmen).
