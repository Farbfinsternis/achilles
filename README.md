# Achilles

Ein **minimaler Harness für agentic coding mit kleinen, lokalen Modellen.**

Achilles ist das Gegenstück zu Odysseus: Odysseus ist ein Generalist (E-Mail,
Kalender, Recherche …) und steckt seine ganze Intelligenz in *Tool-Auswahl*.
Achilles macht nur **eines** — Code in kleinen, getesteten Schritten schreiben —
und stützt sich auf ein **Verifikations-Orakel** (Tests/Compiler), damit ein
schwaches Modell nicht beim ersten Versuch recht haben muss. Es muss nur
**konvergieren**.

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

Nichts davon braucht ein kluges Modell. Es braucht eine kluge **Schleife**.

## Aufbau (jede Datei = ein Konzept)

```
achilles/
  protocol.py   Der Format-Vertrag: parst ```act-Blöcke aus dem Modell-Text.
  tools.py      Die Hände: read_file, write_file, list_dir, run_command.
  llm.py        Der Motor-Anschluss: OpenAI-kompatibel, abhängigkeitsfrei.
  planner.py    Prompt → Checkliste (die "dumme Entscheidung").
  harness.py    Der Orchestrator: Zustandsmaschine + act/verify/commit-Schleife.
  repl.py       Die interaktive Sitzung: reicht Ziele an den Harness, steuert mit :befehlen.
  config.py     Alles Einstellbare an einem Ort.
```

## Loslegen

**1. Ein lokales Modell servieren** (llama.cpp, Ollama, vLLM, LM Studio …), das
einen OpenAI-kompatiblen Endpunkt anbietet. Beispiel llama.cpp:

```
./llama-server -m ornith-7b.gguf -c 65536 --port 8080
```

**2. `achilles.toml` anpassen** — vor allem `model`, `base_url` und das
**`verify_command`** (das Orakel). Ohne Orakel rät Achilles nur.

> **Kontextfenster statt Token-Cap:** `max_tokens = 0` (Default) sendet *kein*
> eigenes Limit — die Engine füllt das Kontextfenster des geladenen Modells.
> Lade das Modell darum mit ausreichend Kontext (llama.cpp `-c 65536`, in LM
> Studio die Context-Length hochsetzen), sonst brechen ganze Datei-Schreibvorgänge
> mittendrin ab. Einen festen Deckel nur setzen, wenn du Output bewusst drosseln
> willst.

**3. Das mitgelieferte Beispiel ausprobieren** (zeigt die Schleife end-to-end):

```
pip install pytest
cd examples/kata
python ../../__main__.py "make the failing tests pass" --verify "python -m pytest -q" -y
```

Du solltest sehen, wie Achilles plant, `mathx.py` schreibt, die Tests laufen
lässt und den Balken grün macht — jeder Schritt committet.

**4. An einem echten Projekt:**

```
python -m achilles "füge einen zweiten Gegnertyp hinzu, der springt" -w pfad/zum/projekt
```

Du **erzählst**, was du willst. Die Zerlegung in kleine Schritte macht der
Planungs-Pass — genau das, was ein Frontier-Modell sonst still im Kopf erledigt.

## Von überall aufrufbar machen

Statt `python -m achilles` kannst du Achilles als echten Befehl installieren.
Einmalig im Repo-Root:

```
pip install -e .
```

Danach liegt `achilles` im PATH und der Workspace-Default ist das **aktuelle
Verzeichnis** — also genau der Ablauf „in ein Projekt wechseln, `achilles`
tippen":

```
cd G:\dev\projects\mein-spiel
achilles "füge einen zweiten Gegnertyp hinzu, der springt"
achilles                       # ohne Ziel: interaktive Sitzung (siehe unten)
```

`-e` (editable) heißt: der Befehl folgt deinen Quell-Änderungen, ohne dass du
neu installieren musst. Die mitgelieferte `achilles.toml` bleibt der globale
Default (Modell, `base_url`), egal aus welchem Ordner du startest; eine
`achilles.toml` *im* Projekt überschreibt sie pro Projekt.

> Hinweis: Der `achilles`-Befehl hängt an der Python-Umgebung, in der du
> `pip install -e .` ausgeführt hast. Tipp ihn in *derselben* (bzw. einer, deren
> `Scripts`-Verzeichnis im PATH liegt). Achilles braucht Python **3.11+**
> (`tomllib`).

## Interaktive Sitzung (REPL)

Statt eines einzelnen Ziels kannst du auch eine laufende Sitzung starten — lass
das Ziel einfach weg:

```
python -m achilles -w pfad/zum/projekt
```

Jede Klartext-Zeile ist ein Ziel und läuft durch dieselbe `act → verify →
commit`-Schleife wie oben. Was über mehrere Ziele hinweg **erhalten bleibt**,
sind Config, Workspace und das Git-Repo — du kannst also mehrere Aufgaben
hintereinander aufgeben, das Modell wechseln oder den Workspace umstellen, ohne
den Prozess neu zu starten. Zeilen, die mit `:` beginnen, steuern die Sitzung
und gehen **nie** ans Modell:

```
achilles> :verify python -m pytest -q     # das Orakel zur Laufzeit setzen
achilles> make the failing tests pass     # ein Ziel — Plan, Schleife, grün, commit
achilles> :status                         # wie viele Plan-Schritte sind erledigt?
achilles> add a square(n) function …      # nächstes Ziel: alter Plan wird archiviert
achilles> :model qwen2.5-coder-7b-instruct  # Modell für die Sitzung wechseln
achilles> :quit
```

| Befehl | Wirkung |
|---|---|
| `:help` | Befehlsübersicht |
| `:config` | aktuelle Sitzungs-Config zeigen |
| `:model [<id>]` | Modell der Sitzung zeigen / wechseln |
| `:verify [<cmd>]` | Orakel-Befehl zeigen / setzen |
| `:workspace [<pfad>]` | Arbeitsverzeichnis zeigen / wechseln |
| `:plan` | aktuellen `.achilles/plan.md` ausgeben |
| `:status` | erledigte vs. offene Plan-Schritte |
| `:quit` (oder Strg-D) | Sitzung beenden |

Der Plan ist **ans Ziel gebunden**: Gibst du dasselbe Ziel erneut ein (oder
startest neu), wird ein unterbrochener Lauf *fortgesetzt*; bei einem neuen Ziel
wird der alte Plan nach `.achilles/plan.<n>.md` archiviert und frisch geplant.
Das gilt auch für die one-shot-CLI.

## Was bewusst (noch) fehlt

Das ist v0.1 — der kleinstmögliche Harness, der wirklich funktioniert. Bewusst
weggelassen, weil später dran:

- **Repo-Map / Code-Retrieval** (tree-sitter + Abhängigkeitsgraph) statt das
  Modell selbst suchen zu lassen.
- **Semantische Output-Kürzung** (Stacktrace behalten, grünes Rauschen wegwerfen).
- **Geteilte Modelle** (stärkeres Modell nur für den Planungs-Pass).
- **Echter Kontext-/Token-Budgetierer** wie in Odysseus (`n_ctx` messen, trimmen).

Die Architektur lässt für jedes davon bewusst Platz.
