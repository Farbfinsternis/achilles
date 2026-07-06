# Achilles Spec-Artefakt (Planungsmodus)

Status: **Entwurf v1** · betrifft den Planungsmodus (`mode: "interview"`).

Dieses Dokument spezifiziert das **Spec-Artefakt** und das Modul `spec.py`, das im
Planungsmodus **vor** `make_plan` läuft: ein festes Interview strukturiert den
rohen Prompt in ein maschinenlesbares Spec, das anschließend Planner und
Definition of Done (DoD) speist. Das Ereignis-/Kommando-Protokoll der Oberfläche
steht in [protocol.md](protocol.md).

Leitidee: Ein roher Prompt — oft nicht einmal in der Trainingssprache des Modells
— ist die größte Fehlerquelle für kleine Modelle. Das Spec normalisiert ihn
strukturell und trennt zwei Sprachebenen sauber:

- **Reasoning/Struktur → Englisch** (Planner und DoD denken in ihrer starken Sprache).
- **Produkt-Content → Originalsprache, wörtlich** (Namen, Copy, Beschriftungen).

Diese Trennlinie ist der Kern; ohne sie importiert man einen Sprach-Mismatch in
die Akzeptanzprüfung (siehe `contains`-Prosa-/Sprach-Bugklasse).

---

## 1. `.achilles/spec.md`

```markdown
# Achilles — Spec

> Spec-Version: 1
> Source-Language: de
> Mode: interview

## Original goal
```text
<roher User-Prompt, wörtlich, unverändert — die Content-Wahrheit>
```

## Purpose
A single-page site advertising a German neighbourhood bakery.

## Audience
Local walk-in customers, mostly older, mobile-first.

## Core features
- Hero with bakery name and tagline
- Opening hours
- Contact / location block

## Prototype scope
Static HTML/CSS/JS, no backend, no cart.

## UI / UX
Warm, rustic; large tap targets; single scroll, no nav menu.

## Verbatim content
- "Bäckerei Sonnenschein"
- "Frische Brötchen täglich"
- "Öffnungszeiten"
```

**Zwei-Ebenen-Regel, hart im Format verankert:**

- `## Purpose … ## UI/UX` = **Englisch** (Reasoning-Input).
- `## Original goal` (fenced, wörtlich) + `## Verbatim content` (gequotete
  Literale) = **Originalsprache, nie übersetzt**.

**Resume-Keying:** Statt eines `> Goal:`-Einzeilers (ein 300-Zeilen-Prompt passt
nicht in eine `>`-Zeile) vergleicht der Resume den **`## Original goal`-Block
exakt** — dieselbe Gleichheits-Logik wie für `plan.md` heute in
`harness.py` (`_stored_goal() != goal`), nur gegen den Block statt die Zeile. Kein
Hash nötig.

---

## 2. Datenmodell (`spec.py`)

```python
@dataclass(frozen=True)
class Slot:
    field: str          # "purpose"
    prompt: str         # "Zweck des Projekts?"
    default: str        # Fallback-Text ODER "" = im normalize()-Call aus goal ableiten

@dataclass
class Spec:
    source_language: str
    original_goal: str          # wörtlich, Originalsprache → Content-Pin
    purpose: str                # EN
    audience: str               # EN
    features: list[str]         # EN
    scope: str                  # EN
    ui_ux: str                  # EN (kann "" sein)
    verbatim: list[str]         # Originalsprache, wörtlich → DoD
```

Fester Fragenkatalog, im Modul verankert (kein Modell erzeugt Fragen):

```python
SLOTS = [
    Slot("purpose",  "Zweck des Projekts?",            ""),
    Slot("audience", "Zielgruppe?",                    "general audience"),
    Slot("features", "Kernfunktionen?",                ""),
    Slot("scope",    "Prototyp-Scope?",                "minimal working prototype"),
    Slot("ui_ux",    "UI/UX — Aussehen & Bedienung?",  ""),
]
```

---

## 3. Öffentliche API (`spec.py`)

```python
def parse_spec(text: str) -> Spec: ...
    # Sektionen lesen (Header-getrieben, wie planner die "- [ ]"-Zeilen liest).

def render_spec(spec: Spec) -> str: ...
    # Spec → spec.md (mit Original-goal-Block + Verbatim-Liste).

def normalize(config, answers: dict[str, str], original_goal: str) -> Spec: ...
    # DER EINE Modell-Call des Planungsmodus. Siehe §4.

def en_goal(spec: Spec) -> str: ...
    # Englische Sicht als EIN goal-String für make_plan / make_acceptance.

def verbatim_criteria(spec: Spec) -> list[Criterion]: ...
    # Verbatim-Strings → deterministische DoD-Kriterien. Siehe §6.

def route_answer(raw: str) -> tuple[str, str]: ...
    # Rein heuristischer 2-Intent-Router. Siehe §5.
```

---

## 4. `normalize()` — der einzige Modell-Call

Input: die rohen Interview-Antworten (Originalsprache, teils Defaults) +
`original_goal`. Vertrag im System-Prompt, in *einer* strikten Regel:

> **Übersetze Purpose/Audience/Features/Scope/UI ins Englische. Extrahiere aus den
> Antworten die wörtlichen Content-Strings (Namen, Copy, Beschriftungen) und gib
> sie UNVERÄNDERT in Originalsprache zurück. Übersetze diese Strings NICHT.**

Constrained-JSON-Shape (analog `PLAN_SCHEMA` / `ACCEPT_SCHEMA` in
`planner.py` / `acceptance.py`):

```json
{
  "purpose": "…EN…", "audience": "…EN…",
  "features": ["…EN…"], "scope": "…EN…", "ui_ux": "…EN…",
  "verbatim": ["Bäckerei Sonnenschein", "Frische Brötchen täglich", "Öffnungszeiten"]
}
```

Leere Slots mit leerem `default` („aus goal ableiten") werden im **selben** Call
aus `original_goal` gefüllt — deshalb *ein* Call, nicht fünf. Ergebnis → `Spec` →
`render_spec` → `.achilles/spec.md`. Dies ist der **einzige** Modell-Kontakt im
Planungsmodus vor `make_plan`.

---

## 5. Interview-Router (Entscheidung)

**Rein heuristisch, zwei Intents, kein Modell im Loop:**

```
skip   → leer / „egal" / „weiß nicht" → Default, nächste Frage
answer → alles andere → Literal-Wert als Slot-Antwort, nächste Frage
```

- **Kein `back`-Intent, keine Meta-Frage-Klassifikation.** Korrekturen laufen
  nicht über den Stream, sondern über das **Spec-Approval-Gate**
  (`approval.request {subject:"spec"}`, `decision:"edit"`): der User sieht das
  fertige Spec und ändert dort. Ein perfekter Inline-Router ist damit unnötig.
- **Warum so:** Determinismus war das Kernprinzip; auf langsamer, lokaler
  Hardware wäre ein Klassifikator-Call pro Interview-Turn teuer für zweifelhaften
  Gewinn. Der einzige Modell-Kontakt bleibt `normalize()`.
- **Non-interaktiv** (kein TTY / Headless): alle Slots nehmen ihren Default →
  leeres Interview → Autopilot-äquivalent; CI-/Test-Pfad bleibt grün.

Dies ersetzt bewusst den früher skizzierten 3-Intent-Router; [protocol.md](protocol.md)
§6 ist entsprechend angeglichen.

---

## 6. Verbatim → Definition of Done

Ein Verbatim-String ist ein Content-Anker, aber `contains:` in `acceptance.py`
braucht einen **Pfad** — und zur Spec-Zeit steht die Dateistruktur noch nicht
fest (der Planner läuft erst danach). Deshalb ein **pfadloses** Primitiv.

### 6.1 Neues Criterion-Kind `contains_any`

„Der String muss in *irgendeiner* Projektdatei wörtlich vorkommen." Das ist
philosophisch das Richtige: „«Bäckerei Sonnenschein» muss im Produkt stehen" ist
eine **Content**-Anforderung, keine Struktur-Anforderung; eine Datei-Bindung wäre
eine Über-Constraint ohne Informationsgrundlage.

```python
def verbatim_criteria(spec):
    return [Criterion(kind="contains_any", text=s) for s in spec.verbatim]
```

**Prüf-Semantik (Erweiterung in `acceptance.py`):**

- **Dateimenge:** alle als UTF-8 dekodierbaren Textdateien unter der
  Projektwurzel. **Ausgeschlossen:** `.git/`, **`.achilles/`**, `node_modules/`,
  `__pycache__/`. Binär-/Asset-Dateien fallen über Dekodier-Fehler automatisch
  raus.
  - `.achilles/` **muss** raus: `spec.md` und `done.md` enthalten die
    Verbatim-Strings selbst — ohne Ausschluss wäre das Kriterium durch die
    eigenen Artefakte trivial grün (falsch-grün).
- **Matching:** Haystack und Needle werden vor dem Vergleich normalisiert:
  1. **HTML-Entities dekodieren** (`&Ouml;ffnungszeiten` → „Öffnungszeiten"),
  2. **jede Whitespace-Folge** (inkl. Zeilenumbrüche) zu einem Leerzeichen
     kollabieren.
  **Casing bleibt strikt** (byte-genau nach Normalisierung).
- **Warum diese Normalisierung:** verhindert die häufigste Falsch-Rot-Klasse in
  deutschsprachigem Content — Entity-kodierte Umlaute rendern korrekt, würden
  byte-roh aber scheitern; harmloser Markup-Umbruch (`Frische\n  Brötchen`) wird
  toleriert. Casing bleibt strikt, weil die exakte Schreibweise einer
  Marke/Headline Teil der Anforderung ist.
- **Bewusstes Residuum:** ein Inline-Tag *mitten* in der Phrase
  („Frische `<em>`Brötchen`</em>` täglich") bricht das Match. Selten bei kurzen
  Content-Strings; qualitativ vom `judge:` mitabgedeckt. Nicht in `contains_any`
  gelöst, sonst wird die Regel unvorhersehbar.
- Dispatch: `contains_any` reiht sich in `_criterion_to_call` / `_check_mechanical`
  ein (grep über die Dateimenge statt `file_contains` auf einen Pfad) und wird
  wie die anderen mechanischen Kriterien geprüft.

### 6.2 DoD-Zusammenbau — `_make_and_save_dod(seed=…)`

Die Signatur bekommt `seed: list[Criterion] = ()`.

1. **Seed zuerst** (die harten, deterministischen `contains_any`-Anker), dann die
   modell-erzeugten Kriterien aus `make_acceptance(cfg, en_goal(spec), tree)`.
2. **Dedup-Key** = `(kind, normalisierter_text)` — dieselbe Normalisierung wie
   §6.1.
3. **Kollisionsregel:** ist ein Verbatim-String bereits als Seed-`contains_any`
   vorhanden, werden **modell-erzeugte `contains:` / `contains_any:` mit
   demselben Text verworfen**. Der deterministische Anker gewinnt gegen die
   brüchige, pfadgebundene Modell-Variante.

Regel 3 ist der eigentliche Zweck: sie verhindert, dass das Modell über eine
pfadgebundene `contains: index.html :: …`-Zeile genau die Falsch-Rot-Brüchigkeit
wieder einschleust, die `contains_any` gerade beseitigt.

```python
criteria  = verbatim_criteria(spec)                      # deterministisch
criteria += make_acceptance(cfg, en_goal(spec), tree)    # Modell: exists/judge aus EN-Spec
criteria  = _dedup_with_seed_priority(criteria)
# render_acceptance(spec.original_goal, criteria) → .achilles/done.md
```

Autopilot ruft mit `seed=()` → Verhalten unverändert.

---

## 7. Einbau in `harness.run()`

```python
def run(self, goal):
    ...
    if self.mode == "interview":
        spec = self._load_spec()
        if spec and self._spec_goal(spec) != goal:
            self._archive_spec(); spec = None
        if not spec:
            answers = self._interview(SLOTS)              # Stream über Channel.request
            spec = spec_mod.normalize(self.cfg, answers, goal)
            self._save_spec(spec)
            if self._approve_spec(spec) is None:          # Gate 1 (spec)
                return False
        goal_for_plan   = spec_mod.en_goal(spec)          # EN → Planner
        self._content_goal = spec.original_goal           # Orig → Content-Pin
        seed_dod        = spec_mod.verbatim_criteria(spec)
    else:                                                  # autopilot
        goal_for_plan   = goal
        self._content_goal = goal
        seed_dod        = []

    steps = make_plan(self.cfg, goal_for_plan, tree)
    ...
    self._make_and_save_dod(goal_for_plan, tree, seed=seed_dod)
    plan = self._approve_loop(...)                        # Gate 2 (plan, existiert)
    return self._execute(self._content_goal, plan)
```

**Kritisch:** `_execute` pinnt weiter die **Originalsprache** als Content-Wahrheit
(`self._content_goal`, vgl. heutigen Goal-Pin in `harness.py`). Die
EN-Übersetzung geht nur an Planner/DoD, **nie** an den Executor — sonst baut er
englische UI statt der Originalsprache.

---

## 8. Interview-I/O über die Channel-Grenze

Das Interview wird **von Anfang an auf der `Channel`-Grenze** aus
[protocol.md](protocol.md) §5 gebaut — kein Terminal-Wegwerf-Interview.
Minimal-Umfang jetzt:

- `Channel` mit `emit(type, data)` und `request(type, data) -> data`, injiziert
  statt des heutigen `log`-Callbacks in `Harness.__init__`.
- `log` wird zu `emit("log", …)` gewrappt; die zwei bestehenden Approval-Gates
  und das Interview laufen über `request`.
- `_interview(slots)` emittiert pro Slot `interview.question` und blockiert auf
  die `answer`-Antwort; `route_answer` deutet sie (§5).

Die vollständige semantische Event-Typisierung (`step.started`,
`verify.result`, …) bleibt ein späterer mechanischer Durchgang; sie ist für das
Interview nicht nötig.

---

## 9. Zusammenfassung der getroffenen Entscheidungen

| # | Entscheidung |
|---|--------------|
| 1 | Zwei-Ebenen-Spec: Struktur EN, Content (`Original goal` + `Verbatim`) wörtlich in Originalsprache. |
| 2 | Resume-Keying über exakten Vergleich des `## Original goal`-Blocks (kein Hash). |
| 3 | `normalize()` ist der einzige Modell-Call; füllt leere Slots im selben Call aus dem goal. |
| 4 | Interview-Router rein heuristisch, 2 Intents (`skip`/`answer`), kein Modell im Loop; Korrektur über Gate 1. |
| 5 | Non-interaktiv = alle Defaults = Autopilot-äquivalent. |
| 6 | Neues `contains_any`-Primitiv (pfadlos) für Verbatim-Strings; `.achilles/` von der Suche ausgeschlossen. |
| 7 | Matching: Entity-Dekodierung + Whitespace-Kollaps, Casing strikt; Inline-Tag-Split bleibt dem `judge:`. |
| 8 | `_make_and_save_dod(seed=…)`: Seed zuerst, Dedup, Seed-Anker schlägt modell-erzeugte pfadgebundene Duplikate. |
| 9 | Content-Pin in `_execute` bleibt Originalsprache; EN geht nur an Planner/DoD. |
| 10 | Interview auf der `Channel`-Grenze gebaut, nicht Terminal-only. |
