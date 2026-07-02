"""
config.py — everything tunable in one place.

Loads achilles.toml from the workspace (if present) and lets environment
variables override any field, so you can point Achilles at a different model
without editing files: e.g. ACHILLES_MODEL=ornith-7b python -m achilles ...
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- the engine ---
    base_url: str = "http://localhost:8080/v1"   # llama.cpp server default; Ollama: http://localhost:11434/v1
    api_key: str = "no-key"                       # local servers ignore this
    model: str = "local-model"
    request_timeout: int = 300

    # --- the oracle (this is the heart of Achilles) ---
    # The command run after every step to decide pass/fail. Empty = no oracle,
    # which means Achilles is flying blind — it will warn you.
    verify_command: str = ""

    # --- the loop ---
    workspace: str = "."
    max_acts_per_step: int = 12   # tool calls allowed before we force a verify
    max_retries_per_step: int = 3 # red verifies tolerated before halting a step
    auto_approve_plan: bool = False
    use_git: bool = True          # checkpoint each green step (cheap rollback)

    # --- the Definition of Done (the CEILING, above the oracle FLOOR) ---
    # The planner emits a second artifact, .achilles/done.md: acceptance criteria
    # the result must satisfy (not just "nothing is broken"). cmd: criteria are
    # checked by running a command; judge: criteria by the model as a strict,
    # context-isolated reviewer. Empty done.md → falls back to oracle-green.
    use_acceptance: bool = True
    max_accept_rounds: int = 3    # fix→re-judge rounds before halting on unmet criteria
    # Model for judge: criteria. Empty → reuse `model` (no second model loaded;
    # the fresh, adversarial, evidence-cited judge context handles circularity).
    judge_model: str = ""

    # --- the tool registry (the model's hands, extensible) ---
    # Manifest tools come from achilles.toml `[[tool]]` blocks (name + command
    # template, no code). tools_dir points at a folder of Python tool plugins.
    tools: list = field(default_factory=list)
    tools_dir: str = ""

    # --- ComfyUI image generation (optional) ---
    # Set comfy_url to enable the generate_image tool: Achilles will unload the
    # LM Studio model, render via ComfyUI, then reload the model. Empty = disabled
    # (the tool isn't even offered to the model). lms_command is LM Studio's CLI;
    # workflows_dir overrides the default ~/.achilles/workflows store.
    comfy_url: str = ""
    comfy_timeout: int = 600       # seconds to wait for one render
    lms_command: str = "lms"       # LM Studio CLI used to swap the model in/out
    workflows_dir: str = ""

    # --- generation ---
    temperature: float = 0.2

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).resolve()


def _apply_toml(cfg: Config, path: Path) -> None:
    if not path.is_file():
        return
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for key, val in data.items():
        # Never let a config file silently relocate the workspace the user
        # passed on the command line.
        if key == "workspace":
            continue
        # `[[tool]]` blocks arrive as a list under "tool"; accumulate them across
        # config layers rather than overwriting.
        if key == "tool" and isinstance(val, list):
            cfg.tools.extend(val)
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, val)


def load_config(workspace: str = ".") -> Config:
    cfg = Config(workspace=workspace)

    # Config is layered, most-global to most-specific (later wins):
    #   1. the shipped achilles.toml next to the package  -> your global default
    #   2. ~/.achilles.toml                               -> your user default
    #   3. <workspace>/achilles.toml                      -> per-project override
    repo_root_toml = Path(__file__).resolve().parent.parent / "achilles.toml"
    home_toml = Path.home() / ".achilles.toml"
    workspace_toml = Path(workspace) / "achilles.toml"

    _apply_toml(cfg, repo_root_toml)
    _apply_toml(cfg, home_toml)
    if workspace_toml.resolve() != repo_root_toml.resolve():
        _apply_toml(cfg, workspace_toml)

    # Environment overrides win over every file.
    for field_name in (
        "base_url", "api_key", "model", "verify_command", "workspace", "judge_model",
    ):
        env = os.environ.get("ACHILLES_" + field_name.upper())
        if env is not None:
            setattr(cfg, field_name, env)

    return cfg
