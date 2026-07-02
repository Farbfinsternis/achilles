"""
comfy.py — the `generate_image` tool: the model's one hand for making pictures.

This is where the three pieces meet. The model, mid-build, says "I need an image
of X here" — nothing about ComfyUI, workflows or VRAM — and this tool performs the
whole atomic choreography behind that intent:

    validate aspect (while the LM is still loaded → fail cheap)
    │  unload the LM Studio model            (lmstudio.unload_all)
    │  queue the workflow, wait for the PNG  (comfy_client)
    │  write the image into the workspace
    │  hand VRAM back                        (comfy_client.free)
    └─ reload the LM Studio model            (lmstudio.load)  ← in finally, ALWAYS

Because Achilles only talks to the model BETWEEN tool calls, the model never
notices its brain was swapped out during this one blocking call. The `finally`
guarantees the brain comes back even if ComfyUI fails.

The model gets a terse one-line result; anything richer (a preview) is a later
polish printed straight to the human, past the model's context.
"""

from pathlib import Path

from . import comfy_client as cc
from . import lmstudio
from . import workflows as wf
from .tools import Tool, ToolContext


# Shown to the model in its tool list. Deliberately evocative: a weak model won't
# reach for image generation unless the description nudges it to CREATE assets
# rather than link placeholders. Aspect is a label, never pixels.
_USAGE = (
    "```act\n"
    "tool: generate_image\n"
    "prompt: warm rustic bakery interior, fresh bread, morning light, photographic\n"
    "path: assets/hero.jpg\n"
    "aspect: landscape\n"
    "```"
)
_DESCRIPTION = (
    "generate a REAL raster image (JPG/PNG) and save it into the workspace. This "
    "is the ONLY correct way to produce a picture, photo, image or illustration. "
    "Do NOT hand-write an SVG, a CSS gradient, or a placeholder as a substitute — "
    "for ANY image the task asks for, call this tool. prompt: English, "
    "descriptive. path: where to save it (e.g. assets/hero.jpg). aspect: square | "
    "landscape | portrait (optional, default landscape). workflow: only if the "
    "user named one, else omit."
)


def build_tool(config) -> Tool:
    """Construct the generate_image Tool, closing over config (comfy_url, model,
    lms_command, the workflow store) — the model-facing run() takes only intent."""
    store = wf.Store(_store_dir(config))

    def run(args: dict, body, ctx: ToolContext) -> str:
        prompt = (args.get("prompt") or body or "").strip()
        path = (args.get("path") or "").strip()
        aspect = (args.get("aspect") or "landscape").strip().lower()
        name = (args.get("workflow") or "").strip() or store.get_default()

        if not prompt:
            return "ERROR: generate_image needs a `prompt` (English, descriptive)."
        if not path:
            return "ERROR: generate_image needs a `path` to save the image to."
        if not name:
            # Not a model-fixable error — a human must register a workflow. Say so
            # plainly; the harness-level setup-halt is a later refinement.
            return ("SETUP NEEDED: no ComfyUI workflow is registered yet. A human "
                    "must run  :workflow register <path-to-api-export.json>  and  "
                    ":workflow default <name>  once. Cannot generate images until then.")

        # Resolve the graph BEFORE touching VRAM: a bad aspect or missing workflow
        # fails here, cheaply, with the LM still loaded.
        try:
            graph = wf.apply(store, name, prompt, aspect)
        except wf.WorkflowError as e:
            return f"ERROR: {e}"

        if not lmstudio.available(config.lms_command):
            return (f"ERROR: `{config.lms_command}` (LM Studio CLI) not on PATH — "
                    "cannot swap the model out for image generation.")

        client = cc.ComfyClient(config.comfy_url)
        if not client.reachable():
            return (f"ERROR: ComfyUI not reachable at {config.comfy_url}. "
                    "Is it running?")

        target = ctx.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            lmstudio.unload_all(config.lms_command, ctx_log)
            prompt_id = client.queue_prompt(graph)
            ctx_log(f"   ComfyUI rendering ({name}, {aspect})…")
            outputs = client.wait_for_outputs(
                prompt_id, timeout=config.comfy_timeout,
                on_tick=lambda s: None)
            image = cc.first_image(outputs)
            if not image:
                return "ERROR: ComfyUI finished but produced no image."
            target.write_bytes(client.image_bytes(image))
            client.free()
        except (cc.ComfyError, lmstudio.LMStudioError) as e:
            return f"ERROR: image generation failed: {e}"
        finally:
            # The one guarantee that matters: the brain always comes back.
            try:
                lmstudio.load(config.model, config.lms_command, ctx_log)
            except lmstudio.LMStudioError as e:
                ctx_log(f"   ⚠ could not reload LM Studio model: {e}")

        return f"OK: wrote {path} ({aspect}, via workflow '{name}')."

    return Tool("generate_image", _DESCRIPTION, run, usage=_USAGE)


def _store_dir(config) -> Path:
    """Where registered workflows live. Configurable, else the self-provisioned
    ~/.achilles/workflows."""
    d = getattr(config, "workflows_dir", "") or ""
    return Path(d) if d else (Path.home() / ".achilles" / "workflows")


# The tool runs inside the harness process, so it can print progress straight to
# the human. Kept as a module hook so it's easy to route elsewhere later.
def ctx_log(msg: str) -> None:
    print(msg)
