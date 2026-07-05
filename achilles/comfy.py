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
from .llm import chat, LLMError
from .tools import Tool, ToolContext


# The image step is prompt-engineering, not coding — so it gets its OWN persona
# rather than riding the generic "coding agent" act-prompt. A dedicated call (while
# the LM is still loaded, before the VRAM swap) rewrites the model's terse brief
# into a dense diffusion prompt. Faithful by instruction: it enriches style, never
# swaps the subject. Best-effort — any failure falls back to the raw brief.
_IMG_PROMPT_SYSTEM = (
    "You are a prompt engineer for a text-to-image diffusion model. Rewrite the "
    "brief into ONE dense, comma-separated English prompt that a diffusion model "
    "renders well. KEEP the exact subject and every specific detail from the brief, "
    "then enrich it with concrete style, composition, lighting, colour and mood "
    "cues plus a couple of quality tags. Stay faithful — never add unrelated "
    "subjects, characters or text. Output ONLY the prompt: no preamble, no quotes."
)


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
    "render a REAL raster image — a photo, illustration, texture or hero background "
    "— with an image model and save it as JPG/PNG. Use this whenever the task wants "
    "a photograph or a painted/realistic picture: it produces actual pixels, which "
    "markup cannot, so it beats settling for a flat placeholder. For a VECTOR "
    "graphic instead — an SVG icon, logo, simple diagram or line art — write the SVG "
    "markup with write_file; that is the right tool for those. prompt: English, "
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
        # A weak model routes an SVG here because it "is an image". But this tool
        # renders raster pixels — writing them to a .svg would corrupt it. Redirect
        # to the right hand, cheaply, before any VRAM swap.
        if path.lower().endswith(".svg"):
            return ("ERROR: generate_image only makes raster images (JPG/PNG). An SVG "
                    "is vector markup — write the SVG yourself with the write_file "
                    "tool instead of rendering it here.")
        if not name:
            # Not a model-fixable error — a human must register a workflow. Say so
            # plainly; the harness-level setup-halt is a later refinement.
            return ("SETUP NEEDED: no ComfyUI workflow is registered yet. A human "
                    "must run  :workflow register <path-to-api-export.json>  and  "
                    ":workflow default <name>  once. Cannot generate images until then.")

        # Refine the brief into a richer diffusion prompt via the prompt-engineer
        # persona — while the LM is still loaded, before the swap. Faithful + best
        # effort: on any failure it returns the brief unchanged.
        prompt = _engineer_prompt(config, prompt, aspect, ctx_log)

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

        # Learn the REAL loaded model-key before we unload it, so the reload below
        # restores exactly what the user had — not config.model, which is often a
        # placeholder ("local-model") that `lms load` can't resolve. Falls back to
        # config.model if detection turns up nothing. Also refresh the remembered
        # key so a later cold start restores this exact model, even if the user
        # switched models in LM Studio since the session began.
        detected = lmstudio.loaded_llm(config.lms_command)
        if detected:
            lmstudio.remember_model(detected)
        reload_target = detected or config.model

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
        except (cc.ComfyError, lmstudio.LMStudioError) as e:
            return f"ERROR: image generation failed: {e}"
        finally:
            # Hand ComfyUI's VRAM back BEFORE reloading the LM — and do it even if
            # writing the image failed. Otherwise the reload can OOM on an 8GB card
            # while ComfyUI still holds VRAM: exactly the failure the finally is
            # supposed to prevent (Bug 4).
            try:
                client.free()
            except cc.ComfyError as e:
                ctx_log(f"   ⚠ could not free ComfyUI VRAM: {e}")
            # The one guarantee that matters: the brain always comes back.
            try:
                lmstudio.load(reload_target, config.lms_command, ctx_log)
            except lmstudio.LMStudioError as e:
                ctx_log(f"   ⚠ could not reload LM Studio model: {e}")

        return f"OK: wrote {path} ({aspect}, via workflow '{name}')."

    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string",
                       "description": "English, descriptive prompt for the image"},
            "path": {"type": "string",
                     "description": "where to save it, e.g. assets/hero.jpg"},
            "aspect": {"type": "string", "enum": ["square", "landscape", "portrait"],
                       "description": "optional, default landscape"},
            "workflow": {"type": "string",
                         "description": "only if the user named one, else omit"},
        },
        "required": ["prompt", "path"],
    }
    return Tool("generate_image", _DESCRIPTION, run, usage=_USAGE, parameters=parameters)


def _engineer_prompt(config, raw: str, aspect: str, log) -> str:
    """Rewrite the model's terse image brief into a richer diffusion prompt under a
    dedicated prompt-engineer persona. A separate chat() call, made while the LM is
    still resident (before the image swap). Best-effort: an LLM error or an empty
    reply returns the raw brief unchanged, so image generation never hinges on it."""
    try:
        out = chat(config, [
            {"role": "system", "content": _IMG_PROMPT_SYSTEM},
            {"role": "user", "content": f"Aspect: {aspect}.\nBrief: {raw}"},
        ], temperature=getattr(config, "temperature", 0.2))
    except LLMError:
        return raw
    out = (out or "").strip().strip('"').strip()
    if not out:
        return raw
    if out != raw:
        log(f"   ✎ image prompt → {out}")
    return out


def _store_dir(config) -> Path:
    """Where registered workflows live. Configurable, else the self-provisioned
    ~/.achilles/workflows."""
    d = getattr(config, "workflows_dir", "") or ""
    return Path(d) if d else (Path.home() / ".achilles" / "workflows")


# The tool runs inside the harness process, so it can print progress straight to
# the human. Kept as a module hook so it's easy to route elsewhere later.
def ctx_log(msg: str) -> None:
    print(msg)
