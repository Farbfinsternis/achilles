"""Command-line interface for Achilles. See `python -m achilles --help`."""

import argparse
import sys

from .config import load_config
from .harness import Harness


def _force_utf8_output() -> None:
    """Windows consoles default to cp1252 and choke on the status glyphs.
    Force UTF-8 on our streams so output never crashes the run (errors are
    replaced, not raised, on a legacy console that still can't render a glyph)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8_output()
    parser = argparse.ArgumentParser(
        prog="achilles",
        description="A minimal agentic-coding harness for small, local models.",
    )
    parser.add_argument("goal", nargs="*",
                        help="What you want, in plain words. Omit it to start "
                             "an interactive session (REPL).")
    parser.add_argument("-w", "--workspace", default=".",
                        help="Project directory to work in (default: current).")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Auto-approve the generated plan.")
    parser.add_argument("--no-git", action="store_true",
                        help="Disable git checkpointing.")
    parser.add_argument("--no-acceptance", action="store_true",
                        help="Skip the Definition-of-Done phase (floor oracle only). "
                             "Useful for many tiny tasks where a per-task judge would "
                             "just add churn.")
    parser.add_argument("--interview", "--plan", dest="interview", action="store_true",
                        help="Planungsmodus: interview you to structure the raw prompt "
                             "into a spec (and a Definition of Done) before planning, "
                             "instead of the autopilot straight-to-plan flow.")
    parser.add_argument("--verify", default=None,
                        help="Override the verify command (the oracle).")
    args = parser.parse_args(argv)

    cfg = load_config(args.workspace)
    if args.yes:
        cfg.auto_approve_plan = True
    if args.no_git:
        cfg.use_git = False
    if args.no_acceptance:
        cfg.use_acceptance = False
    if args.verify is not None:
        cfg.verify_command = args.verify

    # No goal on the command line → drop into the interactive session.
    if not args.goal:
        from .repl import Repl
        return Repl(cfg).run()

    mode = "interview" if args.interview else "autopilot"
    ok = Harness(cfg, mode=mode).run(" ".join(args.goal))
    return 0 if ok else 1
