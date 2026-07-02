"""Enables `python -m achilles "your goal"`.

Also tolerates `python achilles` (pointing Python at the directory), which runs
this file as a loose script with no package context — the relative import would
fail, so we fall back to putting the repo root on sys.path and importing by name.
"""

try:
    from .cli import main
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from achilles.cli import main

raise SystemExit(main())
