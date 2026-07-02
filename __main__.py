"""Convenience shim so `python __main__.py "goal"` works from the repo root.

The canonical invocation is `python -m achilles "goal"`. This just forwards to it
so you don't have to remember the -m form.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from achilles.cli import main  # noqa: E402

raise SystemExit(main())
