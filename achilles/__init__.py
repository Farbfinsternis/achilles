"""Achilles — a minimal agentic-coding harness for small, local models.

The counterpart to Odysseus (a generalist): Achilles does one thing — write code
in small, test-verified steps — and leans on a verification oracle so that a weak
model only has to *converge*, not be right on the first try.
"""

from .config import Config, load_config
from .harness import Harness

__all__ = ["Config", "load_config", "Harness"]
__version__ = "0.1.0"
