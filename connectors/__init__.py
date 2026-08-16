import os
import sys

# connectors/ is a plain sibling of config.py at the project root (not a
# nested subpackage), and every connector module does `from config import
# ...` (an absolute import). That only resolves if the project root is on
# sys.path — true in local dev because orchestrator_api.py's ad-hoc
# connector-factory happens to insert it first, but NOT guaranteed for
# every other context that imports this package (CI/test runners, a
# packaged deployment, etc.), which surfaces as `ModuleNotFoundError: No
# module named 'config'`. Anchoring off this file's own location (always
# correct, regardless of caller or cwd) makes the import self-contained.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .base import BaseConnector
from .factory import get_connector

__all__ = ["BaseConnector", "get_connector"]
