"""Ensure the repo root (and thus the ``deink`` package) is importable during test
collection, mirroring the ``sys.path`` prepend that ``scripts/*.py`` use. Having this file
at the repo root also makes pytest add the root to ``sys.path`` under the default import mode.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
