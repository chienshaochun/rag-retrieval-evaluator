"""Expose the Skill's bundled Python package to repository-level tests."""

import sys
from pathlib import Path


SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "evaluate-rag-retrieval"
    / "scripts"
)

scripts_path = str(SKILL_SCRIPTS)
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
