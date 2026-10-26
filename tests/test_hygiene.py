import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_tracked_bytecode():
    listed = subprocess.check_output(["git", "ls-files", "*.pyc"], cwd=ROOT, text=True)
    assert listed.strip() == ""


def test_gitignore_covers_pycache_and_not_openspec_archive():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in text
    assert "*.pyc" in text or "*.py[cod]" in text
    assert ".venv/" in text
    ignored = subprocess.call(
        ["git", "check-ignore", "-q", "openspec/changes/archive/example.md"],
        cwd=ROOT,
    )
    assert ignored != 0


def test_lock_uses_scikit_learn_not_sklearn():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "scikit-learn" in project
    assert re.search(r'^\s*"sklearn\b', project, re.MULTILINE) is None
    assert (ROOT / "uv.lock").is_file()
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert "scikit-learn" in lock
    assert "name = \"sklearn\"" not in lock
