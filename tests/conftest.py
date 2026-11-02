import pytest


@pytest.fixture(autouse=True)
def _history(tmp_path, monkeypatch):
    """No test may write to the owner's real usage history."""
    monkeypatch.setenv("STYLESYNC_HISTORY", str(tmp_path / "history.jsonl"))
