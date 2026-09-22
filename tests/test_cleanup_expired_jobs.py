import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import cleanup_expired_jobs  # noqa: E402


class _FakeDeleteQuery:
    def __init__(self, recorder):
        self.recorder = recorder

    def in_(self, column, values):
        self.recorder.append((column, values))
        return self

    def execute(self):
        return self


class _FakeSelectResult:
    def __init__(self, rows):
        self.data = rows


class _FakeTable:
    def __init__(self, rows, delete_calls):
        self.rows = rows
        self.delete_calls = delete_calls

    def select(self, *_a, **_kw):
        return _FakeSelectFluent(self.rows)

    def delete(self):
        return _FakeDeleteQuery(self.delete_calls)


class _FakeSelectFluent:
    def __init__(self, rows):
        self.rows = rows

    def lt(self, *_a, **_kw):
        return self

    def execute(self):
        return _FakeSelectResult(self.rows)


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.delete_calls = []

    def table(self, _name):
        return _FakeTable(self.rows, self.delete_calls)


def test_dry_run_reports_count_without_deleting(monkeypatch, capsys):
    client = _FakeClient([{"id": "j1", "status": "completed", "expires_at": "2020-01-01"}])
    monkeypatch.setattr(cleanup_expired_jobs, "get_supabase_client", lambda: client)

    monkeypatch.setattr(sys, "argv", ["cleanup_expired_jobs.py"])
    exit_code = cleanup_expired_jobs.main()

    assert exit_code == 0
    assert client.delete_calls == []
    assert "would be deleted" in capsys.readouterr().out


def test_yes_flag_deletes_expired_rows(monkeypatch, capsys):
    client = _FakeClient(
        [
            {"id": "j1", "status": "completed", "expires_at": "2020-01-01"},
            {"id": "j2", "status": "failed", "expires_at": "2020-01-02"},
        ]
    )
    monkeypatch.setattr(cleanup_expired_jobs, "get_supabase_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["cleanup_expired_jobs.py", "--yes"])

    exit_code = cleanup_expired_jobs.main()

    assert exit_code == 0
    assert client.delete_calls == [("id", ["j1", "j2"])]
    assert "Deleted 2" in capsys.readouterr().out


def test_no_expired_rows_is_a_clean_noop(monkeypatch, capsys):
    client = _FakeClient([])
    monkeypatch.setattr(cleanup_expired_jobs, "get_supabase_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["cleanup_expired_jobs.py", "--yes"])

    exit_code = cleanup_expired_jobs.main()

    assert exit_code == 0
    assert client.delete_calls == []
    assert "No expired forge_jobs rows." in capsys.readouterr().out


def test_no_supabase_configured_returns_error(monkeypatch):
    monkeypatch.setattr(cleanup_expired_jobs, "get_supabase_client", lambda: None)
    monkeypatch.setattr(sys, "argv", ["cleanup_expired_jobs.py"])

    assert cleanup_expired_jobs.main() == 1


class _RaisingSelectTable:
    def select(self, *_a, **_kw):
        return self

    def lt(self, *_a, **_kw):
        return self

    def execute(self):
        raise RuntimeError("network down")


class _RaisingDeleteTable:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_a, **_kw):
        return _FakeSelectFluent(self.rows)

    def delete(self):
        return self

    def in_(self, *_a, **_kw):
        return self

    def execute(self):
        raise RuntimeError("network down")


def test_query_failure_returns_error_without_crashing(monkeypatch, capsys):
    client = _FakeClient([])
    monkeypatch.setattr(client, "table", lambda _name: _RaisingSelectTable())
    monkeypatch.setattr(cleanup_expired_jobs, "get_supabase_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["cleanup_expired_jobs.py"])

    exit_code = cleanup_expired_jobs.main()

    assert exit_code == 1
    assert "Failed to query" in capsys.readouterr().err


def test_delete_failure_returns_error_instead_of_false_success(monkeypatch, capsys):
    client = _FakeClient([{"id": "j1", "status": "completed", "expires_at": "2020-01-01"}])
    monkeypatch.setattr(client, "table", lambda _name: _RaisingDeleteTable(client.rows))
    monkeypatch.setattr(cleanup_expired_jobs, "get_supabase_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["cleanup_expired_jobs.py", "--yes"])

    exit_code = cleanup_expired_jobs.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Failed to delete" in captured.err
    assert "Deleted" not in captured.out
