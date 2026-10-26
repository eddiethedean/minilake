"""Persistence round-trip test.

Regression test for a real bug found while auditing the project: the
MINILAKE_PERSIST env var and persistence.py's save_state/load_state existed,
but were never actually invoked from app.py's lifespan — so setting
MINILAKE_PERSIST=1 silently did nothing. This test starts an isolated
minilake subprocess (not the shared session-scoped server used by other
tests, since this specifically needs a real process restart), creates real
state, restarts the process against the same data dir, and verifies the
state survived — using the real databricks-sdk client throughout.
"""

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from datetime import timedelta

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import DataSourceFormat, TableType
from databricks.sdk.service.sql import ColumnInfo

from minilake import persistence
from minilake.persistence import save_state


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _drain(proc: subprocess.Popen) -> str:
    """Whatever the subprocess printed, without blocking on a live process."""
    if proc.poll() is None:
        return "(process still running)"
    out, err = proc.communicate(timeout=5)
    return (out or b"").decode(errors="replace") + (err or b"").decode(errors="replace")


def _wait_healthy(proc: subprocess.Popen, url: str, timeout: float = 60.0) -> None:
    """Poll until minilake answers, failing early and loudly if it died instead.

    Checking `proc.poll()` matters: without it a crashed server is indistinguishable from a
    slow one, and the test reports a timeout while the actual traceback sits unread in a
    pipe. CI is also several times slower to start than a laptop, hence the generous
    timeout — this waits on a real process spawn, not a request.
    """
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"minilake exited with code {proc.returncode} before serving:\n{_drain(proc)}")
        try:
            if urllib.request.urlopen(f"{url}/_minilake/health", timeout=1).status == 200:
                return
        except Exception as e:
            last_error = e
            time.sleep(0.3)
    raise RuntimeError(f"minilake at {url} did not become healthy in time: {last_error}\n{_drain(proc)}")


def _start_minilake(port: int, data_dir: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["MINILAKE_DATA_DIR"] = data_dir
    env["MINILAKE_PERSIST"] = "1"
    proc = subprocess.Popen(
        ["minilake", "--port", str(port), "--host", "127.0.0.1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_healthy(proc, f"http://127.0.0.1:{port}")
    except Exception:
        _stop_minilake(proc)
        raise
    return proc


def _stop_minilake(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_save_state_serialization_failure_preserves_previous_snapshot(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    previous_snapshot = {"format_version": 1, "modules": {"catalogs": {"existing": True}}}
    snapshot_path.write_text(json.dumps(previous_snapshot))

    asyncio.run(save_state(snapshot_path, {"dbfs": lambda: {"buffer": bytearray(b"pending")}}))

    assert json.loads(snapshot_path.read_text()) == previous_snapshot
    assert not list(tmp_path.glob(".snapshot.json.*.tmp"))


def test_save_state_replace_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot.json"
    previous_snapshot = {"format_version": 1, "modules": {"catalogs": {"existing": True}}}
    snapshot_path.write_text(json.dumps(previous_snapshot))

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    asyncio.run(save_state(snapshot_path, {"catalogs": lambda: {"current": True}}))

    assert json.loads(snapshot_path.read_text()) == previous_snapshot
    assert not list(tmp_path.glob(".snapshot.json.*.tmp"))


@pytest.mark.serial
def test_persist_state_survives_restart():
    """Test: MINILAKE_PERSIST=1 really saves state on shutdown and restores
    it (including re-attaching the catalog's own DuckDB database) on the
    next startup against the same data dir."""
    data_dir = tempfile.mkdtemp(prefix="minilake_persist_")
    port = _free_port()
    proc = None
    try:
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        catalog_name = "persisted_catalog"
        client.catalogs.create(name=catalog_name)
        job = client.jobs.create(name="persisted_idempotent_job", tasks=[])
        token = "persisted-run-now-token"
        original_run = client.jobs.run_now(job_id=job.job_id, idempotency_token=token).result(
            timeout=timedelta(seconds=30)
        )
        _stop_minilake(proc)
        proc = None

        # Restart against the SAME data dir - metadata must survive.
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        restored = client.catalogs.get(name=catalog_name)
        assert restored.name == catalog_name

        # An open upload is transient. Shutting down with a bytearray buffer
        # must not make the snapshot unreadable or discard other saved state.
        empty_handle = client.dbfs.create("/unfinished_empty.txt", overwrite=True).handle
        upload_handle = client.dbfs.create("/unfinished.txt", overwrite=True).handle
        client.dbfs.add_block(upload_handle, base64.b64encode(b"pending").decode("ascii"))
        _stop_minilake(proc)
        proc = None

        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        restored = client.catalogs.get(name=catalog_name)
        assert restored.name == catalog_name
        for path, handle in (
            ("/unfinished_empty.txt", empty_handle),
            ("/unfinished.txt", upload_handle),
        ):
            with pytest.raises(Exception):
                client.dbfs.get_status(path)
            with pytest.raises(Exception):
                client.dbfs.add_block(handle, base64.b64encode(b"stale").decode("ascii"))
        # JSON snapshots stringify dictionary keys. Job and run IDs must be
        # restored as integers so both direct lookups and token retries work.
        assert client.jobs.get(job_id=job.job_id).job_id == job.job_id
        assert client.jobs.get_run(run_id=original_run.run_id).run_id == original_run.run_id
        retried_run = client.jobs.run_now(job_id=job.job_id, idempotency_token=token).result(
            timeout=timedelta(seconds=30)
        )
        assert retried_run.run_id == original_run.run_id
        assert [r.run_id for r in client.jobs.list_runs(job_id=job.job_id)] == [original_run.run_id]

        # And it must be a REAL, still-queryable catalog after restart, not
        # just restored metadata - proves the catalog's DuckDB database was
        # really re-ATTACHed, not just its name remembered.
        client.schemas.create(name="s", catalog_name=catalog_name)
        client.tables.create(
            name="t",
            catalog_name=catalog_name,
            schema_name="s",
            storage_location=f"/data/{catalog_name}/s/t",
            columns=[ColumnInfo(name="id", type_text="INTEGER")],
            table_type=TableType.MANAGED,
            data_source_format=DataSourceFormat.DELTA,
        )

        print("✓ MINILAKE_PERSIST=1 really survives a process restart")
    finally:
        if proc is not None:
            _stop_minilake(proc)
        shutil.rmtree(data_dir, ignore_errors=True)
