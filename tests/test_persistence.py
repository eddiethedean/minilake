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
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.catalog import DataSourceFormat, TableType, VolumeType
from databricks.sdk.service.jobs import JobSettings, SqlTask, SqlTaskQuery, Task
from databricks.sdk.service.sql import ColumnInfo, CreateQueryRequestQuery

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


def _start_minilake(port: int, data_dir: str, cwd: str | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["MINILAKE_DATA_DIR"] = data_dir
    env["MINILAKE_PERSIST"] = "1"
    proc = subprocess.Popen(
        ["minilake", "--port", str(port), "--host", "127.0.0.1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
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


@pytest.mark.serial
def test_full_reset_removes_default_database_across_restart():
    """A full reset must remove unqualified SQL tables from the shared database,
    including after a process restart against the same data directory."""
    data_dir = tempfile.mkdtemp(prefix="minilake_full_reset_")
    port = _free_port()
    proc = None
    table_name = f"reset_survivor_{uuid.uuid4().hex}"
    try:
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        warehouse = client.warehouses.create(name="full_reset_before")
        created = client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"CREATE TABLE {table_name} AS SELECT 777 AS value",
        )
        assert created.status.state.value == "SUCCEEDED"

        request = urllib.request.Request(f"http://127.0.0.1:{port}/_minilake/reset?full=true", method="POST")
        assert urllib.request.urlopen(request, timeout=10).status == 200

        def assert_table_is_gone(client: WorkspaceClient, warehouse_name: str) -> None:
            warehouse = client.warehouses.create(name=warehouse_name)
            with pytest.raises(DatabricksError) as exc:
                client.statement_execution.execute_statement(
                    warehouse_id=warehouse.id,
                    statement=f"SELECT value FROM {table_name}",
                )
            assert "does not exist" in str(exc.value).lower() or "not found" in str(exc.value).lower()

        assert_table_is_gone(client, "full_reset_immediate")

        _stop_minilake(proc)
        proc = None
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        assert_table_is_gone(client, "full_reset_after_restart")
    finally:
        if proc is not None:
            _stop_minilake(proc)
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.mark.serial
def test_legacy_volume_files_migrate_on_restart():
    """Files created under the pre-UC volume path remain readable after upgrade."""
    data_dir = tempfile.mkdtemp(prefix="minilake_volume_migration_")
    port = _free_port()
    proc = None
    try:
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        catalog_name = f"migration_{uuid.uuid4().hex[:8]}"
        schema_name = "migration_schema"
        volume_name = "migration_volume"
        client.catalogs.create(name=catalog_name)
        client.schemas.create(name=schema_name, catalog_name=catalog_name)
        client.volumes.create(
            name=volume_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_type=VolumeType.MANAGED,
        )

        payload = b"legacy volume data survives upgrade\n"
        legacy_file = Path(data_dir) / "dbfs" / "Volumes" / catalog_name / schema_name / volume_name / "legacy.txt"
        legacy_file.parent.mkdir(parents=True)
        legacy_file.write_bytes(payload)
        # The process running before an upgrade cannot have written the new marker.
        (Path(data_dir) / "volumes" / ".legacy_volume_migration_complete").unlink()
        volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/legacy.txt"
        full_volume_name = f"{catalog_name}.{schema_name}.{volume_name}"

        _stop_minilake(proc)
        proc = None
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")

        assert client.files.download(volume_path).contents.read() == payload
        dbfs_data = client.dbfs.read(path=volume_path).data
        assert base64.b64decode(dbfs_data) == payload
        assert (
            Path(data_dir) / "volumes" / catalog_name / schema_name / volume_name / "legacy.txt"
        ).read_bytes() == payload
        assert not legacy_file.exists()

        client.volumes.delete(name=full_volume_name)
        with pytest.raises(DatabricksError):
            client.files.download(volume_path)
        with pytest.raises(DatabricksError):
            client.dbfs.read(path=volume_path)
    finally:
        if proc is not None:
            _stop_minilake(proc)
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.mark.serial
def test_deleted_legacy_volume_files_stay_deleted_after_restart():
    """An upgrade must not import orphaned files when a volume name is reused."""
    data_dir = tempfile.mkdtemp(prefix="minilake_orphaned_volume_")
    port = _free_port()
    proc = None
    try:
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        catalog_name = f"orphan_{uuid.uuid4().hex[:8]}"
        schema_name = "orphan_schema"
        volume_name = "reused_volume"
        full_volume_name = f"{catalog_name}.{schema_name}.{volume_name}"
        volume_path = f"/Volumes/{full_volume_name.replace('.', '/')}/old.txt"
        client.catalogs.create(name=catalog_name)
        client.schemas.create(name=schema_name, catalog_name=catalog_name)
        client.volumes.create(
            name=volume_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_type=VolumeType.MANAGED,
        )

        # Simulate an old release: volume deletion removed its UC directory but
        # left Files/DBFS content under the legacy dbfs/Volumes path.
        legacy_file = Path(data_dir) / "dbfs" / "Volumes" / catalog_name / schema_name / volume_name / "old.txt"
        legacy_file.parent.mkdir(parents=True)
        legacy_file.write_bytes(b"content from deleted volume")
        client.volumes.delete(name=full_volume_name)
        migration_marker = Path(data_dir) / "volumes" / ".legacy_volume_migration_complete"
        migration_marker.unlink()

        _stop_minilake(proc)
        proc = None
        # First upgraded startup must not migrate a volume that had already been deleted.
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        client.volumes.create(
            name=volume_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_type=VolumeType.MANAGED,
        )

        _stop_minilake(proc)
        proc = None
        # A later restart must not reinterpret the new volume as a pre-upgrade one.
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        with pytest.raises(DatabricksError):
            client.files.download(volume_path)
        with pytest.raises(DatabricksError):
            client.dbfs.read(path=volume_path)
        assert legacy_file.read_bytes() == b"content from deleted volume"
    finally:
        if proc is not None:
            _stop_minilake(proc)
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.mark.serial
def test_persisted_jobs_runs_and_outputs_survive_restart():
    """Persisted numeric job/run IDs and task output remain usable after restart."""
    data_dir = tempfile.mkdtemp(prefix="minilake_jobs_persist_")
    port = _free_port()
    proc = None
    try:
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")

        warehouse = client.warehouses.create(name="persisted_job_wh", cluster_size="Small")
        query = client.queries.create(
            query=CreateQueryRequestQuery(
                display_name="persisted_job_query",
                query_text="SELECT 42 AS answer",
                warehouse_id=warehouse.id,
            )
        )
        job = client.jobs.create(
            name="persisted_job",
            tasks=[
                Task(
                    task_key="main",
                    sql_task=SqlTask(warehouse_id=warehouse.id, query=SqlTaskQuery(query_id=query.id)),
                )
            ],
        )
        run = client.jobs.run_now(job_id=job.job_id).result(timeout=timedelta(minutes=2))
        assert run.state.result_state.value == "SUCCESS"
        task_run_id = run.tasks[0].run_id
        assert "answer" in (client.jobs.get_run_output(run_id=task_run_id).logs or "")

        _stop_minilake(proc)
        proc = None

        # Restart against the same snapshot and verify the original numeric IDs
        # still address the job, run, and task output.
        proc = _start_minilake(port, data_dir)
        client = WorkspaceClient(host=f"http://127.0.0.1:{port}", token="dev", auth_type="pat")
        assert job.job_id in {item.job_id for item in client.jobs.list()}
        assert client.jobs.get(job_id=job.job_id).settings.name == "persisted_job"

        restored_run = client.jobs.get_run(run_id=run.run_id)
        assert restored_run.run_id == run.run_id
        assert restored_run.state.result_state.value == "SUCCESS"
        assert "answer" in (client.jobs.get_run_output(run_id=run.run_id).logs or "")
        assert "answer" in (client.jobs.get_run_output(run_id=task_run_id).logs or "")

        client.jobs.update(job_id=job.job_id, new_settings=JobSettings(name="persisted_job_updated"))
        assert client.jobs.get(job_id=job.job_id).settings.name == "persisted_job_updated"

        # Restored counters must not reuse an existing key and overwrite state.
        next_job = client.jobs.create(name="created_after_restart", tasks=[])
        assert next_job.job_id != job.job_id
        assert client.jobs.get(job_id=job.job_id).settings.name == "persisted_job_updated"

        rerun = client.jobs.run_now(job_id=job.job_id).result(timeout=timedelta(minutes=2))
        assert rerun.run_id != run.run_id
        assert client.jobs.get_run(run_id=run.run_id).run_id == run.run_id

        print("✓ persisted jobs, runs, counters, and SQL task output survive a process restart")
    finally:
        if proc is not None:
            _stop_minilake(proc)
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.mark.serial
def test_relative_data_dir_supports_dbfs_and_files(tmp_path):
    """Relative data roots should work through both real HTTP file APIs."""
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir()
    port = _free_port()
    proc = _start_minilake(port, "data", cwd=str(working_dir))
    base_url = f"http://127.0.0.1:{port}"
    dbfs_path = working_dir / "data" / "dbfs" / "hello"
    try:
        dbfs_body = json.dumps({"path": "/hello", "contents": base64.b64encode(b"dbfs").decode("ascii")}).encode()
        request = urllib.request.Request(
            f"{base_url}/api/2.0/dbfs/put",
            data=dbfs_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        assert dbfs_path.read_bytes() == b"dbfs"

        request = urllib.request.Request(
            f"{base_url}/api/2.0/fs/files/hello",
            data=b"files",
            headers={"Content-Type": "application/octet-stream"},
            method="PUT",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 204
        assert dbfs_path.read_bytes() == b"files"
    finally:
        _stop_minilake(proc)
