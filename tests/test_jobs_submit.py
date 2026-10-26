"""`POST /jobs/runs/submit` — one-shot runs with no job behind them.

This is the path `databricks bundle run` and most CI code take. The point of these
tests is that submit reuses the *same* DAG scheduler `run-now` does: dependencies,
`run_if` skipping and cancellation all have to behave identically, and a submitted
run must be visible to `runs/get` / `runs/list` like any other.
"""

import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.jobs import (
    RunIf,
    SqlTask,
    SqlTaskFile,
    SqlTaskQuery,
    SubmitTask,
    TaskDependency,
)
from databricks.sdk.service.sql import CreateQueryRequestQuery

from minilake.services import jobs

RUN_TIMEOUT = timedelta(minutes=5)


@pytest.fixture
def warehouse(workspace_client: WorkspaceClient) -> str:
    return workspace_client.warehouses.create(name="submit_wh", cluster_size="Small").id


@pytest.fixture
def query_id(workspace_client: WorkspaceClient, warehouse: str) -> str:
    return workspace_client.queries.create(
        query=CreateQueryRequestQuery(display_name="submit q", query_text="SELECT 1 AS n", warehouse_id=warehouse)
    ).id


def _task(task_key: str, warehouse: str, query_id: str, **kwargs) -> SubmitTask:
    return SubmitTask(
        task_key=task_key,
        sql_task=SqlTask(warehouse_id=warehouse, query=SqlTaskQuery(query_id=query_id)),
        **kwargs,
    )


def _import_bytes(client: WorkspaceClient, path: str, content: bytes) -> None:
    client.api_client.do(
        "POST",
        f"/api/2.0/workspace-files/import-file{path}",
        query={"overwrite": "true"},
        headers={"Content-Type": "application/octet-stream"},
        data=content,
    )


@pytest.mark.serial
@pytest.mark.workflow
def test_submit_runs_without_a_job(workspace_client: WorkspaceClient, warehouse: str, query_id: str):
    """A submitted run executes and terminates, carrying no job_id."""
    waiter = workspace_client.jobs.submit(run_name="one-shot", tasks=[_task("only", warehouse, query_id)])
    run = waiter.result(timeout=RUN_TIMEOUT)

    assert run.state.result_state.value == "SUCCESS"
    assert run.job_id is None, "a submitted run belongs to no job"
    assert run.run_name == "one-shot"
    assert len(run.tasks) == 1


@pytest.mark.serial
@pytest.mark.workflow
def test_submitted_run_is_visible_to_get_and_list(workspace_client: WorkspaceClient, warehouse: str, query_id: str):
    """A submitted run is an ordinary run everywhere else in the API."""
    run_id = (
        workspace_client.jobs.submit(run_name="visible", tasks=[_task("only", warehouse, query_id)])
        .result(timeout=RUN_TIMEOUT)
        .run_id
    )

    fetched = workspace_client.jobs.get_run(run_id=run_id)
    assert fetched.run_id == run_id
    assert fetched.run_page_url, "the CLI parses a run id out of this URL"

    assert run_id in {r.run_id for r in workspace_client.jobs.list_runs()}


@pytest.mark.serial
@pytest.mark.workflow
def test_submit_honours_task_dependencies(workspace_client: WorkspaceClient, warehouse: str, query_id: str):
    """The DAG scheduler applies to submitted runs: a dependent task runs after
    its parent, and `run_if` still skips."""
    run = workspace_client.jobs.submit(
        run_name="dag",
        tasks=[
            _task("first", warehouse, query_id),
            _task(
                "second",
                warehouse,
                query_id,
                depends_on=[TaskDependency(task_key="first")],
            ),
            SubmitTask(
                task_key="never",
                sql_task=SqlTask(warehouse_id=warehouse, query=SqlTaskQuery(query_id="missing")),
                depends_on=[TaskDependency(task_key="first")],
                run_if=RunIf.AT_LEAST_ONE_FAILED,
            ),
        ],
    ).result(timeout=RUN_TIMEOUT)

    by_key = {t.task_key: t for t in run.tasks}
    assert by_key["first"].state.result_state.value == "SUCCESS"
    assert by_key["second"].state.result_state.value == "SUCCESS"
    # `first` succeeded, so an AT_LEAST_ONE_FAILED branch must not run at all.
    assert by_key["never"].state.life_cycle_state.value == "SKIPPED"


@pytest.mark.error
@pytest.mark.parametrize("graph", ["unknown_dependency", "cycle"])
def test_submit_rejects_invalid_dependency_graph(
    workspace_client: WorkspaceClient, warehouse: str, query_id: str, graph: str
):
    if graph == "unknown_dependency":
        tasks = [
            _task(
                "dependent",
                warehouse,
                query_id,
                depends_on=[TaskDependency(task_key="missing")],
            )
        ]
        expected_message = "unknown task"
    else:
        tasks = [
            _task("first", warehouse, query_id, depends_on=[TaskDependency(task_key="second")]),
            _task("second", warehouse, query_id, depends_on=[TaskDependency(task_key="first")]),
        ]
        expected_message = "cycle"

    with pytest.raises(DatabricksError) as exc:
        workspace_client.jobs.submit(run_name=f"invalid-{graph}", tasks=tasks)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert expected_message in str(exc.value).lower()
    assert list(workspace_client.jobs.list_runs()) == []


@pytest.mark.serial
@pytest.mark.workflow
def test_submit_sql_task_directory_fails_terminally(workspace_client: WorkspaceClient, warehouse: str):
    path = f"/Shared/sql_task_dir_{uuid4().hex[:8]}"
    workspace_client.workspace.mkdirs(path=path)

    run = workspace_client.jobs.submit(
        run_name="sql-directory",
        tasks=[SubmitTask(task_key="directory", sql_task=SqlTask(warehouse_id=warehouse, file=SqlTaskFile(path=path)))],
    ).result(timeout=RUN_TIMEOUT)

    assert run.state.life_cycle_state.value == "TERMINATED"
    assert run.state.result_state.value == "FAILED"
    assert "not a file" in (run.tasks[0].state.state_message or "")
    output = workspace_client.jobs.get_run_output(run_id=run.run_id)
    assert "not a file" in (output.error or "")


@pytest.mark.serial
@pytest.mark.workflow
def test_submit_sql_task_invalid_utf8_fails_terminally(workspace_client: WorkspaceClient, warehouse: str):
    path = f"/Shared/sql_task_invalid_utf8_{uuid4().hex[:8]}.sql"
    _import_bytes(workspace_client, path, b"SELECT \xff")

    run = workspace_client.jobs.submit(
        run_name="sql-invalid-utf8",
        tasks=[
            SubmitTask(task_key="invalid-utf8", sql_task=SqlTask(warehouse_id=warehouse, file=SqlTaskFile(path=path)))
        ],
    ).result(timeout=RUN_TIMEOUT)

    assert run.state.life_cycle_state.value == "TERMINATED"
    assert run.state.result_state.value == "FAILED"
    output = workspace_client.jobs.get_run_output(run_id=run.run_id)
    assert "Unable to read workspace SQL file" in (output.error or "")
    assert "codec can't decode" in (output.error or "")


def test_run_executor_exception_marks_run_failed_with_diagnostics(monkeypatch):
    run_id = 987654321
    run = {
        "run_id": run_id,
        "job_id": None,
        "state": {"life_cycle_state": "PENDING", "result_state": None, "state_message": None},
        "_tasks": [{"task_key": "unexpected"}],
    }
    jobs._state["runs"][run_id] = run
    sibling_started = asyncio.Event()
    release_sibling = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def fail_executor(task, argv):
        if task.task_key == "unexpected":
            await sibling_started.wait()
            raise RuntimeError("executor exploded")
        sibling_started.set()
        try:
            await release_sibling.wait()
        finally:
            sibling_finished.set()
        return jobs.RunTaskInfo(task_key=task.task_key, state=jobs.RunState(life_cycle_state="TERMINATED"))

    run["_tasks"].append({"task_key": "sibling"})

    monkeypatch.setattr(jobs, "_execute_task", fail_executor)
    try:

        async def run_with_sibling():
            running = asyncio.create_task(jobs._execute_run(run_id, []))
            await asyncio.wait_for(sibling_started.wait(), timeout=1)
            await asyncio.sleep(0.01)
            assert not running.done()
            assert run["state"]["life_cycle_state"] == "RUNNING"
            release_sibling.set()
            await running

        asyncio.run(run_with_sibling())
        assert sibling_finished.is_set()
        assert run["state"]["life_cycle_state"] == "TERMINATED"
        assert run["state"]["result_state"] == "FAILED"
        assert "executor exploded" in run["state"]["state_message"]
        assert "RuntimeError: executor exploded" in jobs._state["task_outputs"][run_id]["error"]
    finally:
        jobs._state["runs"].pop(run_id, None)
        jobs._state.get("task_outputs", {}).pop(run_id, None)


@pytest.mark.serial
def test_submit_idempotency_token_reuses_the_run(workspace_client: WorkspaceClient, warehouse: str, query_id: str):
    """Re-submitting with the same token returns the original run, not a second one."""
    first = workspace_client.jobs.submit(
        run_name="idem",
        idempotency_token="token-abc",
        tasks=[_task("only", warehouse, query_id)],
    )
    second = workspace_client.jobs.submit(
        run_name="idem",
        idempotency_token="token-abc",
        tasks=[_task("only", warehouse, query_id)],
    )
    assert first.run_id == second.run_id


@pytest.mark.error
def test_submit_without_tasks_is_rejected(workspace_client: WorkspaceClient):
    with pytest.raises(DatabricksError) as exc:
        workspace_client.jobs.submit(run_name="empty", tasks=[])
    assert "tasks" in str(exc.value).lower()
