"""Public Databricks SDK checks through MiniLake into the native Sparkless server.

Run with ``SPARKLESS_SERVER_BIN=/path/to/sparkless-server uv run pytest
tests/test_sparkless_adapter_integration.py -q``. This deliberately launches both real
processes; the protocol unit tests in ``test_sparkless_adapter.py`` need no native binary.
"""

import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Generator

import pytest
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Disposition, Format

pytestmark = [
    pytest.mark.workflow,
    pytest.mark.skipif(
        not os.getenv("SPARKLESS_SERVER_BIN"), reason="set SPARKLESS_SERVER_BIN to run native integration"
    ),
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(url: str, process: subprocess.Popen, route: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            raise RuntimeError(f"server exited before {route} became ready:\n{stderr}")
        try:
            with urllib.request.urlopen(f"{url}{route}", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"server at {url} did not become ready at {route}")


@pytest.fixture(scope="session")
def minilake_server(tmp_path_factory) -> Generator[str, None, None]:
    """Override the regular fixture with an isolated native-Sparkless pair."""
    binary = Path(os.environ["SPARKLESS_SERVER_BIN"]).resolve()
    if not binary.is_file():
        pytest.fail(f"SPARKLESS_SERVER_BIN does not point to a file: {binary}")
    server_port = _free_port()
    minilake_port = _free_port()
    sparkless_url = f"http://127.0.0.1:{server_port}"
    minilake_url = f"http://127.0.0.1:{minilake_port}"
    sparkless = subprocess.Popen(
        [str(binary), "serve", "--host", "127.0.0.1", "--port", str(server_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    minilake = None
    try:
        _wait_ready(sparkless_url, sparkless, "/capabilities")
        project_root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            "MINILAKE_SPARKLESS_SERVER_URL": sparkless_url,
            "MINILAKE_DATA_DIR": str(tmp_path_factory.mktemp("sparkless-minilake-data")),
            "MINILAKE_NOTEBOOK": "0",
            "MINILAKE_MCP": "0",
        }
        minilake = subprocess.Popen(
            ["uv", "run", "minilake", "--port", str(minilake_port), "--host", "127.0.0.1"],
            cwd=project_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _wait_ready(minilake_url, minilake, "/_minilake/health")
        yield minilake_url
    finally:
        for process in (minilake, sparkless):
            if process is None:
                continue
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stderr:
                process.stderr.close()


@pytest.fixture
def workspace_client(minilake_server: str) -> WorkspaceClient:
    return WorkspaceClient(host=minilake_server, token="test-token", auth_type="pat")


def test_public_sdk_reads_typed_values_and_mutates_native_catalog(
    workspace_client: WorkspaceClient, minilake_server: str
):
    warehouse = workspace_client.warehouses.create(name="sparkless_sdk")
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE DATABASE phase417_sdk",
    )
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE TABLE phase417_sdk.values_table (id BIGINT, amount DECIMAL(12, 3))",
    )
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="INSERT INTO phase417_sdk.values_table VALUES (9007199254740993, 1.250), (2, NULL)",
    )

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT id, amount FROM phase417_sdk.values_table ORDER BY id DESC",
    )
    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == [[9007199254740993, "1.250"], [2, None]]
    columns = result.manifest.schema.columns
    assert [(column.name, column.type_name.value, column.type_text) for column in columns] == [
        ("id", "LONG", "BIGINT"),
        ("amount", "DECIMAL", "DECIMAL(12,3)"),
    ]
    assert [(columns[1].type_precision, columns[1].type_scale)] == [(12, 3)]

    empty = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT CAST(NULL AS DECIMAL(12, 3)) AS amount WHERE false",
    )
    assert empty.result.data_array == []
    assert empty.manifest.schema.column_count == 1
    assert empty.manifest.schema.columns[0].type_text == "DECIMAL(12,3)"
    raw_empty = requests.post(
        f"{minilake_server}/api/2.0/sql/statements",
        json={
            "warehouse_id": warehouse.id,
            "statement": "SELECT CAST(NULL AS DECIMAL(12, 3)) AS amount WHERE false",
        },
        timeout=30,
    ).json()
    assert raw_empty["manifest"]["schema"]["columns"][0]["nullable"] is True


def test_sdk_adapter_exposes_receipt_and_replays_custom_identity_once(
    workspace_client: WorkspaceClient, minilake_server: str
):
    warehouse = workspace_client.warehouses.create(name="sparkless_replay")
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE TABLE replay_table (id BIGINT)",
    )
    statement = "INSERT INTO replay_table VALUES (7)"
    url = f"{minilake_server}/api/2.0/sql/statements"
    body = {
        "warehouse_id": warehouse.id,
        "statement": statement,
        "sparkless_operation_id": "sdk-adapter-replay-1",
    }
    first = requests.post(url, json=body, timeout=30)
    second = requests.post(url, json=body, timeout=30)
    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["operation_id"] == second_body["operation_id"] == "sdk-adapter-replay-1"
    assert first_body["receipt"] == second_body["receipt"]

    query = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT count(*) AS n FROM replay_table",
    )
    assert query.result.data_array == [[1]]


def test_native_decimal_schema_is_preserved_in_arrow_external_link(workspace_client: WorkspaceClient):
    import pyarrow as pa

    warehouse = workspace_client.warehouses.create(name="sparkless_arrow")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT CAST(9007199254740993 AS BIGINT) AS id, CAST(1.250 AS DECIMAL(12, 3)) AS amount",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.ARROW_STREAM,
    )
    link = result.result.external_links[0].external_link
    response = requests.get(link, timeout=30)
    assert response.status_code == 200
    table = pa.ipc.open_stream(response.content).read_all()
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("amount").type == pa.decimal128(12, 3)
    assert table.column("id").to_pylist() == [9007199254740993]
    assert str(table.column("amount").to_pylist()[0]) == "1.250"
