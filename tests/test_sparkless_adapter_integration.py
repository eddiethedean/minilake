"""Public Databricks SDK checks through MiniLake into the native Sparkless server.

Run with ``SPARKLESS_SERVER_BIN=/path/to/sparkless-server uv run pytest
tests/test_sparkless_adapter_integration.py -q``. This deliberately launches both real
processes; the protocol unit tests in ``test_sparkless_adapter.py`` need no native binary.
"""

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class SparklessFaultProxy:
    """Transparent local HTTP forwarder that can drop a committed response once."""

    def __init__(self, upstream_url: str):
        self.upstream_url = upstream_url.rstrip("/")
        self._lock = threading.Lock()
        self._drop_next: set[str] = set()
        self._delay_next: dict[str, float] = {}
        self._delay_next_request: dict[str, list[float]] = {}
        self.dropped_operation_ids: list[str] = []
        self.delayed_operation_ids: list[str] = []
        self.delayed_request_operation_ids: list[str] = []
        self.status_operation_ids: list[str] = []
        self.operation_status_snapshots: list[tuple[str, str, str]] = []
        self.statement_operation_ids: list[str] = []
        self.forwarded_statement_operation_ids: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                self._forward()

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                self._forward()

            def _forward(self):
                content_length = int(self.headers.get("Content-Length", "0"))
                request_body = self.rfile.read(content_length)
                operation_id = None
                status_operation_id = None
                if self.path == "/statements":
                    try:
                        payload = json.loads(request_body)
                        operation_id = payload.get("operation_id") if isinstance(payload, dict) else None
                    except (TypeError, ValueError):
                        pass
                    if operation_id:
                        owner._record_statement(operation_id)
                        request_delay = owner._take_request_delay(operation_id)
                        if request_delay is not None:
                            time.sleep(request_delay)
                if self.path == "/operations/status":
                    try:
                        payload = json.loads(request_body)
                        status_operation_id = payload.get("operation_id") if isinstance(payload, dict) else None
                    except (TypeError, ValueError):
                        status_operation_id = None
                    if status_operation_id:
                        owner._record_status(status_operation_id)
                if operation_id:
                    owner._record_forwarded_statement(operation_id)
                try:
                    upstream = requests.request(
                        self.command,
                        f"{owner.upstream_url}{self.path}",
                        data=request_body,
                        headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                        timeout=30,
                    )
                except requests.RequestException as exc:
                    self.send_error(502, str(exc))
                    return

                if status_operation_id:
                    try:
                        status_body = json.loads(upstream.content)
                    except (TypeError, ValueError):
                        status_body = {}
                    owner._record_status_snapshot(
                        status_operation_id,
                        str(status_body.get("execution_state", "unknown")),
                        str(status_body.get("effect_outcome", "unknown")),
                    )

                if operation_id and upstream.status_code == 200 and owner._should_drop(operation_id):
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.connection.close()
                    return
                delay_seconds = owner._take_delay(operation_id) if operation_id else None
                if delay_seconds is not None:
                    time.sleep(delay_seconds)

                self.send_response(upstream.status_code)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(upstream.content)))
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(upstream.content)
                except OSError:
                    pass

            def log_message(self, _format, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def drop_next_response(self, operation_id: str) -> None:
        with self._lock:
            self._drop_next.add(operation_id)

    def delay_next_response(self, operation_id: str, delay_seconds: float) -> None:
        with self._lock:
            self._delay_next[operation_id] = delay_seconds

    def delay_next_request(self, operation_id: str, delay_seconds: float) -> None:
        """Hold one mutation before forwarding it to the native service."""
        with self._lock:
            self._delay_next_request.setdefault(operation_id, []).append(delay_seconds)

    def _take_request_delay(self, operation_id: str) -> float | None:
        with self._lock:
            delays = self._delay_next_request.get(operation_id)
            if not delays:
                return None
            delay_seconds = delays.pop(0)
            if not delays:
                del self._delay_next_request[operation_id]
            self.delayed_request_operation_ids.append(operation_id)
            return delay_seconds

    def _should_drop(self, operation_id: str) -> bool:
        with self._lock:
            if operation_id not in self._drop_next:
                return False
            self._drop_next.remove(operation_id)
            self.dropped_operation_ids.append(operation_id)
            return True

    def _take_delay(self, operation_id: str) -> float | None:
        with self._lock:
            delay_seconds = self._delay_next.pop(operation_id, None)
            if delay_seconds is not None:
                self.delayed_operation_ids.append(operation_id)
            return delay_seconds

    def _record_status(self, operation_id: str) -> None:
        with self._lock:
            self.status_operation_ids.append(operation_id)

    def _record_status_snapshot(self, operation_id: str, state: str, outcome: str) -> None:
        with self._lock:
            self.operation_status_snapshots.append((operation_id, state, outcome))

    def _record_statement(self, operation_id: str) -> None:
        with self._lock:
            self.statement_operation_ids.append(operation_id)

    def _record_forwarded_statement(self, operation_id: str) -> None:
        with self._lock:
            self.forwarded_statement_operation_ids.append(operation_id)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


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


@dataclass
class IntegrationServers:
    minilake_url: str
    sparkless_url: str
    binary: Path
    sparkless_port: int
    fault_proxy: SparklessFaultProxy
    sparkless_processes: list[subprocess.Popen] = field(default_factory=list)

    def drop_next_sparkless_response(self, operation_id: str) -> None:
        self.fault_proxy.drop_next_response(operation_id)

    def restart_sparkless(self, env_overrides: dict[str, str] | None = None) -> None:
        current = self.sparkless_processes[-1]
        current.terminate()
        try:
            current.wait(timeout=5)
        except subprocess.TimeoutExpired:
            current.kill()
            current.wait(timeout=5)
        environment = os.environ.copy()
        environment.update(env_overrides or {})
        replacement = subprocess.Popen(
            [str(self.binary), "serve", "--host", "127.0.0.1", "--port", str(self.sparkless_port)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.sparkless_processes.append(replacement)
        _wait_ready(self.sparkless_url, replacement, "/capabilities")


@pytest.fixture(scope="session")
def minilake_server(tmp_path_factory) -> Generator[IntegrationServers, None, None]:
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
    fault_proxy = None
    servers = None
    minilake = None
    try:
        _wait_ready(sparkless_url, sparkless, "/capabilities")
        fault_proxy = SparklessFaultProxy(sparkless_url)
        servers = IntegrationServers(minilake_url, sparkless_url, binary, server_port, fault_proxy, [sparkless])
        project_root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            "MINILAKE_SPARKLESS_SERVER_URL": fault_proxy.url,
            "MINILAKE_SPARKLESS_REQUEST_TIMEOUT": "5",
            # Return the first real native status snapshot so deadline integration tests
            # can observe accepted-but-incomplete operations deterministically.
            "MINILAKE_SPARKLESS_STATUS_POLLS": "1",
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
        yield servers
    finally:
        processes = [minilake, *(servers.sparkless_processes if servers else [sparkless])]
        for process in processes:
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
        if fault_proxy:
            fault_proxy.close()


@pytest.fixture
def workspace_client(minilake_server: IntegrationServers) -> WorkspaceClient:
    return WorkspaceClient(host=minilake_server.minilake_url, token="test-token", auth_type="pat")


def test_public_sdk_reads_typed_values_and_mutates_native_catalog(
    workspace_client: WorkspaceClient, minilake_server: IntegrationServers
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
        f"{minilake_server.minilake_url}/api/2.0/sql/statements",
        json={
            "warehouse_id": warehouse.id,
            "statement": "SELECT CAST(NULL AS DECIMAL(12, 3)) AS amount WHERE false",
        },
        timeout=30,
    ).json()
    assert raw_empty["manifest"]["schema"]["columns"][0]["nullable"] is True


def test_public_sdk_identical_submissions_use_distinct_mutation_identities(
    workspace_client: WorkspaceClient,
    minilake_server: IntegrationServers,
):
    warehouse = workspace_client.warehouses.create(name="sparkless_distinct_submissions")
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE TABLE distinct_submissions (id BIGINT)",
    )
    first_operation = len(minilake_server.fault_proxy.statement_operation_ids)
    statement = "INSERT INTO distinct_submissions VALUES (1)"

    first = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=statement,
    )
    second = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=statement,
    )

    submitted_ids = minilake_server.fault_proxy.statement_operation_ids[first_operation:]
    assert len(submitted_ids) == 2, submitted_ids
    assert submitted_ids[0] != submitted_ids[1]
    assert first.status.state.value == second.status.state.value == "SUCCEEDED"
    count = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT COUNT(*) AS n FROM distinct_submissions",
    )
    assert count.result.data_array == [[2]]


def test_public_sdk_route_retains_failed_mutation_and_prevents_effects(
    workspace_client: WorkspaceClient,
    minilake_server: IntegrationServers,
):
    warehouse = workspace_client.warehouses.create(name="sparkless_failed_mutation")
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE TABLE failed_mutation (id INT)",
    )
    instance_id = requests.get(f"{minilake_server.sparkless_url}/capabilities", timeout=10).json()["server_instance_id"]
    operation_id = "sdk-failed-cast-1"
    url = f"{minilake_server.minilake_url}/api/2.0/sql/statements"
    body = {
        "warehouse_id": warehouse.id,
        "statement": "INSERT INTO failed_mutation VALUES ('not-an-integer')",
        "sparkless_operation_id": operation_id,
    }

    failed = requests.post(url, json=body, timeout=30)
    assert failed.status_code == 200, failed.text
    failed_body = failed.json()
    assert failed_body["error_code"] == "QUERYERROR"
    status = requests.post(
        f"{minilake_server.sparkless_url}/operations/status",
        json={"server_instance_id": instance_id, "operation_id": operation_id},
        timeout=10,
    )
    assert status.status_code == 200, status.text
    status_body = status.json()
    assert status_body["execution_state"] == "failed"
    assert status_body["effect_outcome"] == "failed"

    replay = requests.post(url, json=body, timeout=30)
    assert replay.status_code == failed.status_code
    assert replay.json() == failed_body
    count = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT COUNT(*) AS n FROM failed_mutation",
    )
    assert count.result.data_array == [[0]]


def test_sdk_adapter_exposes_receipt_and_replays_custom_identity_once(
    workspace_client: WorkspaceClient, minilake_server: IntegrationServers
):
    warehouse = workspace_client.warehouses.create(name="sparkless_replay")
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE TABLE replay_table (id BIGINT)",
    )
    statement = "INSERT INTO replay_table VALUES (7)"
    url = f"{minilake_server.minilake_url}/api/2.0/sql/statements"
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


def test_public_sdk_route_returns_receipt_for_successful_create_noop(
    workspace_client: WorkspaceClient, minilake_server: IntegrationServers
):
    warehouse = workspace_client.warehouses.create(name="sparkless_create_noop")
    url = f"{minilake_server.minilake_url}/api/2.0/sql/statements"
    created = requests.post(
        url,
        json={
            "warehouse_id": warehouse.id,
            "statement": "CREATE TABLE noop_receipt (id BIGINT)",
            "sparkless_operation_id": "sdk-noop-create-1",
        },
        timeout=30,
    )
    assert created.status_code == 200, created.text
    created_body = created.json()
    assert created_body["receipt"]

    no_op = requests.post(
        url,
        json={
            "warehouse_id": warehouse.id,
            "statement": "CREATE TABLE IF NOT EXISTS noop_receipt (id BIGINT)",
            "sparkless_operation_id": "sdk-noop-create-2",
        },
        timeout=30,
    )
    assert no_op.status_code == 200, no_op.text
    no_op_body = no_op.json()
    assert no_op_body["operation_id"] == "sdk-noop-create-2"
    assert no_op_body["receipt"] == created_body["receipt"]

    instance_id = requests.get(
        f"{minilake_server.sparkless_url}/capabilities", timeout=10
    ).json()["server_instance_id"]
    status = requests.post(
        f"{minilake_server.sparkless_url}/operations/status",
        json={"server_instance_id": instance_id, "operation_id": "sdk-noop-create-2"},
        timeout=10,
    )
    assert status.status_code == 200, status.text
    status_body = status.json()
    assert status_body["execution_state"] == "succeeded"
    assert status_body["effect_outcome"] == "committed"
    assert status_body["receipt"] == created_body["receipt"]


@pytest.mark.parametrize("failure_mode", ["disconnect", "deadline"])
def test_adapter_recovers_committed_write_after_real_response_fault(
    workspace_client: WorkspaceClient,
    minilake_server: IntegrationServers,
    failure_mode: str,
):
    warehouse = workspace_client.warehouses.create(name=f"sparkless_response_{failure_mode}")
    table_name = f"response_fault_{failure_mode}"
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"CREATE TABLE {table_name} (id BIGINT)",
    )
    operation_id = f"sdk-response-{failure_mode}-1"
    if failure_mode == "disconnect":
        minilake_server.drop_next_sparkless_response(operation_id)
    else:
        minilake_server.fault_proxy.delay_next_response(operation_id, 5.25)

    response = requests.post(
        f"{minilake_server.minilake_url}/api/2.0/sql/statements",
        json={
            "warehouse_id": warehouse.id,
            "statement": f"INSERT INTO {table_name} VALUES (5)",
            "sparkless_operation_id": operation_id,
        },
        timeout=30,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["operation_id"] == operation_id
    assert body["receipt"]
    if failure_mode == "disconnect":
        assert operation_id in minilake_server.fault_proxy.dropped_operation_ids
    else:
        assert operation_id in minilake_server.fault_proxy.delayed_operation_ids
    assert operation_id in minilake_server.fault_proxy.status_operation_ids
    assert minilake_server.fault_proxy.statement_operation_ids.count(operation_id) == 1

    outcome = requests.post(
        f"{minilake_server.sparkless_url}/operations/status",
        json={"server_instance_id": body["server_instance_id"], "operation_id": operation_id},
        timeout=10,
    )
    assert outcome.status_code == 200
    assert outcome.json()["effect_outcome"] == "committed"

    count = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT count(*) AS n FROM {table_name}",
    )
    assert count.result.data_array == [[1]]


def test_adapter_returns_pending_when_request_deadline_precedes_acceptance(
    workspace_client: WorkspaceClient,
    minilake_server: IntegrationServers,
):
    warehouse = workspace_client.warehouses.create(name="sparkless_pending_before_acceptance")
    table_name = "pending_before_acceptance"
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"CREATE TABLE {table_name} (id BIGINT)",
    )
    operation_id = "sdk-pending-before-acceptance-1"

    # The adapter makes two same-identity attempts, each with a five-second deadline.
    # Hold both before forwarding so operation-status lookups can prove that neither
    # attempt was accepted when the adapter returns a pending outcome.
    minilake_server.fault_proxy.delay_next_request(operation_id, 12.0)
    minilake_server.fault_proxy.delay_next_request(operation_id, 6.0)
    response = requests.post(
        f"{minilake_server.minilake_url}/api/2.0/sql/statements",
        json={
            "warehouse_id": warehouse.id,
            "statement": f"INSERT INTO {table_name} VALUES (9)",
            "sparkless_operation_id": operation_id,
        },
        timeout=20,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"]["state"] == "RUNNING"
    assert body["operation_id"] == operation_id
    assert body["effect_outcome"] == "pending"
    assert body.get("receipt") is None
    assert minilake_server.fault_proxy.statement_operation_ids.count(operation_id) == 2
    assert minilake_server.fault_proxy.forwarded_statement_operation_ids.count(operation_id) == 0

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if minilake_server.fault_proxy.forwarded_statement_operation_ids.count(operation_id) == 2:
            break
        time.sleep(0.05)
    assert minilake_server.fault_proxy.forwarded_statement_operation_ids.count(operation_id) == 2

    outcome = requests.post(
        f"{minilake_server.sparkless_url}/operations/status",
        json={"server_instance_id": body["server_instance_id"], "operation_id": operation_id},
        timeout=10,
    )
    assert outcome.status_code == 200
    assert outcome.json()["effect_outcome"] == "committed"

    count = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT count(*) AS n FROM {table_name}",
    )
    assert count.result.data_array == [[1]]


def test_sdk_returns_pending_for_accepted_mutation_before_deadline_completion(
    workspace_client: WorkspaceClient,
    minilake_server: IntegrationServers,
):
    # A one-millisecond native response deadline lets the operation continue in its
    # worker while MiniLake resolves the retained status through the real public route.
    minilake_server.restart_sparkless({"SPARKLESS_RESPONSE_TIMEOUT_MS": "1"})
    warehouse = workspace_client.warehouses.create(name="sparkless_accepted_deadline")
    capabilities = requests.get(f"{minilake_server.sparkless_url}/capabilities", timeout=10).json()
    instance_id = capabilities["server_instance_id"]
    create_operation_id = "sdk-accepted-deadline-create-1"
    create = requests.post(
        f"{minilake_server.sparkless_url}/statements",
        json={
            "session": warehouse.id,
            "kind": "sql",
            "code": "CREATE TABLE accepted_deadline (value STRING)",
            "server_instance_id": instance_id,
            "operation_id": create_operation_id,
        },
        timeout=10,
    )
    assert create.status_code in {200, 504}, create.text

    def native_status(operation_id: str) -> dict:
        response = requests.post(
            f"{minilake_server.sparkless_url}/operations/status",
            json={"server_instance_id": instance_id, "operation_id": operation_id},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        return response.json()

    deadline = time.monotonic() + 10
    while True:
        create_status = native_status(create_operation_id)
        if create_status["execution_state"] != "running":
            break
        assert time.monotonic() < deadline, create_status
        time.sleep(0.01)
    assert create_status["effect_outcome"] == "committed", create_status

    # Stay below the server's 256 KiB SQL budget while making parsing/staging longer
    # than the native response deadline, even on a fast runner.
    sql = "INSERT INTO accepted_deadline VALUES " + ",".join(f"('r{value:018d}')" for value in range(10_000))
    assert 200 * 1024 < len(sql.encode()) < 256 * 1024
    first_statement = len(minilake_server.fault_proxy.statement_operation_ids)
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=sql,
    )

    assert result.status.state.value == "RUNNING"
    operation_ids = minilake_server.fault_proxy.statement_operation_ids[first_statement:]
    assert len(operation_ids) == 1, operation_ids
    operation_id = operation_ids[0]
    assert (
        operation_id,
        "running",
        "pending",
    ) in minilake_server.fault_proxy.operation_status_snapshots

    # Deleting the warehouse closes its native session while the accepted mutation
    # still owns a session reference. The operation registry and saved catalog live
    # independently of that session, so close must not discard its eventual outcome.
    workspace_client.warehouses.delete(id=warehouse.id)

    deadline = time.monotonic() + 20
    while True:
        settled = native_status(operation_id)
        if settled["execution_state"] != "running":
            break
        assert time.monotonic() < deadline, settled
        time.sleep(0.01)
    assert settled["effect_outcome"] in {"committed", "cancelled"}, settled
    assert minilake_server.fault_proxy.statement_operation_ids.count(operation_id) == 1

    expected_rows = 10_000 if settled["effect_outcome"] == "committed" else 0
    reader = workspace_client.warehouses.create(name="sparkless_accepted_deadline_reader")
    deadline = time.monotonic() + 10
    while True:
        count = workspace_client.statement_execution.execute_statement(
            warehouse_id=reader.id,
            statement="SELECT COUNT(*) AS n FROM accepted_deadline",
        )
        if count.status.state.value == "SUCCEEDED":
            assert count.result.data_array == [[expected_rows]]
            break
        assert time.monotonic() < deadline, count
        time.sleep(0.01)


def test_sdk_adapter_rejects_reused_identity_with_changed_sql_or_session(
    workspace_client: WorkspaceClient,
    minilake_server: IntegrationServers,
):
    first_warehouse = workspace_client.warehouses.create(name="sparkless_identity_original")
    other_warehouse = workspace_client.warehouses.create(name="sparkless_identity_other")
    url = f"{minilake_server.minilake_url}/api/2.0/sql/statements"
    create = requests.post(
        url,
        json={
            "warehouse_id": first_warehouse.id,
            "statement": "CREATE TABLE identity_conflict (id BIGINT)",
        },
        timeout=30,
    )
    assert create.status_code == 200

    identity = "sdk-identity-conflict-1"
    original = {
        "warehouse_id": first_warehouse.id,
        "statement": "INSERT INTO identity_conflict VALUES (1)",
        "sparkless_operation_id": identity,
    }
    accepted = requests.post(url, json=original, timeout=30)
    assert accepted.status_code == 200

    changed_sql = requests.post(
        url,
        json={**original, "statement": "INSERT INTO identity_conflict VALUES (2)"},
        timeout=30,
    )
    assert changed_sql.status_code == 409

    changed_session = requests.post(
        url,
        json={**original, "warehouse_id": other_warehouse.id},
        timeout=30,
    )
    assert changed_session.status_code == 409

    count = workspace_client.statement_execution.execute_statement(
        warehouse_id=first_warehouse.id,
        statement="SELECT count(*) AS n FROM identity_conflict",
    )
    assert count.result.data_array == [[1]]


def test_concurrent_public_sdk_replays_share_one_mutation_receipt(
    workspace_client: WorkspaceClient,
    minilake_server: IntegrationServers,
):
    warehouse = workspace_client.warehouses.create(name="sparkless_concurrent_replay")
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE TABLE concurrent_replay (id BIGINT)",
    )
    url = f"{minilake_server.minilake_url}/api/2.0/sql/statements"
    body = {
        "warehouse_id": warehouse.id,
        "statement": "INSERT INTO concurrent_replay VALUES (42)",
        "sparkless_operation_id": "sdk-concurrent-replay-1",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: requests.post(url, json=body, timeout=30), range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    first_body, second_body = [response.json() for response in responses]
    assert first_body["operation_id"] == second_body["operation_id"] == body["sparkless_operation_id"]
    assert first_body["receipt"] == second_body["receipt"]

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT count(*) AS n FROM concurrent_replay",
    )
    assert result.result.data_array == [[1]]


def test_custom_retry_is_bound_to_the_original_server_instance(
    workspace_client: WorkspaceClient, minilake_server: IntegrationServers
):
    warehouse = workspace_client.warehouses.create(name="sparkless_instance_binding")
    url = f"{minilake_server.minilake_url}/api/2.0/sql/statements"
    created = requests.post(
        url,
        json={"warehouse_id": warehouse.id, "statement": "CREATE TABLE instance_bound (id BIGINT)"},
        timeout=30,
    )
    assert created.status_code == 200
    instance_id = created.json()["server_instance_id"]

    statement = "INSERT INTO instance_bound VALUES (9)"
    request_body = {
        "warehouse_id": warehouse.id,
        "statement": statement,
        "sparkless_operation_id": "sdk-instance-bound-1",
        "sparkless_server_instance_id": instance_id,
    }
    first = requests.post(url, json=request_body, timeout=30)
    retry = requests.post(url, json=request_body, timeout=30)
    assert first.status_code == retry.status_code == 200
    first_body = first.json()
    retry_body = retry.json()
    assert first_body["server_instance_id"] == instance_id
    assert retry_body["server_instance_id"] == instance_id
    assert first_body["operation_id"] == retry_body["operation_id"] == "sdk-instance-bound-1"
    assert first_body["receipt"] == retry_body["receipt"]

    stale_retry = requests.post(
        url,
        json={
            **request_body,
            "statement": "INSERT INTO instance_bound VALUES (10)",
            "sparkless_operation_id": "sdk-stale-instance-1",
            "sparkless_server_instance_id": "previous-instance",
        },
        timeout=30,
    )
    assert stale_retry.status_code == 409

    count = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT count(*) AS n FROM instance_bound",
    )
    assert count.result.data_array == [[1]]


def test_restart_rejects_a_retry_bound_to_the_previous_instance(
    workspace_client: WorkspaceClient, minilake_server: IntegrationServers
):
    warehouse = workspace_client.warehouses.create(name="sparkless_restart_retry")
    url = f"{minilake_server.minilake_url}/api/2.0/sql/statements"
    capabilities_url = f"{minilake_server.sparkless_url}/capabilities"
    old_instance = requests.get(capabilities_url, timeout=10).json()["server_instance_id"]
    create = requests.post(
        url,
        json={"warehouse_id": warehouse.id, "statement": "CREATE TABLE restart_retry (id BIGINT)"},
        timeout=30,
    )
    assert create.status_code == 200
    request_body = {
        "warehouse_id": warehouse.id,
        "statement": "INSERT INTO restart_retry VALUES (1)",
        "sparkless_operation_id": "sdk-restart-bound-1",
        "sparkless_server_instance_id": old_instance,
    }
    first = requests.post(url, json=request_body, timeout=30)
    assert first.status_code == 200

    minilake_server.restart_sparkless()
    new_instance = requests.get(capabilities_url, timeout=10).json()["server_instance_id"]
    assert new_instance != old_instance
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="CREATE TABLE restart_retry (id BIGINT)",
    )
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="INSERT INTO restart_retry VALUES (2)",
    )

    stale_retry = requests.post(url, json=request_body, timeout=30)
    assert stale_retry.status_code == 409
    count = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement="SELECT count(*) AS n FROM restart_retry",
    )
    assert count.result.data_array == [[1]]


def test_public_sdk_route_preserves_mutation_capacity_failure_without_effects(
    workspace_client: WorkspaceClient, minilake_server: IntegrationServers
):
    warehouse = workspace_client.warehouses.create(name="sparkless_mutation_capacity")
    minilake_server.restart_sparkless({"SPARKLESS_MAX_MUTATION_RECORDS": "1"})
    try:
        workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement="CREATE TABLE mutation_capacity (id BIGINT)",
        )

        response = requests.post(
            f"{minilake_server.minilake_url}/api/2.0/sql/statements",
            json={
                "warehouse_id": warehouse.id,
                "statement": "INSERT INTO mutation_capacity VALUES (1)",
                "sparkless_operation_id": "sdk-mutation-capacity-1",
            },
            timeout=30,
        )
        assert response.status_code == 429, response.text
        assert response.json()["error_code"] == "MUTATIONRETENTIONCAPACITYEXCEEDED"

        count = workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement="SELECT count(*) AS n FROM mutation_capacity",
        )
        assert count.result.data_array == [[0]]
    finally:
        minilake_server.restart_sparkless({"SPARKLESS_MAX_MUTATION_RECORDS": "1024"})


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
