"""Protocol and retry tests for the optional native Sparkless SQL backend."""

import httpx
import pytest

from minilake.errors import DatabricksError
from minilake.services import sparkless_adapter as adapter


def _capabilities(instance="instance-1", agent="1", mutation="1"):
    return {
        "agent_protocol_version": agent,
        "mutation_protocol_version": mutation,
        "server_instance_id": instance,
        "sql": {"reads": ["SELECT", "COUNT(*)"], "mutations": ["CREATE TABLE", "INSERT INTO"]},
    }


def _result(rows=None, operation=None):
    body = {
        "status": "ok",
        "data": {
            "application/json": {
                "schema": {
                    "type": "struct",
                    "fields": [
                        {"name": "n", "type": "long", "nullable": False, "metadata": {}},
                        {"name": "amount", "type": "decimal(12, 3)", "nullable": True, "metadata": {}},
                    ],
                },
                "data": [[9007199254740993, "1.250"]] if rows is None else rows,
            }
        },
    }
    if operation:
        body["operation"] = operation
        body["receipt"] = operation.get("receipt")
    return body


def _install_transport(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(adapter.httpx, "AsyncClient", lambda **_kwargs: client)
    return client


@pytest.mark.asyncio
async def test_result_mapping_keeps_exact_bigint_decimal_and_nullability(monkeypatch):
    monkeypatch.setattr(adapter.settings, "sparkless_server_url", "http://sparkless.test")

    def handler(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json=_capabilities())
        return httpx.Response(200, json=_result())

    client = _install_transport(monkeypatch, handler)
    try:
        result = await adapter.execute_statement("warehouse-1", "SELECT 1")
    finally:
        await client.aclose()

    assert result.columns == ["n", "amount"]
    assert result.column_types == ["BIGINT", "DECIMAL(12,3)"]
    assert result.nullable == [False, True]
    assert result.rows == [[9007199254740993, "1.250"]]


@pytest.mark.asyncio
async def test_ambiguous_write_replays_same_identity_after_status_says_unknown(monkeypatch):
    monkeypatch.setattr(adapter.settings, "sparkless_server_url", "http://sparkless.test")
    monkeypatch.setattr(adapter.settings, "sparkless_status_poll_interval_seconds", 0)
    posts = []
    statuses = []

    def handler(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json=_capabilities())
        if request.url.path == "/operations/status":
            statuses.append(request.read())
            return httpx.Response(404, json={"status": "error", "ename": "UnknownOperation", "evalue": "missing"})
        posts.append(request.read())
        if len(posts) == 1:
            raise httpx.ReadTimeout("response lost", request=request)
        operation = {
            "server_instance_id": "instance-1",
            "operation_id": "stable-write-1",
            "execution_state": "succeeded",
            "effect_outcome": "committed",
            "receipt": {"kind": "table", "name": "default.t", "version": 1},
            "error": None,
        }
        return httpx.Response(200, json=_result(rows=[], operation=operation))

    client = _install_transport(monkeypatch, handler)
    try:
        result = await adapter.execute_statement(
            "warehouse-1", "CREATE TABLE t (id BIGINT)", supplied_operation_id="stable-write-1"
        )
    finally:
        await client.aclose()

    assert len(posts) == 2
    assert posts[0] == posts[1]
    assert len(statuses) == 1
    import json

    submitted = json.loads(posts[0])
    assert submitted["server_instance_id"] == "instance-1"
    assert submitted["operation_id"] == "stable-write-1"
    assert result.operation["effect_outcome"] == "committed"
    assert result.receipt["version"] == 1


@pytest.mark.asyncio
async def test_committed_status_recovers_lost_write_response_without_replay(monkeypatch):
    monkeypatch.setattr(adapter.settings, "sparkless_server_url", "http://sparkless.test")
    monkeypatch.setattr(adapter.settings, "sparkless_status_poll_interval_seconds", 0)
    posted = []
    operation = {
        "server_instance_id": "instance-1",
        "operation_id": "commit-then-timeout",
        "execution_state": "succeeded",
        "effect_outcome": "committed",
        "receipt": {"kind": "table", "name": "default.t", "version": 1},
        "error": None,
    }

    def handler(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json=_capabilities())
        if request.url.path == "/statements":
            posted.append(request.read())
            raise httpx.ReadTimeout("response lost after commit", request=request)
        return httpx.Response(200, json=operation)

    client = _install_transport(monkeypatch, handler)
    try:
        result = await adapter.execute_statement("warehouse-1", "INSERT INTO t VALUES (1)")
    finally:
        await client.aclose()

    assert len(posted) == 1
    assert result.rows == []
    assert result.receipt == operation["receipt"]


@pytest.mark.asyncio
async def test_instance_change_during_ambiguous_write_is_never_retried(monkeypatch):
    monkeypatch.setattr(adapter.settings, "sparkless_server_url", "http://sparkless.test")
    monkeypatch.setattr(adapter.settings, "sparkless_status_poll_interval_seconds", 0)
    posted = []

    def handler(request):
        if request.url.path == "/capabilities":
            return httpx.Response(200, json=_capabilities())
        if request.url.path == "/statements":
            posted.append(request.read())
            raise httpx.ReadTimeout("response lost", request=request)
        return httpx.Response(
            409,
            json={"status": "error", "ename": "ServerInstanceMismatch", "evalue": "server restarted"},
        )

    client = _install_transport(monkeypatch, handler)
    try:
        with pytest.raises(DatabricksError, match="instance changed"):
            await adapter.execute_statement("warehouse-1", "INSERT INTO t VALUES (1)")
    finally:
        await client.aclose()

    assert len(posted) == 1


@pytest.mark.asyncio
async def test_incompatible_protocol_fails_before_statement_submission(monkeypatch):
    monkeypatch.setattr(adapter.settings, "sparkless_server_url", "http://sparkless.test")
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=_capabilities(agent="9"))

    client = _install_transport(monkeypatch, handler)
    try:
        with pytest.raises(DatabricksError, match="protocol version 1"):
            await adapter.execute_statement("warehouse-1", "SELECT 1")
    finally:
        await client.aclose()

    assert paths == ["/capabilities"]


@pytest.mark.asyncio
async def test_close_session_uses_the_sparkless_lifecycle_route(monkeypatch):
    monkeypatch.setattr(adapter.settings, "sparkless_server_url", "http://sparkless.test")
    received = []

    def handler(request):
        received.append((request.url.path, request.read()))
        return httpx.Response(200, json={"status": "ok"})

    client = _install_transport(monkeypatch, handler)
    try:
        await adapter.close_session("warehouse-1")
    finally:
        await client.aclose()

    assert received == [("/close", b'{"session":"warehouse-1"}')]
