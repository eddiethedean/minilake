"""Optional, capability-checked adapter from SQL Statements to Sparkless Server.

The normal MiniLake backend remains DuckDB. Setting
``MINILAKE_SPARKLESS_SERVER_URL`` routes Statement Execution requests to a native
Sparkless HTTP server. Every outbound attempt for a statement uses one operation
identity; after an ambiguous write response the adapter consults retained status and
only replays the same identity against the same server instance.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from minilake.config import settings
from minilake.errors import DatabricksError

_AGENT_PROTOCOL_VERSION = "1"
_MUTATION_PROTOCOL_VERSION = "1"
_SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_TYPE_MAP = {
    "string": "STRING",
    "integer": "INT",
    "short": "SMALLINT",
    "long": "BIGINT",
    "double": "DOUBLE",
    "boolean": "BOOLEAN",
    "date": "DATE",
    "timestamp": "TIMESTAMP",
}


@dataclass
class SparklessResult:
    """Typed positional result plus mutation outcome metadata."""

    columns: list[str]
    column_types: list[Optional[str]]
    nullable: list[Optional[bool]]
    rows: list[list[Any]]
    server_instance_id: str
    operation_id: str
    operation: Optional[dict[str, Any]] = None
    receipt: Optional[dict[str, Any]] = None
    pending: bool = False


def configured() -> bool:
    """Whether the optional Sparkless backend has been selected."""
    return bool(settings.sparkless_server_url)


def _base_url() -> str:
    value = settings.sparkless_server_url
    if not value:
        raise DatabricksError("INVALID_STATE", "Sparkless backend is not configured", 500)
    return value.rstrip("/")


def _server_error(response: httpx.Response, body: Any) -> DatabricksError:
    name = body.get("ename") if isinstance(body, dict) else None
    message = body.get("evalue") if isinstance(body, dict) else None
    name = str(name or "SparklessServerError")
    message = str(message or f"Sparkless Server returned HTTP {response.status_code}")
    if response.status_code == 429 or name.endswith("CapacityExceeded") or name == "MutationRetentionCapacityExceeded":
        status = 429
    elif response.status_code == 409 or name in {
        "OperationIdentityConflict",
        "ServerInstanceMismatch",
        "SessionClosing",
    }:
        status = 409
    elif response.status_code in {502, 503, 504} or name == "DeadlineExceeded":
        status = 503 if name == "ServerDraining" else 504
    elif response.status_code >= 500:
        status = response.status_code
    else:
        status = 400
    return DatabricksError(name.upper(), message, status)


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as exc:
        raise DatabricksError("BAD_GATEWAY", "Sparkless Server returned invalid JSON", 502) from exc
    if not isinstance(value, dict):
        raise DatabricksError("BAD_GATEWAY", "Sparkless Server returned a non-object response", 502)
    return value


async def _capabilities(client: httpx.AsyncClient, base_url: str) -> tuple[str, dict[str, Any]]:
    try:
        response = await client.get(f"{base_url}/capabilities")
    except httpx.RequestError as exc:
        raise DatabricksError("SPARKLESS_UNAVAILABLE", f"Could not read Sparkless capabilities: {exc}", 503) from exc
    if response.status_code != 200:
        raise _server_error(response, _json(response))
    body = _json(response)
    if body.get("agent_protocol_version") != _AGENT_PROTOCOL_VERSION:
        raise DatabricksError(
            "SPARKLESS_PROTOCOL_MISMATCH",
            "Sparkless agent protocol version 1 is required",
            502,
        )
    if body.get("mutation_protocol_version") != _MUTATION_PROTOCOL_VERSION:
        raise DatabricksError(
            "SPARKLESS_PROTOCOL_MISMATCH",
            "Sparkless mutation protocol version 1 is required",
            502,
        )
    instance_id = body.get("server_instance_id")
    if not isinstance(instance_id, str) or not _SAFE_ID.fullmatch(instance_id):
        raise DatabricksError("SPARKLESS_PROTOCOL_MISMATCH", "Sparkless capabilities omitted a valid instance ID", 502)
    sql = body.get("sql")
    if not isinstance(sql, dict) or "SELECT" not in sql.get("reads", []):
        raise DatabricksError("SPARKLESS_PROTOCOL_MISMATCH", "Sparkless server does not advertise SQL reads", 502)
    return instance_id, body


def _decode_schema(payload: dict[str, Any]) -> tuple[list[str], list[Optional[str]], list[Optional[bool]]]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise DatabricksError("BAD_GATEWAY", "Sparkless result omitted the data envelope", 502)
    json_data = data.get("application/json")
    if not isinstance(json_data, dict):
        raise DatabricksError("BAD_GATEWAY", "Sparkless result omitted application/json data", 502)
    schema = json_data.get("schema")
    fields = schema.get("fields") if isinstance(schema, dict) else None
    rows = json_data.get("data")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise DatabricksError("BAD_GATEWAY", "Sparkless result has an invalid schema or row array", 502)

    columns: list[str] = []
    column_types: list[Optional[str]] = []
    nullable: list[Optional[bool]] = []
    for field in fields:
        if not isinstance(field, dict) or not isinstance(field.get("name"), str):
            raise DatabricksError("BAD_GATEWAY", "Sparkless result contains an invalid field", 502)
        columns.append(field["name"])
        native_type = field.get("type")
        if isinstance(native_type, str):
            type_text = _TYPE_MAP.get(native_type.lower())
            if type_text is None:
                decimal = re.fullmatch(r"decimal\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", native_type, re.I)
                type_text = f"DECIMAL({decimal.group(1)},{decimal.group(2)})" if decimal else None
        else:
            type_text = None
        if type_text is None:
            raise DatabricksError(
                "UNSUPPORTED_RESULT_TYPE", f"Sparkless result type is unsupported: {native_type!r}", 501
            )
        column_types.append(type_text)
        is_nullable = field.get("nullable")
        nullable.append(is_nullable if isinstance(is_nullable, bool) else None)

    if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
        raise DatabricksError("BAD_GATEWAY", "Sparkless result row width does not match its schema", 502)
    return columns, column_types, nullable


def _operation(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    operation = payload.get("operation")
    return operation if isinstance(operation, dict) else None


def _success_after_status(operation: dict[str, Any]) -> dict[str, Any]:
    """A committed SQL mutation has no row set; construct its empty result envelope."""
    return {
        "status": "ok",
        "data": {"application/json": {"schema": {"type": "struct", "fields": []}, "data": []}},
        "operation": operation,
        "receipt": operation.get("receipt"),
    }


def _raise_execution_error(payload: dict[str, Any], status_code: int = 400) -> None:
    name = str(payload.get("ename") or "SparklessExecutionError")
    message = str(payload.get("evalue") or "Sparkless Server rejected the statement")
    if name in {"OperationInProgress", "DeadlineExceeded"}:
        raise DatabricksError(
            "SPARKLESS_OUTCOME_UNKNOWN",
            f"Sparkless operation outcome is still pending; operation_id={payload.get('operation', {}).get('operation_id', 'unknown')}",
            504,
        )
    if name in {"MutationRetentionCapacityExceeded", "OperationCapacityExceeded", "SessionCapacityExceeded"}:
        status_code = 429
    elif name in {"OperationIdentityConflict", "ServerInstanceMismatch", "SessionClosing"}:
        status_code = 409
    elif name == "DeadlineExceeded":
        status_code = 504
    raise DatabricksError(name.upper(), message, status_code)


async def _operation_status(
    client: httpx.AsyncClient, base_url: str, instance_id: str, operation_id: str
) -> tuple[str, Optional[dict[str, Any]]]:
    try:
        response = await client.post(
            f"{base_url}/operations/status",
            json={"server_instance_id": instance_id, "operation_id": operation_id},
        )
    except httpx.RequestError:
        return "unavailable", None
    if response.status_code == 404:
        return "unknown", None
    body = _json(response)
    if response.status_code != 200:
        if response.status_code == 409:
            raise DatabricksError(
                "SPARKLESS_OUTCOME_UNKNOWN",
                "Sparkless instance changed while a statement outcome was ambiguous; the mutation was not replayed against the new instance",
                503,
            )
        raise _server_error(response, body)
    state = str(body.get("execution_state", "unknown")).lower()
    outcome = str(body.get("effect_outcome", "unknown")).lower()
    if state == "succeeded" and outcome == "committed":
        return "succeeded", body
    if state == "failed" or outcome in {"failed", "cancelled", "unknown"}:
        return "failed", body
    if state in {"running", "pending"} or outcome == "pending":
        return "pending", body
    return "failed", body


async def _recover_accepted_operation(
    client: httpx.AsyncClient,
    base_url: str,
    instance_id: str,
    operation_id: str,
) -> tuple[str, Optional[dict[str, Any]]]:
    polls = max(1, settings.sparkless_status_polls)
    for attempt in range(polls):
        state, body = await _operation_status(client, base_url, instance_id, operation_id)
        if state != "pending" and state != "unavailable":
            return state, body
        if attempt + 1 < polls:
            await asyncio.sleep(min(settings.sparkless_status_poll_interval_seconds, 1.0))
    return "pending", body


async def execute_statement(
    warehouse_id: str,
    sql: str,
    *,
    request_headers: Optional[dict[str, str]] = None,
    supplied_operation_id: Optional[str] = None,
    supplied_server_instance_id: Optional[str] = None,
    catalog: Optional[str] = None,
    schema_name: Optional[str] = None,
    has_parameters: bool = False,
) -> SparklessResult:
    """Execute SQL through native HTTP with capability negotiation and safe retries."""
    if not _SAFE_ID.fullmatch(warehouse_id):
        raise DatabricksError("INVALID_REQUEST", "warehouse_id is not a valid Sparkless session identifier", 400)
    if catalog is not None or schema_name is not None:
        raise DatabricksError(
            "NOT_IMPLEMENTED", "Sparkless SQL does not support Statement API catalog/schema defaults", 501
        )
    if has_parameters:
        raise DatabricksError("NOT_IMPLEMENTED", "Sparkless SQL does not support Statement API parameters", 501)
    leading = sql.lstrip().upper()
    # Sparkless's accepted read grammar starts at SELECT. Treat every other prefix as
    # potentially mutating for outcome recovery; a false positive is safer than a lost
    # commit being retried under a new identity.
    may_mutate = not (leading.startswith("SELECT") or leading.startswith("WITH"))

    headers = {key.lower(): value for key, value in (request_headers or {}).items()}
    operation_id = supplied_operation_id or headers.get("x-sparkless-operation-id") or uuid.uuid4().hex
    if not _SAFE_ID.fullmatch(operation_id):
        raise DatabricksError("INVALID_REQUEST", "Sparkless operation ID must be 1-128 safe ASCII bytes", 400)
    header_instance_id = headers.get("x-sparkless-server-instance-id")
    if supplied_server_instance_id and header_instance_id and supplied_server_instance_id != header_instance_id:
        raise DatabricksError(
            "INVALID_REQUEST", "Sparkless server instance ID fields do not match", 400
        )
    expected_instance_id = supplied_server_instance_id or header_instance_id
    if expected_instance_id and not _SAFE_ID.fullmatch(expected_instance_id):
        raise DatabricksError(
            "INVALID_REQUEST", "Sparkless server instance ID must be 1-128 safe ASCII bytes", 400
        )

    base_url = _base_url()
    try:
        timeout = httpx.Timeout(settings.sparkless_request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            instance_id, _ = await _capabilities(client, base_url)
            if expected_instance_id and expected_instance_id != instance_id:
                raise DatabricksError(
                    "SERVER_INSTANCE_MISMATCH",
                    "Sparkless server instance changed; the operation was not submitted",
                    409,
                )
            body = {
                "session": warehouse_id,
                "kind": "sql",
                "code": sql,
                "server_instance_id": instance_id,
                "operation_id": operation_id,
            }
            max_attempts = max(1, settings.sparkless_post_attempts)
            last_payload: dict[str, Any] = {}
            for attempt in range(max_attempts):
                try:
                    response = await client.post(f"{base_url}/statements", json=body)
                    last_payload = _json(response)
                except httpx.RequestError:
                    state, status = await _recover_accepted_operation(client, base_url, instance_id, operation_id)
                    if state == "succeeded":
                        last_payload = _success_after_status(status or {})
                        break
                    if state == "failed":
                        remote_error = (status or {}).get("error") or {}
                        raise DatabricksError(
                            str(remote_error.get("ename") or "SPARKLESS_MUTATION_FAILED").upper(),
                            str(
                                remote_error.get("evalue") or "Sparkless mutation failed; see retained operation status"
                            ),
                            400,
                        )
                    if state == "pending":
                        operation = status or {
                            "server_instance_id": instance_id,
                            "operation_id": operation_id,
                            "execution_state": "running",
                            "effect_outcome": "pending",
                        }
                        return SparklessResult([], [], [], [], instance_id, operation_id, operation, None, pending=True)
                    if state == "unknown" and attempt + 1 < max_attempts:
                        continue
                    if may_mutate:
                        return SparklessResult(
                            [],
                            [],
                            [],
                            [],
                            instance_id,
                            operation_id,
                            status
                            or {
                                "server_instance_id": instance_id,
                                "operation_id": operation_id,
                                "execution_state": "running",
                                "effect_outcome": "pending",
                            },
                            None,
                            pending=True,
                        )
                    raise DatabricksError(
                        "SPARKLESS_UNAVAILABLE",
                        "Sparkless read request failed before a result was available",
                        503,
                    )

                if response.status_code >= 500:
                    state, status = await _recover_accepted_operation(client, base_url, instance_id, operation_id)
                    if state == "succeeded":
                        last_payload = _success_after_status(status or {})
                        break
                    if state == "pending":
                        operation = status or {
                            "server_instance_id": instance_id,
                            "operation_id": operation_id,
                            "execution_state": "running",
                            "effect_outcome": "pending",
                        }
                        return SparklessResult([], [], [], [], instance_id, operation_id, operation, None, pending=True)
                    if state == "failed":
                        remote_error = (status or {}).get("error") or {}
                        raise DatabricksError(
                            str(remote_error.get("ename") or "SPARKLESS_MUTATION_FAILED").upper(),
                            str(
                                remote_error.get("evalue") or "Sparkless mutation failed; see retained operation status"
                            ),
                            400,
                        )
                    if state == "unknown" and attempt + 1 < max_attempts:
                        continue
                    if state == "unknown" and may_mutate:
                        return SparklessResult(
                            [],
                            [],
                            [],
                            [],
                            instance_id,
                            operation_id,
                            {
                                "server_instance_id": instance_id,
                                "operation_id": operation_id,
                                "execution_state": "running",
                                "effect_outcome": "pending",
                            },
                            None,
                            pending=True,
                        )
                    raise _server_error(response, last_payload)

                if last_payload.get("status") == "error":
                    error_name = last_payload.get("ename")
                    if error_name in {"OperationInProgress", "DeadlineExceeded"}:
                        state, status = await _recover_accepted_operation(client, base_url, instance_id, operation_id)
                        if state == "succeeded":
                            last_payload = _success_after_status(status or {})
                            break
                        if state == "pending":
                            operation = (
                                status
                                or _operation(last_payload)
                                or {
                                    "server_instance_id": instance_id,
                                    "operation_id": operation_id,
                                    "execution_state": "running",
                                    "effect_outcome": "pending",
                                }
                            )
                            return SparklessResult(
                                [], [], [], [], instance_id, operation_id, operation, None, pending=True
                            )
                        if state == "unknown" and attempt + 1 < max_attempts:
                            continue
                        if state == "unknown" and may_mutate:
                            return SparklessResult(
                                [],
                                [],
                                [],
                                [],
                                instance_id,
                                operation_id,
                                _operation(last_payload)
                                or {
                                    "server_instance_id": instance_id,
                                    "operation_id": operation_id,
                                    "execution_state": "running",
                                    "effect_outcome": "pending",
                                },
                                None,
                                pending=True,
                            )
                    _raise_execution_error(last_payload, response.status_code)
                if response.status_code >= 400:
                    raise _server_error(response, last_payload)
                break
            else:
                raise DatabricksError("SPARKLESS_OUTCOME_UNKNOWN", "Sparkless statement retry limit was exhausted", 503)

            if last_payload.get("status") == "error":
                _raise_execution_error(last_payload)
            columns, column_types, nullable = _decode_schema(last_payload)
            nested = _operation(last_payload)
            receipt = last_payload.get("receipt")
            if not isinstance(receipt, dict) and nested:
                receipt = nested.get("receipt")
            return SparklessResult(
                columns,
                column_types,
                nullable,
                last_payload["data"]["application/json"]["data"],
                instance_id,
                operation_id,
                nested,
                receipt if isinstance(receipt, dict) else None,
            )
    except DatabricksError:
        raise
    except httpx.RequestError as exc:
        raise DatabricksError("SPARKLESS_UNAVAILABLE", f"Sparkless backend request failed: {exc}", 503) from exc


async def get_operation_status(instance_id: str, operation_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    """Read retained operation state without creating or replaying an operation."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.sparkless_request_timeout_seconds)) as client:
            return await _operation_status(client, _base_url(), instance_id, operation_id)
    except httpx.RequestError:
        return "unavailable", None


async def close_session(session_id: str) -> None:
    """Close the Sparkless session owned by a deleted MiniLake warehouse."""
    if not _SAFE_ID.fullmatch(session_id):
        raise DatabricksError("INVALID_REQUEST", "warehouse_id is not a valid Sparkless session identifier", 400)
    base_url = _base_url()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.sparkless_request_timeout_seconds)) as client:
            response = await client.post(f"{base_url}/close", json={"session": session_id})
    except httpx.RequestError as exc:
        raise DatabricksError("SPARKLESS_UNAVAILABLE", f"Could not close Sparkless session: {exc}", 503) from exc
    if response.status_code != 200:
        raise _server_error(response, _json(response))
