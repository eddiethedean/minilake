"""workspace-files endpoints — raw-bytes file sync used by `databricks bundle deploy`.

The Databricks CLI's WSFS filer writes with `POST /api/2.0/workspace-files/import-file/{path}`
and reads back with `GET /api/2.0/workspace-files/{path}`, both moving raw bytes in
the HTTP body. The Python SDK has no typed method for these, so we drive them through
the SDK's low-level `api_client` (still the real SDK client pointed at minilake).

This exact flow is what `databricks bundle deploy` uses to sync a bundle's files and
to pull/push its Terraform deployment state (`state/deployment.json`, etc.).
"""

import pytest
from databricks.sdk import WorkspaceClient

BASE = "/Workspace/Users/minilake-user/.bundle/databricks/dev/files"
STUB = b"# type: ignore\nreveal_type = ...\n"


def _import_file(client: WorkspaceClient, path: str, content: bytes, overwrite: bool = True):
    """POST raw bytes to workspace-files/import-file (WSFS filer write)."""
    return client.api_client.do(
        "POST",
        f"/api/2.0/workspace-files/import-file{path}",
        query={"overwrite": "true" if overwrite else "false"},
        headers={"Content-Type": "application/octet-stream"},
        data=content,
    )


def _read_file(client: WorkspaceClient, path: str) -> bytes:
    """GET raw bytes from workspace-files/{path} (WSFS filer read)."""
    resp = client.api_client.do("GET", f"/api/2.0/workspace-files{path}", raw=True)
    return resp["contents"].read()


@pytest.mark.crud
def test_import_export_file_round_trips_raw_bytes(workspace_client: WorkspaceClient):
    """Import an arbitrary file (.pyi), export it, and get identical bytes back."""
    path = f"{BASE}/typings/__builtins__.pyi"
    _import_file(workspace_client, path, STUB)

    assert _read_file(workspace_client, path) == STUB

    print("✓ workspace-files import/export round-trips raw bytes")


@pytest.mark.crud
def test_imported_file_reports_object_type_file(workspace_client: WorkspaceClient):
    """A file synced via import-file is a FILE (not a NOTEBOOK) in get_status."""
    path = f"{BASE}/artifact.whl"
    _import_file(workspace_client, path, b"PK\x03\x04 not really a wheel")

    status = workspace_client.workspace.get_status(path=path)
    assert status.object_type.value == "FILE"
    assert status.language is None

    print("✓ synced file reports object_type=FILE")


@pytest.mark.workflow
def test_workspace_file_deleted_directory_can_be_reused(workspace_client: WorkspaceClient):
    path = "/Shared/replaced_directory.txt"
    content = b"file after directory deletion"
    workspace_client.workspace.mkdirs(path=path)
    old_directory = workspace_client.workspace.get_status(path=path)
    workspace_client.workspace.delete(path=path, recursive=True)

    _import_file(workspace_client, path, content)

    status = workspace_client.workspace.get_status(path=path)
    assert status.object_type.value == "FILE"
    assert status.object_id != old_directory.object_id
    assert status.created_at is not None
    assert _read_file(workspace_client, path) == content


@pytest.mark.error
def test_workspace_file_import_does_not_replace_directory(workspace_client: WorkspaceClient):
    path = "/Shared/not_a_file_directory"
    workspace_client.workspace.mkdirs(path=path)
    before = workspace_client.workspace.get_status(path=path)

    with pytest.raises(Exception):
        _import_file(workspace_client, path, b"file content")

    after = workspace_client.workspace.get_status(path=path)
    assert after.object_type.value == "DIRECTORY"
    assert after.object_id == before.object_id
    assert list(workspace_client.workspace.list(path=path)) == []


@pytest.mark.error
def test_import_file_overwrite_false_conflicts(workspace_client: WorkspaceClient):
    """Re-importing without overwrite raises (RESOURCE_ALREADY_EXISTS / 400)."""
    path = f"{BASE}/state/terraform.tfstate"
    _import_file(workspace_client, path, b'{"version": 4}')

    with pytest.raises(Exception):
        _import_file(workspace_client, path, b'{"version": 5}', overwrite=False)

    # Original content is untouched.
    assert _read_file(workspace_client, path) == b'{"version": 4}'

    print("✓ import-file honors overwrite=false")


@pytest.mark.error
def test_export_missing_file_raises(workspace_client: WorkspaceClient):
    """Exporting a path that was never imported raises (404)."""
    with pytest.raises(Exception):
        _read_file(workspace_client, f"{BASE}/does_not_exist.txt")

    print("✓ export-file of missing path raises")
