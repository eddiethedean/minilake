"""DBFS endpoints tests — real file-backed storage.

Note: `DbfsExt.put(contents=...)` sends the string verbatim — the real API
requires `contents` to be base64, and the SDK does not encode it for you
(see its docstring: "Alternatively you can pass contents as base64 string").
So every `put()` call below base64-encodes first, matching real usage.
"""

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

from minilake.errors import DatabricksError as MinilakeDatabricksError
from minilake.models.dbfs import MoveDbfsRequest
from minilake.services import dbfs

SAMPLE = b"minilake dbfs real content\n"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.mark.crud
def test_dbfs_put_and_read(workspace_client: WorkspaceClient):
    """Test: put() writes a real file, read() returns the exact same bytes."""
    path = "/test_put.txt"
    workspace_client.dbfs.put(path, contents=_b64(SAMPLE), overwrite=True)

    status = workspace_client.dbfs.get_status(path)
    assert status.is_dir is False
    assert status.file_size == len(SAMPLE)

    print("✓ DBFS put + get_status works with real files")


@pytest.mark.crud
def test_dbfs_create_add_block_close_roundtrip(workspace_client: WorkspaceClient):
    """Test: chunked upload (create/add-block/close) writes a real file."""
    path = "/test_chunked.bin"
    handle = workspace_client.dbfs.create(path, overwrite=True).handle
    workspace_client.dbfs.add_block(handle, _b64(SAMPLE))
    workspace_client.dbfs.close(handle)

    status = workspace_client.dbfs.get_status(path)
    assert status.file_size == len(SAMPLE)

    print("✓ DBFS chunked upload writes real bytes to disk")


@pytest.mark.crud
def test_dbfs_mkdirs_and_list(workspace_client: WorkspaceClient):
    """Test: mkdirs creates a real directory, list shows real contents."""
    workspace_client.dbfs.mkdirs("/test_dir")
    workspace_client.dbfs.put("/test_dir/file.txt", contents=_b64(b"hi"), overwrite=True)

    entries = list(workspace_client.dbfs.list("/test_dir"))
    names = [e.path for e in entries]
    assert "/test_dir/file.txt" in names

    print("✓ DBFS mkdirs + list works with real directories")


@pytest.mark.crud
def test_dbfs_delete(workspace_client: WorkspaceClient):
    """Test: delete removes a real file."""
    workspace_client.dbfs.put("/test_delete.txt", contents=_b64(b"bye"), overwrite=True)
    workspace_client.dbfs.delete("/test_delete.txt")

    with pytest.raises(Exception):
        workspace_client.dbfs.get_status("/test_delete.txt")

    print("✓ DBFS delete removes real files")


@pytest.mark.crud
def test_dbfs_move(workspace_client: WorkspaceClient):
    """Test: move renames a real file."""
    workspace_client.dbfs.put("/test_move_src.txt", contents=_b64(b"moved"), overwrite=True)
    workspace_client.dbfs.move("/test_move_src.txt", "/test_move_dst.txt")

    status = workspace_client.dbfs.get_status("/test_move_dst.txt")
    assert status.file_size == len(b"moved")

    with pytest.raises(Exception):
        workspace_client.dbfs.get_status("/test_move_src.txt")

    print("✓ DBFS move renames real files")


@pytest.mark.error
def test_dbfs_move_rejects_existing_file_destination(workspace_client: WorkspaceClient):
    """Moving onto a file must preserve both files and report a conflict."""
    workspace_client.dbfs.put("/move_existing_src.txt", contents=_b64(b"new"), overwrite=True)
    workspace_client.dbfs.put("/move_existing_dst.txt", contents=_b64(b"original"), overwrite=True)

    with pytest.raises(DatabricksError) as exc:
        workspace_client.dbfs.move("/move_existing_src.txt", "/move_existing_dst.txt")

    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"
    assert base64.b64decode(workspace_client.dbfs.read("/move_existing_src.txt").data) == b"new"
    assert base64.b64decode(workspace_client.dbfs.read("/move_existing_dst.txt").data) == b"original"


@pytest.mark.error
def test_dbfs_move_rejects_existing_empty_directory_destination(workspace_client: WorkspaceClient):
    """Moving a directory onto an existing empty directory must not replace it."""
    workspace_client.dbfs.mkdirs("/move_dir_src")
    workspace_client.dbfs.mkdirs("/move_dir_dst")
    workspace_client.dbfs.put("/move_dir_src/child.txt", contents=_b64(b"source"), overwrite=True)

    with pytest.raises(DatabricksError) as exc:
        workspace_client.dbfs.move("/move_dir_src", "/move_dir_dst")

    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"
    assert workspace_client.dbfs.get_status("/move_dir_src").is_dir
    assert base64.b64decode(workspace_client.dbfs.read("/move_dir_src/child.txt").data) == b"source"
    assert workspace_client.dbfs.get_status("/move_dir_dst").is_dir
    assert list(workspace_client.dbfs.list("/move_dir_dst")) == []


@pytest.mark.error
def test_dbfs_move_fails_safely_when_exclusive_rename_is_unavailable(tmp_path, monkeypatch):
    """Unsupported filesystems must not fall back to an overwriting rename."""
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_bytes(b"source data")

    monkeypatch.setattr(dbfs.sys, "platform", "linux")
    monkeypatch.setattr(dbfs.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(dbfs, "_resolve", lambda path: source if path == "/source.txt" else destination)

    def unexpected_overwriting_rename(_self: Path, _destination: Path):
        raise AssertionError("ordinary Path.rename must not be used as a fallback")

    monkeypatch.setattr(Path, "rename", unexpected_overwriting_rename)

    with pytest.raises(MinilakeDatabricksError) as exc:
        asyncio.run(dbfs.move_path(MoveDbfsRequest(source_path="/source.txt", destination_path="/destination.txt")))

    assert exc.value.error_code == "NOT_IMPLEMENTED"
    assert exc.value.status_code == 501
    assert source.read_bytes() == b"source data"
    assert not destination.exists()


@pytest.mark.error
def test_dbfs_get_status_nonexistent_fails(workspace_client: WorkspaceClient):
    """Test: get_status on a nonexistent path raises an error."""
    with pytest.raises(Exception):
        workspace_client.dbfs.get_status("/does_not_exist.txt")

    print("✓ DBFS get_status of nonexistent path raises error")
