"""Path resolution regressions shared by DBFS and Files."""

from pathlib import Path

import pytest

from minilake.errors import DatabricksError
from minilake.services import dbfs, files


@pytest.mark.parametrize("service", [dbfs, files], ids=["dbfs", "files"])
@pytest.mark.parametrize("root_kind", ["relative", "absolute", "symlink"], ids=["relative", "absolute", "symlink"])
def test_file_apis_resolve_paths_under_data_root(monkeypatch, tmp_path, service, root_kind):
    if root_kind == "relative":
        data_dir = Path("relative-data")
        expected_data_dir = tmp_path / data_dir
    elif root_kind == "absolute":
        data_dir = tmp_path / "absolute-data"
        expected_data_dir = data_dir
    else:
        expected_data_dir = tmp_path / "real-data"
        expected_data_dir.mkdir()
        data_dir = tmp_path / "linked-data"
        data_dir.symlink_to(expected_data_dir, target_is_directory=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service.settings, "data_dir", data_dir)

    root = expected_data_dir / "dbfs"
    resolved = service._resolve("/hello")
    assert resolved == root.resolve() / "hello"

    with pytest.raises(DatabricksError, match="Invalid path"):
        service._resolve("../../outside")

    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(DatabricksError, match="Invalid path"):
        service._resolve("/escape/secret")
