"""Volume CRUD and error handling tests."""

from uuid import uuid4

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.catalog import VolumeType


@pytest.mark.crud
def test_volume_create_returns_name(catalog, workspace_client: WorkspaceClient):
    """Test: Create volume returns volume with correct name."""
    vol_name = f"test_vol_{uuid4().hex[:6]}"

    volume = workspace_client.volumes.create(
        name=vol_name,
        catalog_name=catalog.name,
        schema_name="default",
        volume_type=VolumeType.MANAGED,
    )

    assert volume.name == vol_name
    assert volume.catalog_name == catalog.name

    # Cleanup
    workspace_client.volumes.delete(name=f"{catalog.name}.default.{vol_name}")

    print(f"✓ Volume created: {volume.name}")


@pytest.mark.crud
def test_volume_list_by_schema(catalog, workspace_client: WorkspaceClient):
    """Test: List volumes by schema returns all volumes."""
    vol_name = f"test_vol_{uuid4().hex[:6]}"

    workspace_client.volumes.create(
        name=vol_name,
        catalog_name=catalog.name,
        schema_name="default",
        volume_type=VolumeType.MANAGED,
    )

    volumes = list(workspace_client.volumes.list(catalog_name=catalog.name, schema_name="default"))
    names = [v.name for v in volumes]

    assert vol_name in names

    # Cleanup
    workspace_client.volumes.delete(name=f"{catalog.name}.default.{vol_name}")

    print(f"✓ Volume in schema list: {vol_name}")


@pytest.mark.error
def test_volume_create_under_nonexistent_schema_fails(catalog, workspace_client: WorkspaceClient):
    """Test: Create volume under nonexistent schema raises error."""
    with pytest.raises(Exception):
        workspace_client.volumes.create(
            name="test_vol",
            catalog_name=catalog.name,
            schema_name="nonexistent_schema",
            volume_type=VolumeType.MANAGED,
        )

    print("✓ Volume create under nonexistent schema raises error")


@pytest.mark.serial
@pytest.mark.workflow
def test_schema_delete_requires_force_for_volume(catalog_and_schema, workspace_client: WorkspaceClient):
    cat, schema = catalog_and_schema
    volume_name = f"preserve_{uuid4().hex[:8]}"
    full_name = f"{cat.name}.{schema.name}.{volume_name}"
    workspace_client.volumes.create(
        name=volume_name,
        catalog_name=cat.name,
        schema_name=schema.name,
        volume_type=VolumeType.MANAGED,
    )

    try:
        with pytest.raises(DatabricksError, match="not empty"):
            workspace_client.schemas.delete(full_name=schema.full_name, force=False)

        assert workspace_client.volumes.read(name=full_name).name == volume_name

        workspace_client.schemas.delete(full_name=schema.full_name, force=True)
        with pytest.raises(DatabricksError):
            workspace_client.volumes.read(name=full_name)
    finally:
        try:
            workspace_client.volumes.delete(name=full_name)
        except Exception:
            pass
        try:
            workspace_client.schemas.delete(full_name=schema.full_name, force=True)
        except Exception:
            pass
