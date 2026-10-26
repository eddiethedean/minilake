"""Cascade deletion tests for Unity Catalog schemas and catalogs."""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.catalog import (
    ColumnTypeName,
    CreateFunction,
    CreateFunctionParameterStyle,
    CreateFunctionRoutineBody,
    CreateFunctionSecurityType,
    CreateFunctionSqlDataAccess,
    FunctionParameterInfo,
    FunctionParameterInfos,
    PermissionsChange,
    Privilege,
    SecurableType,
    VolumeType,
)

from minilake.config import settings


def _sql_function(name: str, catalog_name: str, schema_name: str) -> CreateFunction:
    return CreateFunction(
        name=name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        input_params=FunctionParameterInfos(
            parameters=[
                FunctionParameterInfo(
                    name="x",
                    type_text="INT",
                    type_name=ColumnTypeName.INT,
                    position=0,
                )
            ]
        ),
        data_type=ColumnTypeName.INT,
        full_data_type="INT",
        routine_body=CreateFunctionRoutineBody.SQL,
        routine_definition="x + 1",
        parameter_style=CreateFunctionParameterStyle.S,
        is_deterministic=True,
        sql_data_access=CreateFunctionSqlDataAccess.NO_SQL,
        is_null_call=False,
        security_type=CreateFunctionSecurityType.DEFINER,
        specific_name=name,
    )


def _grant(workspace_client: WorkspaceClient, securable_type: SecurableType, full_name: str, privilege: Privilege):
    workspace_client.grants.update(
        securable_type=securable_type,
        full_name=full_name,
        changes=[PermissionsChange(principal="cascade-reader", add=[privilege])],
    )


@pytest.mark.serial
@pytest.mark.workflow
def test_delete_schema_cascades_functions_volumes_and_grants(catalog_and_schema, workspace_client: WorkspaceClient):
    catalog, schema = catalog_and_schema
    function_name = "cascade_fn"
    volume_name = "cascade_vol"
    function_full_name = f"{catalog.name}.{schema.name}.{function_name}"
    volume_full_name = f"{catalog.name}.{schema.name}.{volume_name}"
    volume_path = Path(settings.data_dir) / "volumes" / catalog.name / schema.name / volume_name

    workspace_client.functions.create(function_info=_sql_function(function_name, catalog.name, schema.name))
    workspace_client.volumes.create(
        name=volume_name,
        catalog_name=catalog.name,
        schema_name=schema.name,
        volume_type=VolumeType.MANAGED,
    )
    volume_path.joinpath("marker.txt").write_text("managed volume data")

    _grant(workspace_client, SecurableType.SCHEMA, schema.full_name, Privilege.USE_SCHEMA)
    _grant(workspace_client, SecurableType.FUNCTION, function_full_name, Privilege.EXECUTE)
    _grant(workspace_client, SecurableType.VOLUME, volume_full_name, Privilege.READ_VOLUME)
    _grant(workspace_client, SecurableType.CATALOG, catalog.name, Privilege.USE_CATALOG)

    workspace_client.schemas.delete(full_name=schema.full_name, force=True)

    with pytest.raises(DatabricksError):
        workspace_client.functions.get(name=function_full_name)
    assert not list(workspace_client.functions.list(catalog_name=catalog.name, schema_name=schema.name))
    with pytest.raises(DatabricksError):
        workspace_client.volumes.read(name=volume_full_name)
    assert not list(workspace_client.volumes.list(catalog_name=catalog.name, schema_name=schema.name))
    assert not volume_path.exists()

    for securable_type, full_name in (
        (SecurableType.SCHEMA, schema.full_name),
        (SecurableType.FUNCTION, function_full_name),
        (SecurableType.VOLUME, volume_full_name),
    ):
        permissions = workspace_client.grants.get(securable_type=securable_type, full_name=full_name)
        assert not permissions.privilege_assignments

    # A directory left by an older cleanup must not be adopted without its schema.
    volume_path.mkdir(parents=True)
    volume_path.joinpath("orphan.txt").write_text("orphan")
    try:
        with pytest.raises(DatabricksError):
            workspace_client.volumes.read(name=volume_full_name)
        assert not list(workspace_client.volumes.list(catalog_name=catalog.name, schema_name=schema.name))
    finally:
        shutil.rmtree(volume_path.parent, ignore_errors=True)

    assert workspace_client.grants.get(
        securable_type=SecurableType.CATALOG, full_name=catalog.name
    ).privilege_assignments

    recreated = workspace_client.schemas.create(name=schema.name, catalog_name=catalog.name)
    workspace_client.functions.create(function_info=_sql_function(function_name, catalog.name, recreated.name))
    workspace_client.volumes.create(
        name=volume_name,
        catalog_name=catalog.name,
        schema_name=recreated.name,
        volume_type=VolumeType.MANAGED,
    )


@pytest.mark.serial
@pytest.mark.workflow
def test_delete_catalog_cascades_functions_volumes_and_grants(workspace_client: WorkspaceClient):
    catalog_name = f"cascade_cat_{uuid4().hex[:8]}"
    schema_name = "default"
    function_name = "cascade_fn"
    volume_name = "cascade_vol"
    function_full_name = f"{catalog_name}.{schema_name}.{function_name}"
    volume_full_name = f"{catalog_name}.{schema_name}.{volume_name}"
    volume_path = Path(settings.data_dir) / "volumes" / catalog_name / schema_name / volume_name

    workspace_client.catalogs.create(name=catalog_name)
    workspace_client.functions.create(function_info=_sql_function(function_name, catalog_name, schema_name))
    workspace_client.volumes.create(
        name=volume_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        volume_type=VolumeType.MANAGED,
    )
    volume_path.joinpath("marker.txt").write_text("managed volume data")

    _grant(workspace_client, SecurableType.CATALOG, catalog_name, Privilege.USE_CATALOG)
    _grant(workspace_client, SecurableType.FUNCTION, function_full_name, Privilege.EXECUTE)
    _grant(workspace_client, SecurableType.VOLUME, volume_full_name, Privilege.READ_VOLUME)
    _grant(workspace_client, SecurableType.STORAGE_CREDENTIAL, catalog_name, Privilege.READ_FILES)
    _grant(workspace_client, SecurableType.EXTERNAL_LOCATION, catalog_name, Privilege.READ_FILES)

    workspace_client.catalogs.delete(name=catalog_name, force=True)

    with pytest.raises(DatabricksError):
        workspace_client.functions.get(name=function_full_name)
    assert not list(workspace_client.functions.list(catalog_name=catalog_name, schema_name=schema_name))
    with pytest.raises(DatabricksError):
        workspace_client.volumes.read(name=volume_full_name)
    assert not list(workspace_client.volumes.list(catalog_name=catalog_name, schema_name=schema_name))
    assert not volume_path.exists()

    for securable_type, full_name in (
        (SecurableType.CATALOG, catalog_name),
        (SecurableType.FUNCTION, function_full_name),
        (SecurableType.VOLUME, volume_full_name),
    ):
        permissions = workspace_client.grants.get(securable_type=securable_type, full_name=full_name)
        assert not permissions.privilege_assignments

    for securable_type in (SecurableType.STORAGE_CREDENTIAL, SecurableType.EXTERNAL_LOCATION):
        permissions = workspace_client.grants.get(securable_type=securable_type, full_name=catalog_name)
        assert permissions.privilege_assignments

    workspace_client.catalogs.create(name=catalog_name)
    workspace_client.functions.create(function_info=_sql_function(function_name, catalog_name, schema_name))
    workspace_client.volumes.create(
        name=volume_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        volume_type=VolumeType.MANAGED,
    )
