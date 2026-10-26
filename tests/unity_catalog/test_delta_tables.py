"""EXTERNAL Delta table tests — real Delta Lake files read via DuckDB's delta_scan().

Data is written here with the lightweight `deltalake` (delta-rs) package rather
than Spark, to keep this test fast and independent of Maven/JVM in CI. It writes
a genuinely spec-compliant Delta table (parquet + _delta_log), so it validates
exactly the same read path a real Spark job or the notebook service would use.
The test process shares the same `/data` volume as the minilake server (see
docker-compose.test.yml), so files written here are immediately visible to it.
"""

import os
from uuid import uuid4

import pandas as pd
import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import BadRequest, DatabricksError
from databricks.sdk.service.catalog import ColumnInfo, DataSourceFormat, TableType
from deltalake import DeltaTable, write_deltalake


@pytest.fixture
def delta_catalog_and_schema(workspace_client: WorkspaceClient):
    cat_name = f"delta_cat_{uuid4().hex[:6]}"
    schema_name = f"delta_schema_{uuid4().hex[:6]}"
    cat = workspace_client.catalogs.create(name=cat_name)
    schema = workspace_client.schemas.create(name=schema_name, catalog_name=cat_name)
    yield cat, schema
    try:
        workspace_client.schemas.delete(full_name=f"{cat_name}.{schema_name}")
    except Exception:
        pass
    try:
        workspace_client.catalogs.delete(name=cat_name)
    except Exception:
        pass


@pytest.mark.workflow
def test_external_delta_columns_come_from_the_delta_log(delta_catalog_and_schema, workspace_client: WorkspaceClient):
    """A declared schema that disagrees with the files must lose to the files.

    Declaring INTEGER over a Delta table whose log says `long` is the exact mistake that
    prompted this: it was accepted silently, and GET reported no columns at all, so
    nothing ever contradicted it.
    """
    cat, schema = delta_catalog_and_schema
    table_name = f"drift_{uuid4().hex[:6]}"
    storage_location = f"/data/delta/{cat.name}/{schema.name}/{table_name}"

    # int64 in pandas -> `long` in the Delta log.
    write_deltalake(
        storage_location,
        pd.DataFrame({"id": [1, 2], "nome": ["a", "b"]}),
        mode="overwrite",
    )

    table = workspace_client.tables.create(
        name=table_name,
        catalog_name=cat.name,
        schema_name=schema.name,
        table_type=TableType.EXTERNAL,
        data_source_format=DataSourceFormat.DELTA,
        storage_location=storage_location,
        columns=[
            ColumnInfo(name="id", type_text="INTEGER"),  # wrong on purpose
            ColumnInfo(name="nome", type_text="STRING"),
        ],
    )

    fetched = workspace_client.tables.get(full_name=table.full_name)
    by_name = {c.name: c for c in fetched.columns}

    assert by_name["id"].type_name.value == "LONG", "the Delta log, not the declaration"
    assert by_name["nome"].type_name.value == "STRING"


@pytest.mark.workflow
def test_create_table_using_delta_location_registers_external(
    delta_catalog_and_schema, workspace_client: WorkspaceClient
):
    """`CREATE TABLE ... USING DELTA LOCATION` is the native Databricks spelling.

    It used to be regex-stripped down to a plain DuckDB table at no location that UC never
    heard of — and reported success.
    """
    cat, schema = delta_catalog_and_schema
    table_name = f"ddl_{uuid4().hex[:6]}"
    storage_location = f"/data/delta/{cat.name}/{schema.name}/{table_name}"

    warehouse = workspace_client.warehouses.create(name=f"ddl_wh_{uuid4().hex[:6]}")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=(
            f"CREATE TABLE {cat.name}.{schema.name}.{table_name} "
            f"(id BIGINT, nome STRING) USING DELTA LOCATION '{storage_location}'"
        ),
    )
    assert result.status.state.value == "SUCCEEDED"

    table = workspace_client.tables.get(full_name=f"{cat.name}.{schema.name}.{table_name}")
    assert table.table_type.value == "EXTERNAL"
    assert table.data_source_format.value == "DELTA"
    assert table.storage_location == storage_location

    # And it is immediately queryable by three-part name once the files exist.
    write_deltalake(storage_location, pd.DataFrame({"id": [7], "nome": ["x"]}), mode="overwrite")
    rows = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT nome FROM {cat.name}.{schema.name}.{table_name}",
    )
    assert rows.result.data_array == [["x"]]


@pytest.mark.workflow
def test_create_table_using_delta_location_with_comment_registers_external(
    delta_catalog_and_schema, workspace_client: WorkspaceClient
):
    """A comment before USING DELTA must not turn an external table into a managed one."""
    cat, schema = delta_catalog_and_schema
    table_name = f"ddl_comment_{uuid4().hex[:6]}"
    storage_location = f"/data/delta/{cat.name}/{schema.name}/{table_name}"

    write_deltalake(storage_location, pd.DataFrame({"id": [7, 8]}), mode="overwrite")

    warehouse = workspace_client.warehouses.create(name=f"ddl_comment_wh_{uuid4().hex[:6]}")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=(
            f"CREATE TABLE {cat.name}./* name separator */{schema.name}.{table_name} (id BIGINT) "
            f"/* format metadata */ USING DELTA LOCATION '{storage_location}'"
        ),
    )
    assert result.status.state.value == "SUCCEEDED"

    table = workspace_client.tables.get(full_name=f"{cat.name}.{schema.name}.{table_name}")
    assert table.table_type.value == "EXTERNAL"
    assert table.storage_location == storage_location

    rows = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT id FROM {cat.name}.{schema.name}.{table_name} ORDER BY id",
    )
    assert rows.status.state.value == "SUCCEEDED"
    assert rows.result.data_array == [[7], [8]]


@pytest.mark.workflow
def test_unsupported_replace_external_ddl_preserves_existing_table(
    delta_catalog_and_schema, workspace_client: WorkspaceClient
):
    """An unclassified external-table declaration must not erase a managed table."""
    cat, schema = delta_catalog_and_schema
    table_name = f"replace_guard_{uuid4().hex[:6]}"
    storage_location = f"/data/delta/{cat.name}/{schema.name}/replacement_source"
    write_deltalake(storage_location, pd.DataFrame({"id": [7, 8]}), mode="overwrite")

    warehouse = workspace_client.warehouses.create(name=f"replace_guard_wh_{uuid4().hex[:6]}")
    create = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"CREATE TABLE {cat.name}.{schema.name}.{table_name} AS SELECT 99::BIGINT AS id",
    )
    assert create.status.state.value == "SUCCEEDED"

    with pytest.raises(BadRequest, match='syntax error at or near "USING"'):
        workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=(
                f"CREATE OR REPLACE TABLE {cat.name}.{schema.name}.{table_name} (id BIGINT) "
                f"/* format metadata */ USING DELTA LOCATION '{storage_location}'"
            ),
        )

    rows = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT id FROM {cat.name}.{schema.name}.{table_name}",
    )
    assert rows.status.state.value == "SUCCEEDED"
    assert rows.result.data_array == [[99]]

    table = workspace_client.tables.get(full_name=f"{cat.name}.{schema.name}.{table_name}")
    assert table.table_type.value == "MANAGED"
    assert table.storage_location is None


@pytest.mark.workflow
def test_external_delta_table_read_via_sql(delta_catalog_and_schema, workspace_client: WorkspaceClient):
    """Register an EXTERNAL Delta table, write real Delta data, query it via SQL."""
    cat, schema = delta_catalog_and_schema
    table_name = f"people_{uuid4().hex[:6]}"
    storage_location = f"{os.environ.get('MINILAKE_DATA_DIR', '/data')}/delta/{cat.name}/{schema.name}/{table_name}"

    table = workspace_client.tables.create(
        name=table_name,
        catalog_name=cat.name,
        schema_name=schema.name,
        table_type=TableType.EXTERNAL,
        data_source_format=DataSourceFormat.DELTA,
        storage_location=storage_location,
    )
    assert table.table_type.value == "EXTERNAL"
    assert table.data_source_format.value == "DELTA"

    df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    write_deltalake(storage_location, df, mode="overwrite")

    warehouse = workspace_client.warehouses.create(name=f"delta_wh_{uuid4().hex[:6]}")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT * FROM {cat.name}.{schema.name}.{table_name} ORDER BY id",
    )

    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == [[1, "Alice"], [2, "Bob"], [3, "Charlie"]]

    # Exercise relation forms through the HTTP SQL API with actual Delta files.
    for statement, expected in (
        (f"SELECT '{table.full_name}', $${table.full_name}$$", [[table.full_name, table.full_name]]),
        (f"WITH x AS (FROM {table.full_name}) SELECT count(*) FROM x", [[3]]),
        (f"WITH values AS (SELECT 1) FROM {table.full_name} SELECT count(*)", [[3]]),
        (f"WITH x AS (WITH y AS (SELECT 1) FROM {table.full_name}) SELECT count(*) FROM x", [[3]]),
        (f"SELECT (FROM {table.full_name} SELECT max(id))", [[3]]),
        (f"SELECT (WITH y AS (SELECT 1) FROM {table.full_name} SELECT max(id))", [[3]]),
        (f"SELECT count(*) FROM (SELECT 1 AS n) AS values, {table.full_name}", [[3]]),
        (f"SELECT count(*) FROM (SELECT 1 AS n) values, {table.full_name}", [[3]]),
        (f"SELECT count(*) FROM (PIVOT {table.full_name} ON name USING sum(id)) p", [[1]]),
        (f"SELECT count(*) FROM (PIVOT_WIDER {table.full_name} ON name USING sum(id)) p", [[1]]),
        (f"SELECT count(*) FROM (UNPIVOT {table.full_name} ON id INTO NAME field VALUE amount) p", [[3]]),
        (
            f"SELECT count(*) FROM (SELECT 0 AS Alice, 0 AS Bob, 0 AS Charlie "
            f"UNION ALL PIVOT {table.full_name} ON name USING sum(id)) p",
            [[2]],
        ),
        (
            f"SELECT count(*) FROM (SELECT 0 AS Alice, 0 AS Bob, 0 AS Charlie "
            f"INTERSECT PIVOT {table.full_name} ON name USING sum(id)) p",
            [[0]],
        ),
        (
            f"SELECT count(*) FROM (SELECT 0 AS Alice, 0 AS Bob, 0 AS Charlie "
            f"EXCEPT PIVOT {table.full_name} ON name USING sum(id)) p",
            [[1]],
        ),
        (
            f"SELECT count(*) FROM (SELECT 'Alice' AS name, 'id' AS field, 0 AS amount "
            f"UNION ALL UNPIVOT {table.full_name} ON id INTO NAME field VALUE amount) p",
            [[4]],
        ),
    ):
        response = workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id, statement=statement
        )
        assert response.status.state.value == "SUCCEEDED", (statement, response.status)
        assert response.result.data_array == expected

    def execute(statement):
        response = workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id, statement=statement
        )
        assert response.status.state.value == "SUCCEEDED", (statement, response.status)
        return response

    # VALUES is also a valid unquoted table name, and USING introduces a
    # relation in DELETE and MERGE (while JOIN USING names columns).
    execute(f"CREATE TEMP TABLE values AS FROM {table.full_name}")
    try:
        assert execute("SELECT count(*) FROM values").result.data_array == [[3]]
    finally:
        execute("DROP TABLE values")

    target = f"target_{uuid4().hex[:8]}"
    execute(f"CREATE TEMP TABLE {target} AS SELECT 1 AS id")
    try:
        execute(f"DELETE FROM {target} USING {table.full_name} AS source WHERE {target}.id = source.id")
        assert execute(f"SELECT count(*) FROM {target}").result.data_array == [[0]]
        version = execute("SELECT version()").result.data_array[0][0]
        if tuple(map(int, version.lstrip("v").split(".")[:2])) >= (1, 4):
            execute(f"INSERT INTO {target} VALUES (1)")
            execute(
                f"MERGE INTO {target} USING {table.full_name} AS source ON {target}.id = source.id "
                "WHEN MATCHED THEN UPDATE SET id = source.id + 10"
            )
            assert execute(f"SELECT id FROM {target}").result.data_array == [[11]]
    finally:
        execute(f"DROP TABLE {target}")

    print(f"✓ Real Delta table written externally and read via minilake SQL: {table.full_name}")


@pytest.mark.workflow
def test_external_delta_table_reflects_overwrites(delta_catalog_and_schema, workspace_client: WorkspaceClient):
    """A second real Delta write (overwrite) is immediately visible via SQL — no caching."""
    cat, schema = delta_catalog_and_schema
    table_name = f"counts_{uuid4().hex[:6]}"
    storage_location = f"/data/delta/{cat.name}/{schema.name}/{table_name}"

    workspace_client.tables.create(
        name=table_name,
        catalog_name=cat.name,
        schema_name=schema.name,
        table_type=TableType.EXTERNAL,
        data_source_format=DataSourceFormat.DELTA,
        storage_location=storage_location,
    )

    write_deltalake(storage_location, pd.DataFrame({"n": [1]}), mode="overwrite")

    warehouse = workspace_client.warehouses.create(name=f"delta_wh2_{uuid4().hex[:6]}")
    statement = f"SELECT * FROM {cat.name}.{schema.name}.{table_name}"

    first = workspace_client.statement_execution.execute_statement(warehouse_id=warehouse.id, statement=statement)
    assert first.result.data_array == [[1]]

    write_deltalake(storage_location, pd.DataFrame({"n": [1, 2, 3]}), mode="overwrite")

    second = workspace_client.statement_execution.execute_statement(warehouse_id=warehouse.id, statement=statement)
    assert sorted(row[0] for row in second.result.data_array) == [1, 2, 3]

    print("✓ Delta table overwrite is immediately visible through minilake SQL")


def _make_external_table(workspace_client, cat, schema, table_name, seed_df):
    storage_location = f"/data/delta/{cat.name}/{schema.name}/{table_name}"
    workspace_client.tables.create(
        name=table_name,
        catalog_name=cat.name,
        schema_name=schema.name,
        table_type=TableType.EXTERNAL,
        data_source_format=DataSourceFormat.DELTA,
        storage_location=storage_location,
    )
    write_deltalake(storage_location, seed_df, mode="overwrite")
    return storage_location


@pytest.mark.serial
@pytest.mark.workflow
def test_insert_into_external_delta_table_writes_for_real(delta_catalog_and_schema, workspace_client: WorkspaceClient):
    """INSERT INTO an EXTERNAL Delta table triggers a real Spark write (no explicit columns)."""
    cat, schema = delta_catalog_and_schema
    table_name = f"people_{uuid4().hex[:6]}"
    _make_external_table(
        workspace_client, cat, schema, table_name, pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    )

    warehouse = workspace_client.warehouses.create(name=f"insert_wh_{uuid4().hex[:6]}")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"INSERT INTO {cat.name}.{schema.name}.{table_name} VALUES (3, 'Charlie')",
    )
    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == [[1]]  # num_affected_rows

    select = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT * FROM {cat.name}.{schema.name}.{table_name} ORDER BY id",
    )
    assert select.result.data_array == [[1, "Alice"], [2, "Bob"], [3, "Charlie"]]

    print(f"✓ Real INSERT via Spark landed in Delta table: {table_name}")


@pytest.mark.serial
@pytest.mark.workflow
def test_insert_into_external_delta_table_with_explicit_columns(
    delta_catalog_and_schema, workspace_client: WorkspaceClient
):
    """INSERT INTO table (col, col) VALUES (...) works even with reordered columns."""
    cat, schema = delta_catalog_and_schema
    table_name = f"people_cols_{uuid4().hex[:6]}"
    _make_external_table(workspace_client, cat, schema, table_name, pd.DataFrame({"id": [1], "name": ["Alice"]}))

    warehouse = workspace_client.warehouses.create(name=f"insert_cols_wh_{uuid4().hex[:6]}")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=(f"INSERT INTO {cat.name}.{schema.name}.{table_name} (name, id) VALUES ('Bob', 2)"),
    )
    assert result.status.state.value == "SUCCEEDED"

    select = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT * FROM {cat.name}.{schema.name}.{table_name} ORDER BY id",
    )
    assert select.result.data_array == [[1, "Alice"], [2, "Bob"]]

    print(f"✓ Real INSERT with explicit column list worked: {table_name}")


@pytest.mark.serial
@pytest.mark.workflow
@pytest.mark.parametrize(
    "statement_template",
    [
        pytest.param("INSERT INTO {target} SELECT id FROM {source}", id="insert-select"),
        pytest.param("DELETE FROM {target} WHERE id = 1", id="delete"),
        pytest.param("UPDATE {target} SET id = 2 WHERE id = 1", id="update"),
    ],
)
@pytest.mark.parametrize(
    ("catalog", "requested_schema"),
    [
        pytest.param("missing_catalog", "missing_schema", id="missing-catalog-and-schema"),
        pytest.param("main", None, id="missing-catalog-matches-default-schema"),
        pytest.param("", None, id="empty-catalog"),
        pytest.param(None, "", id="empty-schema"),
    ],
)
def test_external_delta_write_rejects_invalid_namespace(
    delta_catalog_and_schema,
    workspace_client: WorkspaceClient,
    statement_template: str,
    catalog: str,
    requested_schema: str | None,
):
    """Invalid statement namespaces must not mutate an EXTERNAL Delta table."""
    cat, schema = delta_catalog_and_schema
    table_name = f"people_invalid_ns_{uuid4().hex[:6]}"
    storage_location = _make_external_table(workspace_client, cat, schema, table_name, pd.DataFrame({"id": [1]}))
    warehouse = workspace_client.warehouses.create(name=f"insert_invalid_ns_wh_{uuid4().hex[:6]}")
    source_table = f"namespace_source_{uuid4().hex[:6]}"

    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"CREATE TABLE {source_table} AS SELECT 99::BIGINT AS id",
    )
    try:
        with pytest.raises(DatabricksError):
            workspace_client.statement_execution.execute_statement(
                warehouse_id=warehouse.id,
                statement=statement_template.format(
                    target=f"{cat.name}.{schema.name}.{table_name}", source=source_table
                ),
                catalog=catalog,
                schema=requested_schema,
            )

        assert DeltaTable(storage_location).to_pandas()["id"].tolist() == [1]
    finally:
        workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"DROP TABLE IF EXISTS {source_table}",
        )


@pytest.mark.serial
@pytest.mark.workflow
def test_delete_from_external_delta_table(delta_catalog_and_schema, workspace_client: WorkspaceClient):
    """DELETE FROM an EXTERNAL Delta table triggers a real Spark DeltaTable.delete()."""
    cat, schema = delta_catalog_and_schema
    table_name = f"people_del_{uuid4().hex[:6]}"
    _make_external_table(
        workspace_client,
        cat,
        schema,
        table_name,
        pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}),
    )

    warehouse = workspace_client.warehouses.create(name=f"delete_wh_{uuid4().hex[:6]}")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"DELETE FROM {cat.name}.{schema.name}.{table_name} WHERE id = 2",
    )
    assert result.status.state.value == "SUCCEEDED"

    select = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT * FROM {cat.name}.{schema.name}.{table_name} ORDER BY id",
    )
    assert select.result.data_array == [[1, "Alice"], [3, "Charlie"]]

    print(f"✓ Real DELETE via Spark removed the row: {table_name}")


@pytest.mark.serial
@pytest.mark.workflow
def test_update_external_delta_table(delta_catalog_and_schema, workspace_client: WorkspaceClient):
    """UPDATE an EXTERNAL Delta table triggers a real Spark DeltaTable.update()."""
    cat, schema = delta_catalog_and_schema
    table_name = f"people_upd_{uuid4().hex[:6]}"
    _make_external_table(
        workspace_client, cat, schema, table_name, pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    )

    warehouse = workspace_client.warehouses.create(name=f"update_wh_{uuid4().hex[:6]}")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"UPDATE {cat.name}.{schema.name}.{table_name} SET name = 'Bobby' WHERE id = 2",
    )
    assert result.status.state.value == "SUCCEEDED"

    select = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"SELECT * FROM {cat.name}.{schema.name}.{table_name} ORDER BY id",
    )
    assert select.result.data_array == [[1, "Alice"], [2, "Bobby"]]

    print(f"✓ Real UPDATE via Spark modified the row: {table_name}")
