"""SQL Statement Execution API tests."""

import csv
import io
import json
import uuid

import pytest
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.catalog import DataSourceFormat, TableType
from databricks.sdk.service.sql import ColumnInfo, Disposition, Format


@pytest.mark.crud
@pytest.mark.parametrize(
    "statement, expected_value, expected_column",
    [
        pytest.param("SELECT 1 AS value", 1, "value", id="unquoted-identifier"),
        pytest.param("SELECT `value` FROM (SELECT 1 AS `value`)", 1, "value", id="backtick-reference"),
        pytest.param("SELECT '`hello`' AS `value`", "`hello`", "value", id="string-literal"),
        pytest.param("SELECT 'it''s `quoted`' AS `value`", "it's `quoted`", "value", id="escaped-single-quote"),
        pytest.param(r"SELECT E'it\'s' AS `value`", "it's", "value", id="escape-string"),
        pytest.param(r"SELECT e'it\'s `quoted`' AS `value`", "it's `quoted`", "value", id="lowercase-escape-string"),
        pytest.param(r"SELECT E'path\\' AS `value`", "path\\", "value", id="escape-string-backslash"),
        pytest.param(
            "SELECT E'it'\n'\\'s' AS `value`",
            "it's",
            "value",
            id="escape-string-newline-continuation",
        ),
        pytest.param(
            "SELECT E'it' -- continuation\n'\\'s' AS `value`",
            "it's",
            "value",
            id="escape-string-comment-continuation",
        ),
        pytest.param(
            "SELECT E'it'\u00a0\n'\\'s' AS `value`",
            "it's",
            "value",
            id="escape-string-unicode-space-continuation",
        ),
        pytest.param(r"SELECT 'path\' AS `value`", "path\\", "value", id="ordinary-string-backslash"),
        pytest.param("SELECT $$it's text$$ AS `value`", "it's text", "value", id="dollar-string"),
        pytest.param("SELECT $tag$it's text$tag$ AS `value`", "it's text", "value", id="tagged-dollar-string"),
        pytest.param(
            "SELECT $a\u0301$it's text$a\u0301$ AS `value`", "it's text", "value", id="combining-mark-dollar-tag"
        ),
        pytest.param("SELECT $a\u00a0$it's text$a\u00a0$ AS `value`", "it's text", "value", id="whitespace-dollar-tag"),
        pytest.param("SELECT $😀$it's text$😀$ AS `value`", "it's text", "value", id="symbol-dollar-tag"),
        pytest.param(
            "SELECT col😀$tag$ AS `value` FROM (SELECT 1 AS col😀$tag$)", 1, "value", id="symbol-in-identifier"
        ),
        pytest.param(
            "SELECT col\u0085$tag$ AS `value` FROM (SELECT 1 AS col\u0085$tag$)",
            1,
            "value",
            id="unicode-identifier-before-dollar",
        ),
        pytest.param("SELECT $$`hello`$$ AS `value`", "`hello`", "value", id="dollar-string-backticks"),
        pytest.param(
            "SELECT $tag$it's $$ `quoted` $other$ text$tag$ AS `value`",
            "it's $$ `quoted` $other$ text",
            "value",
            id="dollar-string-matching-tag",
        ),
        pytest.param("SELECT col$tag$ AS `value` FROM (SELECT 1 AS col$tag$)", 1, "value", id="dollar-in-identifier"),
        pytest.param('SELECT 1 AS "a`b`c"', 1, "a`b`c", id="double-quoted-identifier"),
        pytest.param("SELECT 1 AS `a``b`", 1, "a`b", id="escaped-backtick"),
        pytest.param('SELECT 1 AS `a"b`', 1, 'a"b', id="identifier-with-double-quote"),
        pytest.param("SELECT 1 -- unmatched ` in comment\n AS `value`", 1, "value", id="line-comment"),
        pytest.param("SELECT /* unmatched ` in comment */ 1 AS `value`", 1, "value", id="block-comment"),
        pytest.param(
            "SELECT /* outer /* inner */ unmatched ` */ 1 AS `value`",
            1,
            "value",
            id="nested-block-comment",
        ),
    ],
)
def test_backtick_identifier_translation(workspace_client: WorkspaceClient, statement, expected_value, expected_column):
    """Translate identifier quotes without changing string values or other SQL tokens."""
    wh = workspace_client.warehouses.create(name="backtick_wh")

    response = workspace_client.statement_execution.execute_statement(warehouse_id=wh.id, statement=statement)

    assert response.status.state.value == "SUCCEEDED"
    assert response.result.data_array == [[expected_value]]
    assert response.manifest.schema.columns[0].name == expected_column


@pytest.mark.crud
def test_backtick_identifier_after_unicode_typed_literal(workspace_client: WorkspaceClient):
    """A combining mark in a type name must not make its trailing e an E-string prefix."""
    wh = workspace_client.warehouses.create(name="unicode_type_wh")
    type_name = f"enum_{uuid.uuid4().hex}\u0301e"
    setup = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"CREATE TYPE {type_name} AS ENUM ('path\\')",
    )
    assert setup.status.state.value == "SUCCEEDED"

    response = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"SELECT {type_name}'path\\' AS `value`",
    )

    assert response.status.state.value == "SUCCEEDED"
    assert response.result.data_array == [["path\\"]]
    assert response.manifest.schema.columns[0].name == "value"


@pytest.fixture
def warehouse_with_table(workspace_client: WorkspaceClient):
    """Fixture: warehouse, catalog, schema, and table pre-created."""
    wh = workspace_client.warehouses.create(name="stmt_wh")
    workspace_client.catalogs.create(name="stmt_cat")
    workspace_client.schemas.create(name="stmt_schema", catalog_name="stmt_cat")
    workspace_client.tables.create(
        name="stmt_table",
        catalog_name="stmt_cat",
        schema_name="stmt_schema",
        storage_location="/data/stmt_cat/stmt_schema/stmt_table",
        columns=[
            ColumnInfo(name="id", type_text="INTEGER"),
            ColumnInfo(name="value", type_text="VARCHAR"),
        ],
        table_type=TableType.MANAGED,
        data_source_format=DataSourceFormat.DELTA,
    )
    return wh, "stmt_cat", "stmt_schema", "stmt_table"


@pytest.mark.crud
def test_execute_select_empty_table(workspace_client: WorkspaceClient):
    """Test: Execute SELECT on empty table returns empty result."""
    wh = workspace_client.warehouses.create(name="empty_select_wh")
    workspace_client.catalogs.create(name="empty_cat")
    workspace_client.schemas.create(name="empty_schema", catalog_name="empty_cat")
    workspace_client.tables.create(
        name="empty_table",
        catalog_name="empty_cat",
        schema_name="empty_schema",
        storage_location="/data/empty_cat/empty_schema/empty_table",
        columns=[ColumnInfo(name="id", type_text="INTEGER")],
        table_type=TableType.MANAGED,
        data_source_format=DataSourceFormat.DELTA,
    )

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT * FROM empty_cat.empty_schema.empty_table",
    )

    assert result.status.state.value == "SUCCEEDED"
    assert result.result is not None
    assert len(result.result.data_array) == 0

    print("✓ SELECT on empty table returns empty result")


@pytest.mark.crud
def test_result_manifest_carries_column_types(workspace_client: WorkspaceClient):
    """The result manifest reports each column's type.

    Read through `manifest.schema`, not `result.columns`: the SDK's `ResultData`
    has no `columns` field, so a schema published only there never reaches a
    typed client.
    """
    wh = workspace_client.warehouses.create(name="coltype_wh")

    response = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT 1 AS an_int, 'x' AS a_string, count(*) AS a_count",
    )

    assert response.manifest is not None
    schema = response.manifest.schema
    assert schema.column_count == 3

    by_name = {c.name: c for c in schema.columns}
    assert by_name["an_int"].type_text == "INTEGER"
    assert by_name["a_string"].type_text == "VARCHAR"
    assert by_name["a_count"].type_text == "BIGINT"
    assert [c.position for c in schema.columns] == [0, 1, 2]
    assert response.manifest.total_row_count == 1

    print("✓ Result manifest carries DuckDB column types")


@pytest.mark.crud
def test_get_statement_returns_manifest(workspace_client: WorkspaceClient):
    """Polling a statement returns the same schema the execute call did."""
    wh = workspace_client.warehouses.create(name="manifest_poll_wh")
    executed = workspace_client.statement_execution.execute_statement(warehouse_id=wh.id, statement="SELECT 7 AS lucky")

    fetched = workspace_client.statement_execution.get_statement(statement_id=executed.statement_id)

    assert fetched.manifest is not None
    assert [c.name for c in fetched.manifest.schema.columns] == ["lucky"]
    assert fetched.manifest.schema.columns[0].type_text == "INTEGER"

    print("✓ get_statement returns the result manifest")


@pytest.mark.crud
def test_execute_insert_and_select(warehouse_with_table, workspace_client):
    """Test: INSERT and SELECT data through SQL statements."""
    wh, cat, schema, table = warehouse_with_table

    # INSERT
    insert_result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"INSERT INTO {cat}.{schema}.{table} VALUES (1, 'one'), (2, 'two')",
    )
    assert insert_result.status.state.value == "SUCCEEDED"

    # SELECT
    select_result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"SELECT * FROM {cat}.{schema}.{table} ORDER BY id",
    )
    assert select_result.status.state.value == "SUCCEEDED"
    assert len(select_result.result.data_array) == 2
    assert select_result.result.data_array[0] == [1, "one"]
    assert select_result.result.data_array[1] == [2, "two"]

    print("✓ INSERT and SELECT executed successfully")


@pytest.mark.parametrize(
    ("catalog", "schema"),
    [
        pytest.param("missing_namespace_catalog", "missing_schema", id="missing-catalog"),
        pytest.param("namespace_valid_catalog", "missing_schema", id="missing-schema"),
        pytest.param("main", None, id="missing-catalog-matches-default-schema"),
        pytest.param("", None, id="empty-catalog"),
        pytest.param(None, "", id="empty-schema"),
        pytest.param("", "", id="both-empty"),
    ],
)
@pytest.mark.parametrize(
    "statement_template",
    [
        pytest.param("DELETE FROM {table}", id="delete"),
        pytest.param("UPDATE {table} SET id = id + 1", id="update"),
        pytest.param("INSERT INTO {table} VALUES (30)", id="insert"),
    ],
)
@pytest.mark.workflow
def test_invalid_namespace_rejects_statement_without_fallback(
    workspace_client: WorkspaceClient,
    catalog: str,
    schema: str | None,
    statement_template: str,
):
    """Invalid explicit catalog/schema must not run SQL in the connection's old namespace."""
    warehouse = workspace_client.warehouses.create(name="invalid_namespace_wh")
    if catalog == "namespace_valid_catalog":
        workspace_client.catalogs.create(name=catalog)

    table_name = f"namespace_guard_{uuid.uuid4().hex[:8]}"
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"CREATE TABLE {table_name} AS SELECT 10 AS id UNION ALL SELECT 20",
    )

    try:
        with pytest.raises(DatabricksError):
            workspace_client.statement_execution.execute_statement(
                warehouse_id=warehouse.id,
                statement=statement_template.format(table=table_name),
                catalog=catalog,
                schema=schema,
            )

        result = workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"SELECT id FROM {table_name} ORDER BY id",
        )
        assert result.result.data_array == [[10], [20]]
    finally:
        workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"DROP TABLE IF EXISTS {table_name}",
        )


@pytest.mark.serial
@pytest.mark.workflow
def test_catalog_only_namespace_is_not_confused_with_same_named_schema(workspace_client: WorkspaceClient):
    """A catalog-only context must select that catalog when its name is also a schema."""
    warehouse = workspace_client.warehouses.create(name="namespace_collision_wh")
    catalog = f"namespace_catalog_{uuid.uuid4().hex[:8]}"
    table_name = f"namespace_collision_{uuid.uuid4().hex[:8]}"
    workspace_client.catalogs.create(name=catalog)

    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f'CREATE SCHEMA unity_catalog."{catalog}"',
    )
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"CREATE TABLE unity_catalog.{catalog}.{table_name} AS SELECT 99 AS id",
    )
    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=f"CREATE TABLE {catalog}.main.{table_name} AS SELECT 1 AS id",
    )

    try:
        result = workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"DELETE FROM {table_name}",
            catalog=catalog,
        )
        assert result.result.data_array == [[1]]

        intended_catalog = workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"SELECT id FROM {catalog}.main.{table_name}",
        )
        same_named_schema = workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"SELECT id FROM unity_catalog.{catalog}.{table_name}",
        )
        assert intended_catalog.result.data_array == []
        assert same_named_schema.result.data_array == [[99]]
    finally:
        workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"DROP TABLE IF EXISTS {catalog}.main.{table_name}",
        )
        workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f"DROP TABLE IF EXISTS unity_catalog.{catalog}.{table_name}",
        )
        workspace_client.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            statement=f'DROP SCHEMA IF EXISTS unity_catalog."{catalog}" CASCADE',
        )


@pytest.mark.crud
def test_execute_statement_with_syntax_error(workspace_client: WorkspaceClient):
    """Test: SQL with syntax error returns error status."""
    wh = workspace_client.warehouses.create(name="error_wh")

    with pytest.raises(Exception):
        workspace_client.statement_execution.execute_statement(
            warehouse_id=wh.id,
            statement="INVALID SQL STATEMENT",
        )

    print("✓ Syntax error raises exception")


@pytest.mark.crud
def test_execute_create_table_as_select(workspace_client: WorkspaceClient):
    """Test: CREATE TABLE AS SELECT (CTAS) creates real table."""
    wh = workspace_client.warehouses.create(name="ctas_wh")
    workspace_client.catalogs.create(name="ctas_cat")
    workspace_client.schemas.create(name="ctas_schema", catalog_name="ctas_cat")

    # CTAS
    ctas_result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="""
            CREATE TABLE ctas_cat.ctas_schema.ctas_table AS
            SELECT 1 AS id, 'test' AS name
        """,
    )
    assert ctas_result.status.state.value == "SUCCEEDED"

    # Verify table exists via GET
    table = workspace_client.tables.get(full_name="ctas_cat.ctas_schema.ctas_table")
    assert table.name == "ctas_table"

    # Verify data via SELECT
    select_result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT * FROM ctas_cat.ctas_schema.ctas_table",
    )
    assert len(select_result.result.data_array) == 1
    assert select_result.result.data_array[0] == [1, "test"]

    print("✓ CREATE TABLE AS SELECT created real table with data")


@pytest.mark.crud
def test_get_statement_returns_status(workspace_client):
    """Test: GET /api/2.0/sql/statements/{id} returns statement status and result."""
    wh = workspace_client.warehouses.create(name="get_stmt_wh")
    workspace_client.catalogs.create(name="get_cat")
    workspace_client.schemas.create(name="get_schema", catalog_name="get_cat")
    workspace_client.tables.create(
        name="get_table",
        catalog_name="get_cat",
        schema_name="get_schema",
        storage_location="/data/get_cat/get_schema/get_table",
        columns=[ColumnInfo(name="id", type_text="INTEGER")],
        table_type=TableType.MANAGED,
        data_source_format=DataSourceFormat.DELTA,
    )

    # Execute statement
    execute_result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT COUNT(*) FROM get_cat.get_schema.get_table",
    )
    statement_id = execute_result.statement_id

    # Get statement
    get_result = workspace_client.statement_execution.get_statement(statement_id=statement_id)

    assert get_result.statement_id == statement_id
    assert get_result.status.state.value == "SUCCEEDED"
    assert get_result.result is not None

    print("✓ GET statement returns status and result")


@pytest.mark.error
def test_arrow_format_requires_external_links(workspace_client: WorkspaceClient):
    """Real Databricks rule: ARROW_STREAM/CSV require EXTERNAL_LINKS, not INLINE."""
    wh = workspace_client.warehouses.create(name="fmt_rule_wh")

    with pytest.raises(Exception):
        workspace_client.statement_execution.execute_statement(
            warehouse_id=wh.id,
            statement="SELECT 1",
            disposition=Disposition.INLINE,
            format=Format.ARROW_STREAM,
        )

    print("✓ ARROW_STREAM with INLINE disposition correctly rejected")


@pytest.mark.workflow
def test_external_links_json_array_real_fetch(workspace_client: WorkspaceClient):
    """EXTERNAL_LINKS + JSON_ARRAY: real self-hosted link serves the real row data."""
    wh = workspace_client.warehouses.create(name="ext_links_wh")

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.JSON_ARRAY,
    )

    assert result.status.state.value == "SUCCEEDED"
    assert result.result.external_links is not None
    link = result.result.external_links[0]
    assert link.row_count == 2

    # Real HTTP fetch of the link — exactly what a real client does for
    # Databricks' presigned cloud storage URLs, just pointed at minilake itself.
    resp = requests.get(link.external_link)
    assert resp.status_code == 200
    rows = json.loads(resp.text)
    assert rows == [[1, "Alice"], [2, "Bob"]]

    print(f"✓ EXTERNAL_LINKS JSON_ARRAY real fetch returned real data: {link.external_link}")


@pytest.mark.workflow
def test_external_links_csv_real_fetch(workspace_client: WorkspaceClient):
    """EXTERNAL_LINKS + CSV: real self-hosted link serves real CSV content."""
    wh = workspace_client.warehouses.create(name="ext_links_csv_wh")

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.CSV,
    )

    link = result.result.external_links[0]
    resp = requests.get(link.external_link)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")

    reader = csv.reader(io.StringIO(resp.text))
    parsed = list(reader)
    assert parsed == [["id", "name"], ["1", "Alice"], ["2", "Bob"]]

    print(f"✓ EXTERNAL_LINKS CSV real fetch returned real CSV: {link.external_link}")


@pytest.mark.workflow
def test_external_links_arrow_stream_real_fetch(workspace_client: WorkspaceClient):
    """EXTERNAL_LINKS + ARROW_STREAM: real self-hosted link serves a real Arrow IPC stream."""
    import pyarrow as pa

    wh = workspace_client.warehouses.create(name="ext_links_arrow_wh")

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.ARROW_STREAM,
    )

    link = result.result.external_links[0]
    resp = requests.get(link.external_link)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.arrow.stream"

    reader = pa.ipc.open_stream(resp.content)
    table = reader.read_all()
    assert table.column("id").to_pylist() == [1, 2]
    assert table.column("name").to_pylist() == ["Alice", "Bob"]

    print(f"✓ EXTERNAL_LINKS ARROW_STREAM real fetch returned a real Arrow table: {link.external_link}")


@pytest.mark.workflow
def test_external_links_arrow_stream_preserves_duplicate_column_names(workspace_client: WorkspaceClient):
    """ARROW_STREAM keeps duplicate result names and their positional values."""
    import pyarrow as pa

    wh = workspace_client.warehouses.create(name="ext_links_arrow_duplicate_names_wh")

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT 1 AS same, 2 AS same",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.ARROW_STREAM,
    )

    assert result.manifest.schema.column_count == 2
    assert [column.name for column in result.manifest.schema.columns] == ["same", "same"]

    link = result.result.external_links[0]
    resp = requests.get(link.external_link)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.arrow.stream"

    table = pa.ipc.open_stream(resp.content).read_all()
    assert table.column_names == ["same", "same"]
    assert table.num_columns == 2
    assert table.column(0).to_pylist() == [1]
    assert table.column(1).to_pylist() == [2]


@pytest.mark.workflow
def test_get_statement_result_chunk_n(workspace_client: WorkspaceClient):
    """get_statement_result_chunk_n fetches real chunk metadata by index."""
    wh = workspace_client.warehouses.create(name="chunk_wh")

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement="SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.JSON_ARRAY,
    )

    chunk = workspace_client.statement_execution.get_statement_result_chunk_n(
        statement_id=result.statement_id, chunk_index=0
    )
    assert chunk.row_count == 2
    assert chunk.external_links[0].external_link is not None

    print("✓ get_statement_result_chunk_n returns real chunk metadata")
