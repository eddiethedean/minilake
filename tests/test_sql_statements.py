"""SQL Statement Execution API tests."""

import csv
import io
import json
import os
import uuid

import pytest
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.catalog import DataSourceFormat, TableType
from databricks.sdk.service.sql import ColumnInfo, Disposition, Format

from minilake.services import sql_statements as sql_service
from minilake.services import unity_catalog


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


@pytest.mark.parametrize(
    "statement, expected",
    [
        pytest.param(
            "SELECT 'rewrite.default.external' AS value",
            "SELECT 'rewrite.default.external' AS value",
            id="single-quoted-string",
        ),
        pytest.param(
            "SELECT $$rewrite.default.external$$ AS value",
            "SELECT $$rewrite.default.external$$ AS value",
            id="dollar-quoted-string",
        ),
        pytest.param(
            r"SELECT E'it\'s rewrite.default.external' AS value",
            r"SELECT E'it\'s rewrite.default.external' AS value",
            id="escaped-string",
        ),
        pytest.param(
            'SELECT 1 AS "rewrite.default.external"',
            'SELECT 1 AS "rewrite.default.external"',
            id="quoted-alias",
        ),
        pytest.param(
            "SELECT 1 /* rewrite.default.external */ -- rewrite.default.external\n",
            "SELECT 1 /* rewrite.default.external */ -- rewrite.default.external\n",
            id="comments",
        ),
        pytest.param(
            "SELECT * FROM rewrite.default.external",
            "SELECT * FROM delta_scan('/data/external''s')",
            id="from-relation-and-escaped-storage-path",
        ),
        pytest.param(
            'SELECT * FROM "rewrite"."default"."external"',
            "SELECT * FROM delta_scan('/data/external''s')",
            id="quoted-from-relation",
        ),
        pytest.param(
            "SELECT * FROM (SELECT 1 AS a) x JOIN (SELECT 2 AS b) y ON true, rewrite.default.external",
            "SELECT * FROM (SELECT 1 AS a) x JOIN (SELECT 2 AS b) y ON true, delta_scan('/data/external''s')",
            id="comma-relation-after-join-on",
        ),
        pytest.param(
            "SELECT * FROM (SELECT 1 AS a) x JOIN (SELECT 1 AS a) y USING (a), rewrite.default.external",
            "SELECT * FROM (SELECT 1 AS a) x JOIN (SELECT 1 AS a) y USING (a), delta_scan('/data/external''s')",
            id="comma-relation-after-join-using",
        ),
        pytest.param(
            "SELECT * FROM set.default.external",
            "SELECT * FROM delta_scan('/data/set')",
            id="unreserved-keyword-catalog",
        ),
        pytest.param(
            "SELECT * FROM values.default.external",
            "SELECT * FROM delta_scan('/data/values')",
            id="values-keyword-catalog",
        ),
        pytest.param(
            "WITH x AS (FROM rewrite.default.external) SELECT * FROM x",
            "WITH x AS (FROM delta_scan('/data/external''s')) SELECT * FROM x",
            id="cte-from-first",
        ),
        pytest.param(
            "WITH values AS (SELECT 1) FROM rewrite.default.external",
            "WITH values AS (SELECT 1) FROM delta_scan('/data/external''s')",
            id="values-cte-from-first",
        ),
        pytest.param(
            "CREATE TEMP TABLE values AS FROM rewrite.default.external",
            "CREATE TEMP TABLE values AS FROM delta_scan('/data/external''s')",
            id="values-table-from-first",
        ),
        pytest.param(
            "DELETE FROM target USING rewrite.default.external AS source WHERE target.id = source.id",
            "DELETE FROM target USING delta_scan('/data/external''s') AS source WHERE target.id = source.id",
            id="delete-using-external-source",
        ),
        pytest.param(
            "MERGE INTO target USING rewrite.default.external AS source ON target.id = source.id "
            "WHEN MATCHED THEN DELETE",
            "MERGE INTO target USING delta_scan('/data/external''s') AS source ON target.id = source.id "
            "WHEN MATCHED THEN DELETE",
            id="merge-using-external-source",
        ),
        pytest.param(
            "SELECT (FROM rewrite.default.external SELECT max(id))",
            "SELECT (FROM delta_scan('/data/external''s') SELECT max(id))",
            id="scalar-subquery-from-first",
        ),
        pytest.param(
            "SELECT substring('abcdef' FROM rewrite.default.external)",
            "SELECT substring('abcdef' FROM rewrite.default.external)",
            id="substring-from-expression",
        ),
        pytest.param(
            "SELECT substring($$abcdef$$ FROM rewrite.default.external)",
            "SELECT substring($$abcdef$$ FROM rewrite.default.external)",
            id="substring-dollar-string-from-expression",
        ),
        pytest.param(
            "PIVOT rewrite.default.external ON category USING sum(id)",
            "PIVOT delta_scan('/data/external''s') ON category USING sum(id)",
            id="standalone-pivot-source",
        ),
        pytest.param(
            "SELECT * FROM (PIVOT rewrite.default.external ON category USING sum(id)) p",
            "SELECT * FROM (PIVOT delta_scan('/data/external''s') ON category USING sum(id)) p",
            id="nested-pivot-source",
        ),
        pytest.param(
            "PIVOT_WIDER rewrite.default.external ON category USING sum(id)",
            "PIVOT_WIDER delta_scan('/data/external''s') ON category USING sum(id)",
            id="pivot-wider-source",
        ),
        pytest.param(
            "UNPIVOT rewrite.default.external ON id INTO NAME field VALUE amount",
            "UNPIVOT delta_scan('/data/external''s') ON id INTO NAME field VALUE amount",
            id="unpivot-source",
        ),
        pytest.param(
            "SELECT extract(year FROM rewrite.default.external)",
            "SELECT extract(year FROM rewrite.default.external)",
            id="extract-from-expression",
        ),
        pytest.param(
            "SELECT 0 IS DISTINCT FROM rewrite.default.external",
            "SELECT 0 IS DISTINCT FROM rewrite.default.external",
            id="distinct-from-expression",
        ),
        pytest.param(
            "SELECT s.from, rewrite.default.external FROM (SELECT {'from': 1} AS s)",
            "SELECT s.from, rewrite.default.external FROM (SELECT {'from': 1} AS s)",
            id="qualified-field-named-from",
        ),
    ],
)
def test_external_table_rewrite_preserves_sql_context(monkeypatch, statement, expected):
    """Only external table relation names are rewritten, with safe path quoting."""
    monkeypatch.setitem(
        unity_catalog._state,
        "external_tables",
        {
            "rewrite.default.external": {"storage_location": "/data/external's"},
            "set.default.external": {"storage_location": "/data/set"},
            "values.default.external": {"storage_location": "/data/values"},
        },
    )

    assert sql_service._rewrite_sql_for_duckdb(statement) == expected


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
@pytest.mark.parametrize(
    "mutation, expected_rows",
    [
        pytest.param(
            "DELETE FROM {table} -- DDL example: ) USING DELTA\nWHERE id = 1",
            [[2]],
            id="delete-where-after-comment",
        ),
        pytest.param(
            "UPDATE {table} SET id = id + 10 -- DDL example: ) USING DELTA\nWHERE id = 1",
            [[2], [11]],
            id="update-where-after-comment",
        ),
    ],
)
def test_dml_where_after_ddl_like_comment_is_preserved(workspace_client, mutation, expected_rows):
    """DDL compatibility rewriting must not truncate DML at comment text."""
    wh = workspace_client.warehouses.create(name="ddl_comment_dml_wh")
    table = f"ddl_comment_dml_{uuid.uuid4().hex[:8]}"
    setup = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"CREATE TABLE {table} AS SELECT 1 AS id UNION ALL SELECT 2",
    )
    assert setup.status.state.value == "SUCCEEDED"

    mutation_result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=mutation.format(table=table),
    )
    assert mutation_result.status.state.value == "SUCCEEDED"

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"SELECT id FROM {table} ORDER BY id",
    )
    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == expected_rows


@pytest.mark.crud
def test_ctas_where_after_ddl_like_comment_is_preserved(workspace_client: WorkspaceClient):
    """A comment in CTAS must not make the compatibility pass remove its filter."""
    wh = workspace_client.warehouses.create(name="ddl_comment_ctas_wh")
    source = f"ddl_comment_source_{uuid.uuid4().hex[:8]}"
    target = f"ddl_comment_target_{uuid.uuid4().hex[:8]}"
    setup = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"CREATE TABLE {source} AS SELECT 1 AS id UNION ALL SELECT 2",
    )
    assert setup.status.state.value == "SUCCEEDED"

    ctas = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=(f"CREATE TABLE {target} AS SELECT id FROM {source} -- DDL example: ) USING DELTA\nWHERE id = 1"),
    )
    assert ctas.status.state.value == "SUCCEEDED"

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"SELECT id FROM {target}",
    )
    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == [[1]]


@pytest.mark.crud
def test_ctas_using_sample_clause_is_preserved(workspace_client: WorkspaceClient):
    """A CTAS function call must not be mistaken for a DDL column list."""
    wh = workspace_client.warehouses.create(name="ddl_comment_sample_wh")
    target = f"ddl_comment_sample_{uuid.uuid4().hex[:8]}"

    ctas = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=(
            f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM range(100) /* preserve sampling */ USING SAMPLE 0 ROWS"
        ),
    )
    assert ctas.status.state.value == "SUCCEEDED"

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"SELECT count(*) FROM {target}",
    )
    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == [[0]]


@pytest.mark.crud
def test_delete_comparison_preserves_continued_escape_string(workspace_client: WorkspaceClient):
    """DDL-like text and semicolons in an E-string must remain literal data."""
    wh = workspace_client.warehouses.create(name="ddl_comment_escape_wh")
    table = f"ddl_comment_escape_{uuid.uuid4().hex[:8]}"
    decoy = f"ddl_comment_decoy_{uuid.uuid4().hex[:8]}"
    note = f"a'; CREATE TABLE {decoy} (id INTEGER) /* literal data */ USING DELTA; tail"
    escaped_note = "'" + note.replace("'", "''") + "'"

    setup = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=(f"CREATE TABLE {table} AS SELECT 1 AS id, {escaped_note} AS note UNION ALL SELECT 2, 'other'"),
    )
    assert setup.status.state.value == "SUCCEEDED"

    deletion = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=(
            f"DELETE FROM {table} WHERE note <> E'a'\n"
            rf"'\'; CREATE TABLE {decoy} (id INTEGER) /* literal data */ USING DELTA; tail'"
        ),
    )
    assert deletion.status.state.value == "SUCCEEDED"

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"SELECT id FROM {table} ORDER BY id",
    )
    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == [[1]]


@pytest.mark.crud
def test_multistatement_delete_where_after_ddl_like_comment_is_preserved(workspace_client: WorkspaceClient):
    """Rewriting one DDL statement must preserve a later commented DELETE filter."""
    wh = workspace_client.warehouses.create(name="ddl_comment_multi_wh")
    guard = f"ddl_comment_guard_{uuid.uuid4().hex[:8]}"
    marker = f"ddl_comment_marker_{uuid.uuid4().hex[:8]}"
    setup = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"CREATE TABLE {guard} AS SELECT 1 AS id UNION ALL SELECT 2",
    )
    assert setup.status.state.value == "SUCCEEDED"

    executed = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=(
            f"CREATE TABLE {marker} (id INTEGER);\nDELETE FROM {guard} -- DDL example: ) USING DELTA\nWHERE id = 1"
        ),
    )
    assert executed.status.state.value == "SUCCEEDED"

    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=f"SELECT id FROM {guard} ORDER BY id",
    )
    assert result.status.state.value == "SUCCEEDED"
    assert result.result.data_array == [[2]]


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
def test_statement_cache_overflow_preserves_newest_external_result(workspace_client: WorkspaceClient):
    """Crossing the statement cache limit must not erase the newest result."""
    wh = workspace_client.warehouses.create(name="statement_cache_overflow_wh")
    cache_limit = max(1, int(os.getenv("MINILAKE_STATEMENT_CACHE_SIZE", "500")))
    verify_at = {cache_limit - 1, cache_limit, cache_limit + 1}

    for value in range(cache_limit + 2):
        result = workspace_client.statement_execution.execute_statement(
            warehouse_id=wh.id,
            statement=f"SELECT {value} AS n",
            disposition=Disposition.EXTERNAL_LINKS,
            format=Format.JSON_ARRAY,
        )

        if value in verify_at:
            assert result.result.external_links is not None
            link = result.result.external_links[0]
            assert link.row_count == 1
            response = requests.get(link.external_link)
            assert response.status_code == 200
            assert json.loads(response.text) == [[value]]


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


@pytest.mark.workflow
def test_external_links_arrow_stream_preserves_sql_types_and_values(workspace_client: WorkspaceClient):
    """Arrow results retain SQL types, binary bytes, and nested values on the wire."""
    from datetime import date, datetime
    from decimal import Decimal

    import pyarrow as pa

    wh = workspace_client.warehouses.create(name="ext_links_arrow_types_wh")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=(
            "SELECT DATE '2024-01-01' AS date_value, "
            "DATE '10000-01-01' AS far_date_value, "
            "DATE 'infinity' AS infinite_date_value, "
            "TIMESTAMP '2024-01-02 03:04:05.123456' AS timestamp_value, "
            "CAST(123.45 AS DECIMAL(7, 2)) AS decimal_value, "
            "from_hex('FF00') AS binary_value, "
            "NULL::INTEGER AS null_value, "
            "[1, 2]::INTEGER[] AS list_value, "
            "struct_pack(label := 'nested') AS struct_value"
        ),
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.ARROW_STREAM,
    )

    link = result.result.external_links[0]
    response = requests.get(link.external_link)
    assert response.status_code == 200

    table = pa.ipc.open_stream(response.content).read_all()
    assert table.schema.field("date_value").type == pa.date32()
    assert table.schema.field("far_date_value").type == pa.date32()
    assert table.schema.field("infinite_date_value").type == pa.date32()
    assert table.schema.field("timestamp_value").type == pa.timestamp("us")
    assert table.schema.field("decimal_value").type == pa.decimal128(7, 2)
    assert table.schema.field("binary_value").type == pa.binary()
    assert table.schema.field("null_value").type == pa.int32()
    assert table.schema.field("list_value").type == pa.list_(pa.int32())
    assert table.schema.field("struct_value").type == pa.struct([("label", pa.string())])

    assert table.column("date_value").to_pylist() == [date(2024, 1, 1)]
    assert table.column("far_date_value").cast(pa.int32()).to_pylist() == [
        date.max.toordinal() - date(1970, 1, 1).toordinal() + 1
    ]
    assert table.column("infinite_date_value").cast(pa.int32()).to_pylist() == [2**31 - 1]
    assert table.column("timestamp_value").to_pylist() == [datetime(2024, 1, 2, 3, 4, 5, 123456)]
    assert table.column("decimal_value").to_pylist() == [Decimal("123.45")]
    assert table.column("binary_value").to_pylist() == [b"\xff\x00"]
    assert table.column("null_value").to_pylist() == [None]
    assert table.column("list_value").to_pylist() == [[1, 2]]
    assert table.column("struct_value").to_pylist() == [{"label": "nested"}]


def test_arrow_chunk_serialization_preserves_extension_field_metadata():
    """Arrow IPC chunks retain metadata needed to interpret opaque SQL types."""
    import pyarrow as pa

    from minilake.services.sql_statements import _serialize_arrow_table

    field_metadata = {
        b"ARROW:extension:name": b"arrow.opaque",
        b"ARROW:extension:metadata": b'{"type_name":"bignum","vendor_name":"DuckDB"}',
    }
    schema = pa.schema([pa.field("value", pa.binary(), metadata=field_metadata)])
    source = pa.Table.from_arrays([pa.array([b"\x80\x00\x01{"])], schema=schema)

    serialized = _serialize_arrow_table(source, 0, source.num_rows)
    restored = pa.ipc.open_stream(serialized).read_all()

    restored_field = restored.schema.field("value")
    if hasattr(restored_field.type, "extension_name"):
        assert restored_field.type.extension_name == "arrow.opaque"
        assert restored_field.type.storage_type == pa.binary()
        assert restored_field.type.type_name == "bignum"
        assert restored_field.type.vendor_name == "DuckDB"
    else:
        assert restored_field.type == pa.binary()
        assert restored_field.metadata == field_metadata
    assert restored.column("value").to_pylist() == [b"\x80\x00\x01{"]


@pytest.mark.workflow
@pytest.mark.parametrize(
    "statement, expected_rows",
    [
        ("SELECT NULL::INTEGER AS value", [None]),
        ("SELECT NULL::INTEGER AS value WHERE FALSE", []),
    ],
    ids=["typed-all-null", "typed-zero-rows"],
)
def test_external_links_arrow_stream_preserves_typed_null_and_empty_columns(
    workspace_client: WorkspaceClient, statement: str, expected_rows: list
):
    """Null-only and empty Arrow results retain the declared SQL column type."""
    import pyarrow as pa

    wh = workspace_client.warehouses.create(name="ext_links_arrow_empty_wh")
    result = workspace_client.statement_execution.execute_statement(
        warehouse_id=wh.id,
        statement=statement,
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.ARROW_STREAM,
    )

    link = result.result.external_links[0]
    response = requests.get(link.external_link)
    assert response.status_code == 200

    table = pa.ipc.open_stream(response.content).read_all()
    assert table.schema.field("value").type == pa.int32()
    assert table.column("value").to_pylist() == expected_rows

