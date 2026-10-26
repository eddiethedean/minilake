"""Unity Catalog and composite tools, driven through the MCP client."""

import pytest

pytestmark = pytest.mark.anyio


async def test_catalog_schema_table_crud(mcp):
    """The UC hierarchy can be built and inspected entirely through tools."""
    await mcp.call("create_catalog", {"name": "uccat", "comment": "test"})

    catalogs = await mcp.call("list_catalogs", {})
    assert any(c["name"] == "uccat" for c in catalogs["catalogs"])

    await mcp.call("create_schema", {"name": "sch", "catalog_name": "uccat"})
    schemas = await mcp.call("list_schemas", {"catalog_name": "uccat"})
    assert any(s["name"] == "sch" for s in schemas["schemas"])

    await mcp.call(
        "create_table",
        {
            "name": "tbl",
            "catalog_name": "uccat",
            "schema_name": "sch",
            "columns": [
                {"name": "id", "type_text": "INTEGER"},
                {"name": "label", "type_text": "VARCHAR"},
            ],
        },
    )

    described = await mcp.call("describe_table", {"full_name": "uccat.sch.tbl"})
    assert [c["name"] for c in described["columns"]] == ["id", "label"]

    exists = await mcp.call("table_exists", {"full_name": "uccat.sch.tbl"})
    assert exists["table_exists"] is True

    await mcp.call("delete_table", {"full_name": "uccat.sch.tbl"})
    gone = await mcp.call("table_exists", {"full_name": "uccat.sch.tbl"})
    assert gone["table_exists"] is False


async def test_describe_catalog_tree_returns_columns_in_one_call(mcp):
    """The composite walk returns schemas, tables and column types together."""
    await mcp.call(
        "seed_table",
        {
            "full_name": "treecat.sales.orders",
            "columns": [
                {"name": "id", "type_text": "INTEGER"},
                {"name": "total", "type_text": "DOUBLE"},
            ],
            "rows": [[1, 9.5]],
        },
    )

    tree = await mcp.call("describe_catalog_tree", {"catalog": "treecat", "include_columns": True})

    assert tree["catalog"] == "treecat"
    sales = next(s for s in tree["schemas"] if s["schema"] == "sales")
    orders = next(t for t in sales["tables"] if t["name"] == "orders")
    assert [c["name"] for c in orders["columns"]] == ["id", "total"]


async def test_seed_table_creates_hierarchy_and_inserts(mcp):
    """seed_table creates catalog, schema and table, then inserts — one call."""
    result = await mcp.call(
        "seed_table",
        {
            "full_name": "seedcat.seedsch.people",
            "columns": [
                {"name": "id", "type_text": "INTEGER"},
                {"name": "name", "type_text": "VARCHAR"},
            ],
            "rows": [[1, "Alice"], [2, "Bob"]],
        },
    )
    assert result["rows_inserted"] == 2

    counted = await mcp.call("run_sql", {"statement": "SELECT COUNT(*) FROM seedcat.seedsch.people"})
    assert counted["rows"][0][0] == 2


async def test_seed_table_escapes_quotes_in_values(mcp):
    """String values containing quotes must not break the generated INSERT."""
    await mcp.call(
        "seed_table",
        {
            "full_name": "quotecat.s.t",
            "columns": [{"name": "note", "type_text": "VARCHAR"}],
            "rows": [["O'Brien"]],
        },
    )
    result = await mcp.call("run_sql", {"statement": "SELECT note FROM quotecat.s.t"})
    assert result["rows"][0][0] == "O'Brien"


async def test_seed_table_rejects_a_malformed_name(mcp):
    """A two-part name gets an actionable message, not a stack trace."""
    message = await mcp.call_expecting_error(
        "seed_table",
        {"full_name": "only.two", "columns": [{"name": "a", "type_text": "INTEGER"}], "rows": []},
    )
    assert "three-part" in message


async def test_seed_table_surfaces_a_bad_column_type(mcp):
    """The type error must reach the caller, not be swallowed into a later confusion.

    seed_table used to catch every error from POST /tables, so an unsupported type
    resurfaced as a DuckDB 'table does not exist' from the next statement.
    """
    message = await mcp.call_expecting_error(
        "seed_table",
        {
            "full_name": "badtype.s.t",
            "columns": [{"name": "c", "type_text": "NOT_A_TYPE"}],
            "rows": [[1]],
        },
    )
    assert "NOT_A_TYPE" in message
    assert "does not exist" not in message


@pytest.mark.parametrize(
    ("case", "columns", "rows", "recreate", "error_fragment"),
    [
        ("invalid_value", [{"name": "id", "type_text": "INTEGER"}], [["not an integer"]], False, "not an integer"),
        ("invalid_recreate_type", [{"name": "id", "type_text": "NOT_A_TYPE"}], [[43]], True, "not_a_type"),
        (
            "invalid_recreate_value",
            [{"name": "id", "type_text": "INTEGER"}],
            [["not an integer"]],
            True,
            "not an integer",
        ),
    ],
)
async def test_seed_table_failed_replacement_preserves_existing_table(
    mcp, case, columns, rows, recreate, error_fragment
):
    """A failed replacement must leave both the existing table and its rows intact."""
    full_name = f"seedpreserve.s.{case}"
    original_columns = [{"name": "id", "type_text": "INTEGER"}]
    await mcp.call(
        "seed_table",
        {"full_name": full_name, "columns": original_columns, "rows": [[41], [42]]},
    )

    error = await mcp.call_expecting_error(
        "seed_table",
        {"full_name": full_name, "columns": columns, "rows": rows, "recreate": recreate},
    )

    assert error_fragment in error.lower()
    result = await mcp.call("run_sql", {"statement": f'SELECT id FROM "seedpreserve"."s"."{case}" ORDER BY id'})
    assert result["rows"] == [[41], [42]]
    table = await mcp.call("describe_table", {"full_name": full_name})
    assert [column["name"] for column in table["columns"]] == ["id"]


async def test_seed_table_recreate_invalid_target_preserves_sql_created_table(mcp):
    """Reject a UC-invalid target name before recreate drops a SQL-created table."""
    full_name = "seedinvalidname.s.daily-sales"
    await mcp.call("setup_fixture", {"catalog": "seedinvalidname", "schema": "s"})
    await mcp.call(
        "run_sql",
        {"statement": 'CREATE TABLE "seedinvalidname"."s"."daily-sales" (id INTEGER)'},
    )
    await mcp.call(
        "run_sql",
        {"statement": 'INSERT INTO "seedinvalidname"."s"."daily-sales" VALUES (41), (42)'},
    )

    error = await mcp.call_expecting_error(
        "seed_table",
        {
            "full_name": full_name,
            "columns": [{"name": "id", "type_text": "INTEGER"}],
            "rows": [[43]],
            "recreate": True,
        },
    )

    assert "invalid table name 'daily-sales'" in error.lower()
    result = await mcp.call(
        "run_sql",
        {"statement": 'SELECT id FROM "seedinvalidname"."s"."daily-sales" ORDER BY id'},
    )
    assert result["rows"] == [[41], [42]]


async def test_seed_table_recreate_trailing_newline_preserves_sql_created_table(mcp):
    """Reject a trailing newline before recreate can drop a SQL-created table."""
    full_name = "seednewline.s.orders\n"
    await mcp.call("setup_fixture", {"catalog": "seednewline", "schema": "s"})
    await mcp.call(
        "run_sql",
        {"statement": 'CREATE TABLE "seednewline"."s"."orders\n" (id INTEGER)'},
    )
    await mcp.call(
        "run_sql",
        {"statement": 'INSERT INTO "seednewline"."s"."orders\n" VALUES (41), (42)'},
    )

    error = await mcp.call_expecting_error(
        "seed_table",
        {
            "full_name": full_name,
            "columns": [{"name": "id", "type_text": "INTEGER"}],
            "rows": [[43]],
            "recreate": True,
        },
    )

    assert "invalid table name" in error.lower()
    result = await mcp.call(
        "run_sql",
        {"statement": 'SELECT id FROM "seednewline"."s"."orders\n" ORDER BY id'},
    )
    assert result["rows"] == [[41], [42]]


async def test_seed_table_accepts_databricks_types(mcp):
    """Databricks type names are what a model will reach for first."""
    result = await mcp.call(
        "seed_table",
        {
            "full_name": "dbtypes.s.vendas",
            "columns": [
                {"name": "id", "type_text": "BIGINT"},
                {"name": "cliente", "type_text": "STRING"},
                {"name": "valor", "type_text": "DECIMAL(10,2)"},
                {"name": "dia", "type_text": "DATE"},
            ],
            "rows": [[1, "Ana", 10.5, "2026-07-01"]],
        },
    )
    assert result["rows_inserted"] == 1

    described = await mcp.call("describe_table", {"full_name": "dbtypes.s.vendas"})
    assert [c["type_name"] for c in described["columns"]] == ["LONG", "STRING", "DECIMAL", "DATE"]


async def test_describe_table_reports_types(mcp):
    """describe_table now reads columns straight from the UC response."""
    await mcp.call(
        "seed_table",
        {
            "full_name": "descr.s.t",
            "columns": [{"name": "a", "type_text": "INT"}],
            "rows": [[1]],
        },
    )

    described = await mcp.call("describe_table", {"full_name": "descr.s.t"})
    assert described["columns"][0]["name"] == "a"
    assert described["columns"][0]["type_text"] == "int"


async def test_seed_table_quotes_identifiers(mcp):
    """A table whose name collides with a SQL keyword must still work.

    seed_table interpolated the three-part name unquoted, so `desc.s.t` produced
    `DELETE FROM desc.s.t` and a parser error.
    """
    result = await mcp.call(
        "seed_table",
        {
            "full_name": "desc.order.select",
            "columns": [{"name": "a", "type_text": "INT"}],
            "rows": [[1]],
        },
    )
    assert result["rows_inserted"] == 1


async def test_setup_fixture_returns_a_usable_prefix(mcp):
    """setup_fixture yields a catalog, schema and warehouse ready to use."""
    result = await mcp.call("setup_fixture", {"catalog": "fixcat"})

    assert result["qualified_prefix"] == "fixcat.default"
    assert result["warehouse_id"]

    created = await mcp.call(
        "run_sql",
        {"statement": f"CREATE TABLE {result['qualified_prefix']}.t (x INTEGER)"},
    )
    assert created["state"] == "SUCCEEDED"


async def test_setup_fixture_is_idempotent(mcp):
    """Calling it twice on the same catalog must not fail on 'already exists'."""
    await mcp.call("setup_fixture", {"catalog": "idem"})
    again = await mcp.call("setup_fixture", {"catalog": "idem"})
    assert again["catalog"] == "idem"
