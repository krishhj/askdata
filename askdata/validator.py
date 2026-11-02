import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from askdata import config

UNSAFE = "unsafe"
INVALID = "invalid"

ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create,
    exp.Alter, exp.TruncateTable, exp.Command, exp.Grant, exp.Revoke,
    exp.Copy, exp.Into, exp.Lock,
)
BLOCKED_SCHEMAS = {"pg_catalog", "information_schema"}

Result = tuple[bool, str | None, str | None, str | None]


def _function_name(node: exp.Func) -> str:
    # Functions sqlglot does not know become Anonymous and keep their own name.
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return node.sql_name().lower()


def _check_functions(tree: exp.Expression) -> str | None:
    for node in tree.find_all(exp.Func):
        name = _function_name(node)
        if name in config.DENIED_FUNCTIONS or name.startswith(config.DENIED_FUNCTION_PREFIXES):
            return f"The function {name} is not allowed."
    return None


def _check_tables(tree: exp.Expression) -> tuple[str, str] | None:
    # Returns (category, reason) for the first table that is not allowed.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name, schema = table.name.lower(), table.db.lower()
        # Set-returning functions such as generate_series appear as nameless tables.
        if not name:
            continue
        if schema in BLOCKED_SCHEMAS or name.startswith("pg_"):
            return UNSAFE, f"System catalogs such as {schema or name} are not allowed."
        if table.catalog or schema not in ("", "public"):
            return INVALID, f"Use plain table names, not {table.sql()}."
        if name not in config.ALLOWED_TABLES and name not in cte_names:
            allowed = ", ".join(sorted(config.ALLOWED_TABLES))
            return INVALID, f"Table {name} does not exist. Available tables: {allowed}."
    return None


def _parse_one(sql: str) -> tuple[exp.Expression | None, str | None, str | None]:
    # Returns the single statement, or (None, category, reason).
    try:
        trees = [t for t in sqlglot.parse(sql, read="postgres") if t is not None]
    except ParseError as exc:
        return None, INVALID, f"The SQL could not be parsed: {str(exc).splitlines()[0]}"
    if not trees:
        return None, INVALID, "The query is empty."
    if len(trees) > 1:
        return None, UNSAFE, "Only one statement is allowed."
    return trees[0], None, None


def validate(sql: str) -> Result:
    """Check that sql is one safe read-only query: (ok, cleaned_sql, category, reason)."""
    tree, category, reason = _parse_one(sql or "")
    if tree is None:
        return False, None, category, reason

    # Checked across the whole tree, because a write can hide inside a CTE.
    forbidden = next(tree.find_all(*FORBIDDEN_NODES), None)
    if forbidden is not None:
        return False, None, UNSAFE, "Only read-only SELECT queries are allowed."
    if not isinstance(tree, ALLOWED_ROOTS):
        return False, None, UNSAFE, "Only SELECT queries are allowed."

    reason = _check_functions(tree)
    if reason:
        return False, None, UNSAFE, reason
    table_problem = _check_tables(tree)
    if table_problem:
        return False, None, *table_problem

    if not tree.args.get("limit"):
        tree = tree.limit(config.ROW_LIMIT)
    # Regenerate from the parsed tree, so what runs is exactly what was checked.
    return True, tree.sql(dialect="postgres", pretty=True), None, None
