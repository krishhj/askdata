import os

from dotenv import load_dotenv

# Streamlit Cloud exposes top-level secrets as environment variables, so this
# one lookup works in both places and the package never needs to import streamlit.
load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0
# Room for the model's reasoning plus the JSON reply. Groq counts this limit
# against the free tier's 8,000 tokens per minute before the call runs, so a
# larger value makes long prompts fail with a rate limit however long we wait.
MAX_TOKENS = 2048
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K_TABLES = 4
TOP_K_EXAMPLES = 3
MAX_RETRIES = 2
ROW_LIMIT = 500
HISTORY_TURNS = 2
SUMMARY_ROWS = 20

BAR_MAX_ROWS = 25
# Bar labels longer than this switch the bar chart to horizontal.
LONG_LABEL_CHARS = 12

# Read these as config.USE_..., not with "from config import", so the
# evaluation can switch them at runtime.
USE_TABLE_RETRIEVAL = True
USE_EXAMPLES = True
USE_SCHEMA_NOTES = True

# Evaluation. Extra columns let an answer that adds, say, a count next to the
# requested label still match; set False for a strict column-for-column match.
EVAL_ALLOW_EXTRA_COLUMNS = True
# Each call uses about 2,000 to 3,000 tokens, so this keeps a run near the
# free tier's 8,000 tokens per minute.
EVAL_SLEEP_SECONDS = 15
EVAL_RATE_LIMIT_WAIT_SECONDS = 60
EVAL_RATE_LIMIT_TRIES = 5

# The Supabase pooler allows few connections, so each engine keeps a small pool.
DB_POOL_SIZE = 2

ALLOWED_TABLES = {
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "customers",
    "products",
    "sellers",
    "category_translation",
}
DENIED_FUNCTIONS = {
    "pg_sleep",
    "pg_read_file",
    "pg_ls_dir",
    "dblink",
    "lo_import",
    "lo_export",
    "set_config",
    # These run a SQL string inside the database, out of the validator's sight.
    "query_to_xml",
    "query_to_xmlschema",
    "query_to_xml_and_xmlschema",
    "cursor_to_xml",
    "table_to_xml",
}
# Prefixes for system, large object and remote connection functions.
DENIED_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")


def require_env(name: str) -> str:
    """Return a required setting, or fail with a message naming the missing one."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to .env or the Streamlit secrets.")
    return value
