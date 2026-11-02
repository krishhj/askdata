import sys
from pathlib import Path

from sqlalchemy import create_engine, text

from askdata.config import require_env

RAW_DATA = Path(__file__).resolve().parent.parent / "raw_data"

# Listed parents first, because foreign keys need the parent table to exist.
CSV_FILES = {
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
}

# Zip prefixes are text because numeric columns would lose leading zeros.
# Money is numeric(10, 2) because floats would make totals drift by fractions of a cent.
CREATE_TABLES = """
CREATE TABLE customers (
    customer_id text,
    customer_unique_id text NOT NULL,
    customer_zip_code_prefix text,
    customer_city text,
    customer_state text
);
CREATE TABLE sellers (
    seller_id text,
    seller_zip_code_prefix text,
    seller_city text,
    seller_state text
);
CREATE TABLE products (
    product_id text,
    product_category_name text,
    product_name_lenght integer,
    product_description_lenght integer,
    product_photos_qty integer,
    product_weight_g integer,
    product_length_cm integer,
    product_height_cm integer,
    product_width_cm integer
);
CREATE TABLE category_translation (
    product_category_name text,
    product_category_name_english text NOT NULL
);
CREATE TABLE orders (
    order_id text,
    customer_id text NOT NULL,
    order_status text NOT NULL,
    order_purchase_timestamp timestamp NOT NULL,
    order_approved_at timestamp,
    order_delivered_carrier_date timestamp,
    order_delivered_customer_date timestamp,
    order_estimated_delivery_date timestamp
);
CREATE TABLE order_items (
    order_id text,
    order_item_id integer,
    product_id text NOT NULL,
    seller_id text NOT NULL,
    shipping_limit_date timestamp,
    price numeric(10, 2) NOT NULL,
    freight_value numeric(10, 2) NOT NULL
);
CREATE TABLE order_payments (
    order_id text,
    payment_sequential integer,
    payment_type text NOT NULL,
    payment_installments integer,
    payment_value numeric(10, 2) NOT NULL
);
CREATE TABLE order_reviews (
    review_id text,
    order_id text,
    review_score smallint NOT NULL,
    review_comment_title text,
    review_comment_message text,
    review_creation_date timestamp,
    review_answer_timestamp timestamp
);
"""

# review_id repeats across orders in the source data, so its key is composite.
# order_items.order_id and order_payments.order_id get no separate index because
# they already lead their composite primary keys.
ADD_CONSTRAINTS = """
ALTER TABLE customers ADD PRIMARY KEY (customer_id);
ALTER TABLE sellers ADD PRIMARY KEY (seller_id);
ALTER TABLE products ADD PRIMARY KEY (product_id);
ALTER TABLE category_translation ADD PRIMARY KEY (product_category_name);
ALTER TABLE orders ADD PRIMARY KEY (order_id);
ALTER TABLE order_items ADD PRIMARY KEY (order_id, order_item_id);
ALTER TABLE order_payments ADD PRIMARY KEY (order_id, payment_sequential);
ALTER TABLE order_reviews ADD PRIMARY KEY (review_id, order_id);

ALTER TABLE orders ADD FOREIGN KEY (customer_id) REFERENCES customers;
ALTER TABLE order_items ADD FOREIGN KEY (order_id) REFERENCES orders;
ALTER TABLE order_payments ADD FOREIGN KEY (order_id) REFERENCES orders;
ALTER TABLE order_reviews ADD FOREIGN KEY (order_id) REFERENCES orders;
ALTER TABLE order_items ADD FOREIGN KEY (product_id) REFERENCES products;
ALTER TABLE order_items ADD FOREIGN KEY (seller_id) REFERENCES sellers;

CREATE INDEX ON orders (customer_id);
CREATE INDEX ON orders (order_purchase_timestamp);
CREATE INDEX ON order_items (product_id);
CREATE INDEX ON order_items (seller_id);
CREATE INDEX ON order_reviews (order_id);
CREATE INDEX ON products (product_category_name);
"""


def copy_csv(conn, table: str, path: Path) -> None:
    """Stream one CSV file into a table with COPY."""
    with conn.connection.cursor() as cursor:
        with cursor.copy(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)") as copy:
            with path.open("rb") as csv_file:
                while chunk := csv_file.read(1024 * 1024):
                    copy.write(chunk)


def run_statements(conn, sql: str) -> None:
    for statement in sql.split(";"):
        if statement.strip():
            conn.execute(text(statement))


def main() -> int:
    missing = [name for name in CSV_FILES.values() if not (RAW_DATA / name).exists()]
    if missing:
        print(f"Missing from {RAW_DATA}: {', '.join(missing)}")
        return 1

    engine = create_engine(require_env("ADMIN_DB_URL"))
    # One transaction: if any step fails, the database keeps its previous state.
    with engine.begin() as conn:
        for table in reversed(CSV_FILES):
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        run_statements(conn, CREATE_TABLES)
        for table, file_name in CSV_FILES.items():
            print(f"Loading {table}")
            copy_csv(conn, table, RAW_DATA / file_name)
        # Loading before adding keys is faster, and it checks every row at once.
        print("Adding keys and indexes")
        run_statements(conn, ADD_CONSTRAINTS)
        conn.execute(text("ANALYZE"))

        print("\nRow counts")
        for table in CSV_FILES:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            print(f"{table:<22}{count:>8,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
