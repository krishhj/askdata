-- Run after scripts/load_data.py, in the Supabase SQL Editor.
-- Replace the two passwords in the editor only. Never save real passwords in this file.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'askdata_reader') THEN
        CREATE ROLE askdata_reader;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'askdata_feedback') THEN
        CREATE ROLE askdata_feedback;
    END IF;
END
$$;

ALTER ROLE askdata_reader LOGIN PASSWORD 'READER_PASSWORD_HERE';
ALTER ROLE askdata_feedback LOGIN PASSWORD 'FEEDBACK_PASSWORD_HERE';

-- Reader
GRANT CONNECT ON DATABASE postgres TO askdata_reader;
GRANT USAGE ON SCHEMA public TO askdata_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO askdata_reader;
-- load_data.py recreates the tables, and new tables would otherwise lose this grant.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO askdata_reader;
ALTER ROLE askdata_reader SET default_transaction_read_only = on;
ALTER ROLE askdata_reader SET statement_timeout = '5s';

-- Feedback
CREATE SCHEMA IF NOT EXISTS app;
CREATE TABLE IF NOT EXISTS app.feedback (
    id serial PRIMARY KEY,
    created_at timestamptz DEFAULT now(),
    question text,
    sql text,
    rating smallint
);
GRANT CONNECT ON DATABASE postgres TO askdata_feedback;
GRANT USAGE ON SCHEMA app TO askdata_feedback;
GRANT INSERT, SELECT ON app.feedback TO askdata_feedback;
-- Inserting into a serial column draws from its sequence, which needs its own grant.
GRANT USAGE ON SEQUENCE app.feedback_id_seq TO askdata_feedback;

-- Supabase's public API can reach tables in public through these roles.
-- The app never uses that API, so close it.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM anon, authenticated;
