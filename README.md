# AskData

Ask a business question in plain English and get back an answer, a chart, the result table and the SQL behind it.

I built AskData on the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): about 100,000 orders placed between September 2016 and October 2018, with their items, payments, reviews, customers, products and sellers. You type something like "Which customer states have the highest share of late deliveries?" and the app writes one PostgreSQL query, checks it is safe, runs it on a read-only database user, and shows the result. Every answer can be inspected and the SQL can be edited and run again.

**Live demo:** 

## What it does

- Turns a question into a single `SELECT` query using an LLM (`openai/gpt-oss-120b` on Groq).
- Gives the model only the schema it needs: table descriptions, notes on the data's traps, and similar solved examples chosen by embedding similarity.
- Checks every query with a SQL parser before it runs, and runs it as a database user that cannot write.
- Retries up to twice when a query fails, sending the model its own SQL and the database error.
- Shows a one-sentence summary, a chart picked from the shape of the result, the table, a CSV download, and a "How I got this" panel with the SQL, explanation and assumptions.
- Lets you edit the SQL and run it again, through the same safety checks.
- Supports follow-up questions such as "now only for 2018".
- Has an Evaluation page with results from a 60-question test set.

## How it works

```
question
   |
   v
retrieval ----> pick relevant tables (plus the tables they join to)
   |            and the 3 most similar solved examples
   v
generator ----> prompt = rules + schema notes + examples + last 2 turns + question
   |            Groq in JSON mode returns {sql, explanation, assumptions}
   v
validator ----> parse with sqlglot; reject writes, extra statements, system
   |            catalogs and dangerous functions; add LIMIT 500
   |   unsafe -> stop and show why
   |   invalid -> retry with the reason
   v
executor -----> run as askdata_reader (read-only, 5 s timeout)
   |   database error -> retry with the error
   v
summary + chart + table
```

The code is plain Python functions in `askdata/`, with no framework:

| Module | Job |
|---|---|
| `config.py` | Every setting and feature flag in one place |
| `schema.py` | Loads `data/schema.yaml` and builds the schema text for the prompt |
| `retrieval.py` | Embeds tables and examples with `all-MiniLM-L6-v2`; cosine similarity with numpy |
| `generator.py` | Builds the prompt, calls Groq, parses the reply, writes the summary |
| `validator.py` | Decides whether a query is safe, invalid or fine to run |
| `executor.py` | Runs SQL as the read-only user and returns a DataFrame |
| `charts.py` | Chooses a metric, line, bar, scatter or no chart |
| `pipeline.py` | Ties it together with the retry loop; also edited SQL and feedback |

The package never imports Streamlit. `app.py` and `pages/1_Evaluation.py` are the only UI code.

### Why the schema notes matter

The Olist data has traps that make plausible SQL wrong. I wrote them into `data/schema.yaml` so the model sees them:

- `customer_id` is unique per order, so counting people needs `customer_unique_id`.
- Revenue is `SUM(order_items.price)`. `payment_value` differs because of vouchers and instalments.
- Category names are in Portuguese and two have no English translation, so the translation needs a `LEFT JOIN`.
- Joining items, payments and reviews to each other multiplies rows.
- The data ends in 2018, so "last month" means the month before the latest order, not today.

## Safety

I treat the model's SQL as untrusted and check it in two independent layers:

1. **Validator.** `sqlglot` parses the query. It rejects anything that is not exactly one `SELECT`/`WITH`/`UNION` query, any write or DDL node anywhere in the tree (including a `DELETE` hidden in a CTE), `SELECT INTO`, `FOR UPDATE`, `pg_catalog` and `information_schema`, and functions such as `pg_sleep`, `dblink` and `query_to_xml`. What runs is regenerated from the parsed tree, so it is exactly what was checked.
2. **Database role.** Queries run as `askdata_reader`, which has only `SELECT` grants, read-only transactions by default and a 5-second statement timeout. Feedback is written by a separate role that can only insert into `app.feedback`.

Unsafe queries are never retried, so the model is never asked to find a way past the validator. The tests check each layer on its own, and `eval/attacks.yaml` holds 24 hostile inputs that the evaluation runs through the full pipeline while checking that no table changes.

## Results

See the Evaluation page in the app. It shows accuracy for five setups that add one feature at a time (schema notes, examples, retries, table retrieval), accuracy by difficulty, latency, token use, the attack results, and every question the model got wrong next to the gold SQL.

A generated query counts as correct when it returns the same rows as a hand-checked gold query, ignoring column names, column order and float noise beyond two decimals.

## Run it locally

You need Python 3.11, a Postgres database (I use Supabase's free tier) and a free Groq API key.

1. Create an environment and install the packages:

   ```bash
   conda create -n askdata python=3.11 -y
   conda activate askdata
   pip install -r requirements.txt
   ```

2. Download the Olist CSVs from Kaggle into `raw_data/`.

3. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY` and `ADMIN_DB_URL`. With Supabase, use the session pooler connection string and change `postgresql://` to `postgresql+psycopg://`.

4. Check the connections, then load the data:

   ```bash
   python -m scripts.smoke_test
   python -m scripts.load_data
   ```

5. Run `scripts/setup_roles.sql` in the Supabase SQL Editor, replacing the two password placeholders in the editor only. Add `READER_DB_URL` and `FEEDBACK_DB_URL` to `.env` (the user names are `askdata_reader.<project_ref>` and `askdata_feedback.<project_ref>`), then check the roles:

   ```bash
   python -m scripts.check_roles
   ```

6. Run the tests and start the app:

   ```bash
   pytest
   streamlit run app.py
   ```

To ask one question from the terminal: `python -m scripts.ask "How many orders were canceled?"`

### Running the evaluation

```bash
python -m eval.run_eval
```

It refuses to start until every entry in `eval/gold.yaml` is marked verified. Answers are saved one at a time, so if Groq's daily limit stops a run, the same command picks up where it left off. `--setups E` runs one setup and `--limit 5` makes a trial run that writes to `eval/results/trial/`.

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub. `.env`, `raw_data/` and `.streamlit/secrets.toml` are git-ignored; `eval/results/` is committed so the Evaluation page has data.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repository with `app.py` as the main file. Under Advanced settings, choose Python 3.11.
3. In the app's Secrets, paste the contents of `.streamlit/secrets.toml.example` with real values. Keep the keys at the top level: Streamlit Cloud exposes them as environment variables, which is how `askdata/config.py` reads them. Do not add `ADMIN_DB_URL`; the deployed app only needs the reader and feedback users.
4. The first start installs the packages and downloads the embedding model, which takes a few minutes. `requirements.txt` points pip at PyTorch's CPU index so the server does not download the much larger GPU build.

Two things to know about the free tiers:

- Streamlit Cloud puts an app to sleep after a period without visitors. Opening it wakes it in about a minute, so open it before showing it to anyone.
- Supabase pauses free projects after a week without activity. If the app cannot connect, restore the project from the Supabase dashboard.

## Project structure

```
app.py                  Streamlit chat app
pages/1_Evaluation.py   Evaluation results page
askdata/                The pipeline (no Streamlit imports)
data/schema.yaml        Table descriptions, column notes and data traps
data/examples.yaml      26 solved question and SQL pairs used as few-shot examples
scripts/                Smoke test, data loader, role setup and checks, one-off questions
eval/                   Gold questions, attacks, result comparison and the runner
tests/                  Validator, attacks, executor, retrieval, charts, pipeline, comparison, leakage
```

## Limitations

- **Free-tier budget.** Groq's free tier allows 200,000 tokens a day for this model, which is roughly 70 to 80 questions. A full evaluation needs several days of that budget, and the live demo shares it.
- **One dataset.** The schema notes and examples are written for Olist. A new database needs its own notes, examples and gold questions.
- **Answers vary between runs.** Even at temperature 0, the hosted model sometimes writes different SQL for the same question. Both versions are often correct, but accuracy moves a little between runs.
- **Execution accuracy has blind spots.** Two different queries can return the same rows on this data by coincidence, and a correct answer phrased differently (for example quarter numbers instead of quarter dates) counts as wrong.
- **Retrieval pulls in a lot.** `orders` joins to four tables, so whenever it is retrieved most of the schema comes with it.
- **Strict summaries.** The summary may only use numbers that appear in the result. When the model breaks that rule, the app falls back to a plain sentence such as "The query returned 12 rows."

## Future work

- Run the full ablation regularly and track accuracy over time.
- Use the thumbs-up and thumbs-down feedback to find weak spots and grow the example set.
- Ask a clarifying question when a question is ambiguous instead of guessing and listing assumptions.
- Cache answers to repeated questions to save tokens.
- Try a second dataset to see how much of the schema-notes approach carries over.
