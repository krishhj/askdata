import json
import sys

from groq import Groq
from sqlalchemy import create_engine, text

from askdata.config import GROQ_MODEL, TEMPERATURE, require_env


def check_database() -> None:
    # The reader role does not exist until Phase 1, so this uses the admin URL.
    engine = create_engine(require_env("ADMIN_DB_URL"))
    with engine.connect() as conn:
        user, version = conn.execute(text("SELECT current_user, version()")).one()
    print(f"Connected as {user}: {version}")


def check_groq() -> None:
    client = Groq(api_key=require_env("GROQ_API_KEY"))
    # JSON mode is what the pipeline relies on, so test it rather than plain text.
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        # A reasoning model spends tokens thinking first, so a tiny limit can leave no reply.
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": 'Reply with this JSON only: {"status": "ok"}'}],
    )
    reply = json.loads(response.choices[0].message.content)
    print(f"{GROQ_MODEL} replied: {reply}")


def main() -> int:
    failed = False
    # Run both checks even if the first fails, so one run shows every problem.
    for check in (check_database, check_groq):
        try:
            check()
        except Exception as exc:
            failed = True
            print(f"{check.__name__} failed: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
