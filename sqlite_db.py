import sqlite3
from typing import NamedTuple

from typing_extensions import overload

from env_var import SQLITE_DB

db = sqlite3.connect(SQLITE_DB)
cur = db.cursor()

# Enforce foreign key constraints
_ = cur.execute("PRAGMA foreign_keys = ON")


@overload
def convert_consent(consent: bool) -> str: ...


@overload
def convert_consent(consent: str) -> bool: ...


def convert_consent(consent: bool | str) -> str | bool:
    if isinstance(consent, str):
        return True if consent == "TRUE" else False
    else:
        return "TRUE" if consent else "FALSE"


class UserInfo(NamedTuple):
    id: int
    consent: bool


def check_consent(discord_id: int) -> UserInfo | None:
    # DB stores them as strings, since that's what Discord's API recommends
    res: tuple[int, str] | None = cur.execute(
        "SELECT id, consent FROM users WHERE discord_id=?", (discord_id,)
    ).fetchone()

    if res is None:
        return None

    id, consent_str = res
    return UserInfo(id, convert_consent(consent_str))


def log_post(post_id: int, author: UserInfo, use_llm: bool) -> int:
    """Logs a post and returns its ID in the database. Requires that the author gave consent."""
    assert author.consent

    _ = cur.execute(
        "INSERT INTO posts (post_id, author, use_llm) VALUES (?, ?, ?)",
        (post_id, author.id, convert_consent(use_llm)),
    )
    db.commit()

    db_post_id = cur.lastrowid
    if db_post_id is None:
        raise RuntimeError(
            "Unable to read post of last ID. Are you sharing the same cursor across threads?"
        )

    print(f"Logged post {db_post_id} by {author}")

    return db_post_id


def log_llm_review(
    post_id: int, author: UserInfo, relevance: int, helpfulness: int, correctness: int
):
    """Logs a review of an LLM response. Requires that the author gave consent."""
    assert author.consent

    _ = cur.execute(
        "INSERT INTO llm_reviews (post_id, author, relevance, helpfulness, correctness) VALUES (?, ?, ?, ?, ?)",
        (post_id, author.id, relevance, helpfulness, correctness),
    )
    db.commit()

    print(f"Logged LLM review for post {post_id} by {author}")


def log_retrieval_review(
    post_id: int, author: UserInfo, relevance: int, helpfulness: int
):
    """Logs a review of a retrieval response. Requires that the author gave consent."""
    assert author.consent

    _ = cur.execute(
        "INSERT INTO retrieval_reviews (post_id, author, relevance, helpfulness) VALUES (?, ?, ?, ?)",
        (post_id, author.id, relevance, helpfulness),
    )
    db.commit()

    print(f"Logged retrieval review for post {post_id} by {author}")
