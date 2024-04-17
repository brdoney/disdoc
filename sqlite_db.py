import sqlite3
from typing import NamedTuple

from typing_extensions import overload

from env_var import SQLITE_DB
from llm import LLMType

db = sqlite3.connect(SQLITE_DB)

# Enforce foreign key constraints
_ = db.execute("PRAGMA foreign_keys = ON")


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
    res: tuple[int, str] | None = db.execute(
        "SELECT id, consent FROM users WHERE discord_id=?", (discord_id,)
    ).fetchone()

    if res is None:
        return None

    id, consent_str = res
    return UserInfo(id, convert_consent(consent_str))


def log_post(
    post_discord_id: int, author: UserInfo, use_llm: bool, llm_type: LLMType
) -> int:
    """Logs a post and returns its ID in the database. Requires that the author gave consent."""
    assert author.consent

    cur = db.execute(
        "INSERT INTO posts (post_id, author, use_llm, llm_type) VALUES (?, ?, ?, ?)",
        (post_discord_id, author.id, convert_consent(use_llm), llm_type.name),
    )
    db.commit()

    db_post_id = cur.lastrowid
    cur.close()

    if db_post_id is None:
        raise RuntimeError(
            "Unable to read post of last ID. Are you sharing the same cursor across threads?"
        )

    print(f"Logged post {db_post_id} by {author} -> {db_post_id}")

    return db_post_id


def log_post_times(post_id: int, retrieval_time: float, generation_time: float | None):
    if generation_time is None:
        _ = db.execute(
            "UPDATE posts SET retrieval_time=? WHERE post_id=?",
            (retrieval_time, post_id),
        )
    else:
        _ = db.execute(
            "UPDATE posts SET retrieval_time=?, generation_time=? WHERE id=?",
            (retrieval_time, generation_time, post_id),
        )
    db.commit()

    print(f"Updated times for {post_id}")


def log_llm_review(
    post_id: int, author: UserInfo, relevance: int, helpfulness: int, correctness: int
):
    """Logs a review of an LLM response. Requires that the author gave consent."""
    assert author.consent

    _ = db.execute(
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

    _ = db.execute(
        "INSERT INTO retrieval_reviews (post_id, author, relevance, helpfulness) VALUES (?, ?, ?, ?)",
        (post_id, author.id, relevance, helpfulness),
    )
    db.commit()

    print(f"Logged retrieval review for post {post_id} by {author}")
