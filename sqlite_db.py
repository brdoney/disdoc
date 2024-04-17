import sqlite3
from typing import Literal, NamedTuple

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
) -> int | None:
    """Logs a post and returns its ID in the database if a user has given consent,
    otherwise does nothing and returns `None`."""
    if not author.consent:
        return None

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


def _update_grade(
    cur: sqlite3.Cursor,
    user: UserInfo,
    col: Literal["llm_reviews"] | Literal["retrieval_reviews"],
):
    # Use sub-query so we don't store PID anywhere
    query = f"""
    UPDATE grading 
        SET {col} = {col} + 1
        WHERE pid IN (SELECT pid FROM users WHERE id = ?)
    """
    # Alternative using UPDATE FROM
    # query = f"""
    # UPDATE grading
    #     SET {col} = {col} + 1
    #     FROM (SELECT pid, id FROM users WHERE id = ?) as u
    #     WHERE grading.pid = u.pid
    # """
    _ = cur.execute(query, (user.id,))


def log_llm_review(
    post_id: int | None,
    author: UserInfo,
    relevance: int,
    helpfulness: int,
    correctness: int,
):
    """Log an LLM review by updating a user's grade and, if they gave consent, storing its contents."""
    cur = db.cursor()
    if post_id is not None:
        assert author.consent, "Trying to log LLM review for user who denied consent"
        _ = cur.execute(
            "INSERT OR REPLACE INTO llm_reviews (post_id, author, relevance, helpfulness, correctness) VALUES (?, ?, ?, ?, ?)",
            (post_id, author.id, relevance, helpfulness, correctness),
        )
    _update_grade(cur, author, "llm_reviews")

    db.commit()

    print(f"Logged LLM review for post {post_id}")


def log_retrieval_review(
    post_id: int | None, author: UserInfo, relevance: int, helpfulness: int
):
    """Log an retrieval review by updating a user's grade and, if they gave consent, storing its contents."""
    cur = db.cursor()
    if post_id is not None:
        assert (
            author.consent
        ), "Trying to log retrieval review for user who denied consent"
        _ = cur.execute(
            "INSERT OR REPLACE INTO retrieval_reviews (post_id, author, relevance, helpfulness) VALUES (?, ?, ?, ?)",
            (post_id, author.id, relevance, helpfulness),
        )
    _update_grade(cur, author, "retrieval_reviews")

    db.commit()

    print(f"Logged retrieval review for post {post_id}")
