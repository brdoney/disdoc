import sqlite3


def check_consent(db: sqlite3.Cursor, discord_id: int) -> bool | None:
    # DB stores them as strings, since that's what Discord's API recommends
    str_id = str(discord_id)
    res = db.execute(  # type: ignore[reportAny]
        "SELECT consent FROM consent WHERE discord_id=?", (str_id,)
    ).fetchone()

    if res is None:
        return None
    consent_status = True if res == "TRUE" else False
    return consent_status
