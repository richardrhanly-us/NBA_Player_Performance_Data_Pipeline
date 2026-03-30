import os
import psycopg

def get_db_connection():
    return psycopg.connect(os.environ["DATABASE_URL"])

from datetime import datetime
from zoneinfo import ZoneInfo

def insert_line_snapshot(player_name, game_date, line, sportsbook):
    insert_sql = """
    INSERT INTO line_snapshots (
        player_name, game_date, sportsbook_line, sportsbook, captured_at
    ) VALUES (
        %s, %s, %s, %s, %s
    );
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                insert_sql,
                (
                    player_name,
                    game_date,
                    line,
                    sportsbook,
                    datetime.now(ZoneInfo("America/Chicago")),
                ),
            )
        conn.commit()
