import os
import psycopg

conn = psycopg.connect(os.environ["DATABASE_URL"])

try:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS legacy")
        cur.execute("ALTER TABLE public.subscriptions SET SCHEMA legacy")

    conn.commit()
    print("Moved public.subscriptions to legacy.subscriptions")

except Exception:
    conn.rollback()
    raise

finally:
    conn.close()
