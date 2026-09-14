import os
import psycopg

conn = psycopg.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
SELECT *
FROM schema_migrations
ORDER BY 1
""")

print("MIGRATIONS:")
for row in cur.fetchall():
    print(row)

print("\nPUBLIC ACCOUNT TABLES:")
cur.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN (
    'users',
    'subscriptions',
    'entitlement_overrides',
    'stripe_events'
  )
ORDER BY table_name
""")

for row in cur.fetchall():
    print(row[0])

print("\nLEGACY:")
cur.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'legacy'
ORDER BY table_name
""")

for row in cur.fetchall():
    print(row[0])

conn.close()
