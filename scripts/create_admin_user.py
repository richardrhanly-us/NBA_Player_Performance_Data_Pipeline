"""
Step 11: one-time bootstrap -- grants ADMIN to a single user account via
an entitlement_overrides row (the same mechanism apps/adminapp.py's
"Users & Access" tab uses for grant_pro_override, just with
override_tier=ADMIN and no expiry).

This is a MANUAL, human-invoked script -- never called automatically.
Run it once against the real database, after the target account has
signed in at least once (so a users row exists) OR let it create one
directly (useful for the very first admin, before anyone else can grant
you access).

Usage:
    # Supabase-authenticated account (sign in once first, then look up
    # its Supabase user id, e.g. from the Supabase dashboard):
    python scripts/create_admin_user.py --auth-provider supabase \\
        --auth-subject <supabase-user-uuid> --email you@example.com

    # Dev-mode account (no Supabase configured) -- subject is derived
    # deterministically from the email, matching DevAuthProvider:
    python scripts/create_admin_user.py --auth-provider dev \\
        --email you@example.com
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services import accounts_repository
from src.services.auth_providers import compute_dev_auth_subject
from src.services.db_connection import get_prediction_db_connection


def log(msg):
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-provider", required=True, choices=["supabase", "dev"])
    parser.add_argument("--email", required=True)
    parser.add_argument(
        "--auth-subject",
        help="Required for --auth-provider supabase; derived automatically for dev.",
    )
    parser.add_argument("--display-name", default=None)
    args = parser.parse_args()

    if args.auth_provider == "dev":
        auth_subject = compute_dev_auth_subject(args.email)
    else:
        if not args.auth_subject:
            parser.error("--auth-subject is required for --auth-provider supabase")
        auth_subject = args.auth_subject

    if not os.environ.get("DATABASE_URL"):
        raise ValueError("DATABASE_URL not found in environment.")

    conn = get_prediction_db_connection()
    try:
        user = accounts_repository.get_or_create_user_by_auth_subject(
            conn,
            auth_provider=args.auth_provider,
            auth_subject=auth_subject,
            email=args.email,
            display_name=args.display_name,
        )
        override_id = accounts_repository.create_override(
            conn,
            user_id=user["id"],
            override_tier="ADMIN",
            reason="bootstrap via scripts/create_admin_user.py",
            created_by="create_admin_user.py",
            expires_at=None,
        )
    finally:
        conn.close()

    log(f"[ADMIN BOOTSTRAP] user id={user['id']} email={user['email']} -> ADMIN")
    log(f"[ADMIN BOOTSTRAP] entitlement_overrides id={override_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
