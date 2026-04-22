"""no-op: bridge for a2b3c4d5e6f7 (uncommitted 1536→768 migration)

This migration is a no-op placeholder to satisfy alembic version tracking.
The actual 1536→768 dimension change was applied manually this morning
and is reflected in the current codebase.

Revision ID: a2b3c4d5e6f7
Revises: 119a52b73c60
Create Date: 2026-04-22 09:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a2b3c4d5e6f7"
down_revision: str | None = "119a52b73c60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: database already at 768 dimensions."""
    pass


def downgrade() -> None:
    """No-op."""
    pass
