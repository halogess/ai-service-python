import os
import sys

from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import engine


def ensure_column():
    with engine.begin() as conn:
        result = conn.execute(
            text("SHOW COLUMNS FROM dokumen LIKE 'dokumen_images_path'")
        )
        if result.fetchone():
            return False
        conn.execute(
            text(
                "ALTER TABLE dokumen "
                "ADD COLUMN dokumen_images_path varchar(255) DEFAULT NULL"
            )
        )
        return True


def backfill_images_path():
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE dokumen "
                "SET dokumen_images_path = CONCAT('/dokumen/', mhs_nrp, '/', dokumen_id, '/images') "
                "WHERE dokumen_images_path IS NULL OR dokumen_images_path = ''"
            )
        )


if __name__ == "__main__":
    added = ensure_column()
    if added:
        print("Added column dokumen_images_path")
    else:
        print("Column dokumen_images_path already exists")
    backfill_images_path()
    print("Backfilled dokumen_images_path")
