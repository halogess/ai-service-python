import argparse
import importlib
import os
import sys

from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import database


def get_engine_with_fallback():
    try:
        with database.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return database.engine
    except Exception:
        os.environ["DB_HOST"] = "localhost"
        importlib.reload(database)
        with database.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return database.engine


def table_exists(conn, table_name):
    row = conn.execute(
        text("SELECT 1 FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = :name"),
        {"name": table_name},
    ).fetchone()
    return row is not None


def print_ref_counts(conn, table_name, ref_tipe_column):
    if not table_exists(conn, table_name):
        print(f"- {table_name}: table not found")
        return

    rows = conn.execute(
        text(
            f"SELECT {ref_tipe_column} AS ref_tipe, COUNT(*) AS cnt "
            f"FROM {table_name} GROUP BY {ref_tipe_column} ORDER BY {ref_tipe_column}"
        )
    ).fetchall()

    if not rows:
        print(f"- {table_name}: empty")
        return

    print(f"- {table_name}:")
    for row in rows:
        print(f"  {row.ref_tipe}: {row.cnt}")


def print_before_after(conn, label):
    print(f"\n=== {label} ===")
    print_ref_counts(conn, "dokumen_section", "dsec_ref_tipe")
    print_ref_counts(conn, "dokumen_elemen_visual", "dev_ref_tipe")
    print_ref_counts(conn, "kesalahan", "kesalahan_ref_tipe")


def print_planned_changes(conn):
    print("\n=== Planned Changes ===")

    dsec_buku = conn.execute(
        text("SELECT COUNT(*) AS cnt FROM dokumen_section WHERE dsec_ref_tipe='buku'")
    ).scalar() or 0
    print(f"- dokumen_section buku->bab rows: {dsec_buku}")

    dev_buku = conn.execute(
        text("SELECT COUNT(*) AS cnt FROM dokumen_elemen_visual WHERE dev_ref_tipe='buku'")
    ).scalar() or 0
    print(f"- dokumen_elemen_visual buku->bab rows: {dev_buku}")

    if table_exists(conn, "kesalahan"):
        kesalahan_buku_rows = conn.execute(
            text("SELECT COUNT(*) AS cnt FROM kesalahan WHERE kesalahan_ref_tipe='buku'")
        ).scalar() or 0
        kesalahan_from_buku = conn.execute(
            text(
                "SELECT COUNT(*) AS cnt "
                "FROM kesalahan k "
                "JOIN bab b ON b.buku_id = k.kesalahan_ref_id "
                "WHERE k.kesalahan_ref_tipe='buku'"
            )
        ).scalar() or 0
        kesalahan_from_bab = conn.execute(
            text(
                "SELECT COUNT(*) AS cnt "
                "FROM kesalahan k "
                "JOIN bab b ON b.bab_id = k.kesalahan_ref_id "
                "WHERE k.kesalahan_ref_tipe='buku'"
            )
        ).scalar() or 0
        print(f"- kesalahan buku rows total: {kesalahan_buku_rows}")
        print(f"- kesalahan rows convertible via buku_id->bab_id expansion: {kesalahan_from_buku}")
        print(f"- kesalahan rows convertible via direct bab_id mapping: {kesalahan_from_bab}")
    else:
        print("- kesalahan: table not found")


def migrate_ref_types(conn):
    conn.execute(
        text("UPDATE dokumen_section SET dsec_ref_tipe='bab' WHERE dsec_ref_tipe='buku'")
    )
    conn.execute(
        text("UPDATE dokumen_elemen_visual SET dev_ref_tipe='bab' WHERE dev_ref_tipe='buku'")
    )


def migrate_kesalahan(conn):
    if not table_exists(conn, "kesalahan"):
        print("kesalahan table not found; skipping kesalahan migration")
        return

    # Expand buku-level rows into bab-level rows (one row per bab under the same buku).
    conn.execute(
        text(
            "INSERT INTO kesalahan (kesalahan_kategori, kesalahan_ref_tipe, kesalahan_ref_id, kesalahan_lokasi) "
            "SELECT k.kesalahan_kategori, 'bab', b.bab_id, k.kesalahan_lokasi "
            "FROM kesalahan k "
            "JOIN bab b ON b.buku_id = k.kesalahan_ref_id "
            "WHERE k.kesalahan_ref_tipe='buku' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM kesalahan k2 "
            "  WHERE k2.kesalahan_kategori = k.kesalahan_kategori "
            "    AND k2.kesalahan_ref_tipe='bab' "
            "    AND k2.kesalahan_ref_id = b.bab_id "
            "    AND (k2.kesalahan_lokasi <=> k.kesalahan_lokasi)"
            ")"
        )
    )

    # Remove the old buku-level rows that have been expanded.
    conn.execute(
        text(
            "DELETE k FROM kesalahan k "
            "JOIN bab b ON b.buku_id = k.kesalahan_ref_id "
            "WHERE k.kesalahan_ref_tipe='buku'"
        )
    )

    # Convert remaining legacy rows where ref_id already points to bab_id.
    conn.execute(
        text(
            "UPDATE kesalahan k "
            "JOIN bab b ON b.bab_id = k.kesalahan_ref_id "
            "SET k.kesalahan_ref_tipe='bab' "
            "WHERE k.kesalahan_ref_tipe='buku'"
        )
    )

    unresolved = conn.execute(
        text("SELECT COUNT(*) AS cnt FROM kesalahan WHERE kesalahan_ref_tipe='buku'")
    ).scalar() or 0
    print(f"kesalahan unresolved legacy buku rows after migration: {unresolved}")


def run(apply_changes):
    engine = get_engine_with_fallback()
    with engine.begin() as conn:
        print_before_after(conn, "Before")
        print_planned_changes(conn)

        if not apply_changes:
            print("\nDry-run only. Re-run with --apply to execute migration.")
            return

        migrate_ref_types(conn)
        migrate_kesalahan(conn)
        print_before_after(conn, "After")
        print("\nMigration applied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate ref_tipe buku -> bab for chapter-linked tables")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, script runs in dry-run mode.",
    )
    args = parser.parse_args()
    run(apply_changes=args.apply)
