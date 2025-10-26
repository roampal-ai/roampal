#!/usr/bin/env python3
"""
Migration script to remove legacy tables from book database.
Run this once to clean up existing databases.

Usage: python scripts/migrate_book_database.py
"""
import sqlite3
import sys
from pathlib import Path

def migrate_database():
    """Remove legacy quotes, models, and summaries tables"""
    db_path = Path("data/books/books.db")

    if not db_path.exists():
        print(f"[OK] Database not found at {db_path} - nothing to migrate")
        return True

    print(f"Migrating database: {db_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if legacy tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        legacy_tables = ['quotes', 'models', 'summaries']
        tables_to_drop = [t for t in legacy_tables if t in tables]

        if not tables_to_drop:
            print("[OK] No legacy tables found - database is already clean")
            conn.close()
            return True

        # Drop legacy tables
        for table in tables_to_drop:
            print(f"  Dropping table: {table}")
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        conn.commit()
        conn.close()

        print(f"[OK] Successfully removed {len(tables_to_drop)} legacy tables")
        return True

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)