# -*- coding: utf-8 -*-
"""
objects_db.py — таблица objects (стройплощадки) в SQLite + синхронизация из конфига.
"""
import sqlite3

from config import DEFAULT_OBJECT_ID, DEFAULT_OBJECT_NAME


def ensure_objects_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id      TEXT PRIMARY KEY,
            name    TEXT,
            address TEXT
        )
    """)


def sync_objects(db_path: str, objects: list[dict]):
    """Создать таблицу objects и upsert'нуть объекты из cameras.yaml. Всегда есть дефолт."""
    conn = sqlite3.connect(db_path)
    try:
        ensure_objects_table(conn)
        rows = list(objects) + [{"id": DEFAULT_OBJECT_ID, "name": DEFAULT_OBJECT_NAME, "address": ""}]
        seen = set()
        for o in rows:
            if o["id"] in seen:
                continue
            seen.add(o["id"])
            conn.execute(
                "INSERT INTO objects (id, name, address) VALUES (?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, address=excluded.address",
                (o["id"], o.get("name", ""), o.get("address", "")),
            )
        conn.commit()
    finally:
        conn.close()
