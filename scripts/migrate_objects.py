# -*- coding: utf-8 -*-
r"""
migrate_objects.py — миграция БД под объекты (Задача 2), без потери данных.

Делает:
  1) создаёт таблицу objects и синхронизирует объекты из cameras.yaml (+ дефолтный);
  2) добавляет колонку object_id в events и vehicle_events (идемпотентно);
  3) бэкфилл: существующим событиям без object_id ставит 'default'.

Запуск:
  python scripts\migrate_objects.py
"""
import os
import sys
import sqlite3

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from config import load_settings, load_objects, DEFAULT_OBJECT_ID
from objects_db import sync_objects


def _add_col(conn, table, col, typ):
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        return True
    return False


def main() -> int:
    cfg = load_settings()
    db = cfg["paths"]["db"]
    if not os.path.exists(db):
        print(f"БД ещё нет ({db}) — создастся при первом запуске. Синхронизирую только объекты.")
        sync_objects(db, load_objects())
        return 0

    # 1) таблица objects + объекты из конфига
    sync_objects(db, load_objects())

    conn = sqlite3.connect(db)
    added = []
    for table in ("events", "vehicle_events"):
        try:
            if _add_col(conn, table, "object_id", "TEXT DEFAULT 'default'"):
                added.append(f"{table}.object_id")
            # бэкфилл существующих строк
            conn.execute(f"UPDATE {table} SET object_id=? WHERE object_id IS NULL",
                         (DEFAULT_OBJECT_ID,))
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_obj ON {table}(object_id)")
        except sqlite3.OperationalError as e:
            print(f"  {table}: {e}")
    # композитные индексы под агрегацию аналитики (Задача 3)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_obj_ts ON events(object_id, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person_ts ON events(person, ts)")
    except sqlite3.OperationalError:
        pass
    conn.commit()

    objs = conn.execute("SELECT id, name FROM objects").fetchall()
    conn.close()
    print(f"OK. Добавлено: {added or 'нет (уже актуально)'}. Объекты в БД: {objs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
