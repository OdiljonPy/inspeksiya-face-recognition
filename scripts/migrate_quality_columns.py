# -*- coding: utf-8 -*-
r"""
migrate_quality_columns.py — миграция БД: добавить колонки метрик качества в events.

Идемпотентно: если колонки уже есть — ничего не делает, данные не теряются.
(EventLog делает то же на старте, но по ТЗ миграция вынесена отдельным скриптом.)

Запуск:
  python scripts\migrate_quality_columns.py
"""
import os
import sys
import sqlite3

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from config import load_settings

COLUMNS = (("full_path", "TEXT"), ("q_det", "REAL"), ("q_px", "REAL"),
           ("q_blur", "REAL"), ("q_yaw", "REAL"))


def main() -> int:
    db = load_settings()["paths"]["db"]
    if not os.path.exists(db):
        print(f"БД ещё нет ({db}) — миграция не нужна, создастся при первом запуске.")
        return 0
    conn = sqlite3.connect(db)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    added = []
    for col, typ in COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE events ADD COLUMN {col} {typ}")
            added.append(col)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person ON events(person)")
    conn.commit()
    conn.close()
    print(f"OK. Добавлено колонок: {added or 'нет (уже актуально)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
