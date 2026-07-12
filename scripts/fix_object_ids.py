# -*- coding: utf-8 -*-
"""
fix_object_ids.py — разовая миграция: привести object_id старых событий к ID
объектов из cameras.yaml (события писались с 'avloniy'/'102maktab', а объекты
объявлены как 'obj_avloniy'/'obj_102maktab').

Запуск: python scripts/fix_object_ids.py
"""
import os
import sys
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import load_settings

# старый object_id -> новый
RENAMES = {
    "avloniy": "obj_avloniy",
    "102maktab": "obj_102maktab",
}


def main():
    db = load_settings()["paths"]["db"]
    if not os.path.exists(db):
        print(f"БД не найдена: {db} — мигрировать нечего.")
        return 0
    conn = sqlite3.connect(db)
    total = 0
    for table in ("events", "vehicle_events"):
        for old, new in RENAMES.items():
            try:
                cur = conn.execute(
                    f"UPDATE {table} SET object_id=? WHERE object_id=?", (new, old))
                if cur.rowcount:
                    print(f"{table}: {old} -> {new}: {cur.rowcount} строк")
                    total += cur.rowcount
            except sqlite3.OperationalError:
                pass  # таблицы/колонки может не быть — пропускаем
    conn.commit()
    conn.close()
    print(f"Готово, обновлено строк: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
