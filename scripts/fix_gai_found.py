# -*- coding: utf-8 -*-
r"""
fix_gai_found.py — починить ошибочные статусы «есть в базе ГАИ».

Проблема: сервис ГАИ на некоторые номера отвечает HTTP 200 с pResult=1, но БЕЗ
данных машины (пустой Vehicle, нет владельца). Старая классификация смотрела
только на pResult -> такие номера помечались found и попадали под фильтр
gai=found, хотя машины в базе нет.

Скрипт БЕЗ запросов к ГАИ переклассифицирует сохранённые ответы (plate_info.gai_json)
по новой строгой логике (gai_check.gai_found) и правит plate_info + все события
номера. Для перевёрнутых в not_found дополнительно ставит has_contract=0
(машины нет в базе -> фактур быть не может).

Запуск (на сервере): python scripts/fix_gai_found.py
                     python scripts/fix_gai_found.py --dry-run   # только показать
"""
import os
import sys
import json
import sqlite3
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import load_settings
from anpr.gai_check import gai_found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="показать, ничего не менять")
    args = ap.parse_args()

    db = load_settings()["paths"]["db"]
    if not os.path.exists(db):
        print(f"БД не найдена: {db}")
        return 0
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT plate_normalized, gai_status, gai_json FROM plate_info "
        "WHERE gai_status='found'").fetchall()
    print(f"Номеров со статусом found: {len(rows)}")

    flipped = 0
    for r in rows:
        plate = r["plate_normalized"]
        try:
            data = json.loads(r["gai_json"]) if r["gai_json"] else None
        except (ValueError, TypeError):
            data = None
        if gai_found(data):
            continue                       # действительно есть в базе — не трогаем
        flipped += 1
        reason = "нет сохранённого ответа" if not data else "pResult=1, но данных машины нет"
        print(f"  {plate}: found -> not_found ({reason})")
        if args.dry_run:
            continue
        conn.execute("UPDATE plate_info SET gai_status='not_found' WHERE plate_normalized=?",
                     (plate,))
        conn.execute("UPDATE vehicle_events SET gai_status='not_found' "
                     "WHERE plate_normalized=?", (plate,))
        # нет в базе ГАИ -> владельца/ИНН нет -> фактур быть не может
        conn.execute("UPDATE vehicle_events SET has_contract=0 "
                     "WHERE plate_normalized=? AND (has_contract IS NULL OR has_contract=1)",
                     (plate,))
    if not args.dry_run:
        conn.commit()
    conn.close()
    print(f"{'Нашлось бы' if args.dry_run else 'Исправлено'} номеров: {flipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
