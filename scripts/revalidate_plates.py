# -*- coding: utf-8 -*-
r"""
revalidate_plates.py — перепроверить ВСЕ события транспорта по ТЕКУЩИМ правилам
валидации (settings.anpr.plate_regex) и обновить флаги valid/region_uncertain.

Зачем: правила ужесточились (14.07.2026 — только два типа: «01 A 001 AA» и
«01 001 AAA», прицепные «4 цифры+2 буквы» убраны), а старые записи в БД хранят
флаги старой валидации — например «011477DA» показывался как OK.

Запуск:
  python scripts/revalidate_plates.py            # только перефлаговать
  python scripts/revalidate_plates.py --purge-invalid   # + УДАЛИТЬ невалидные события и их снимки
"""
import os
import sys
import sqlite3
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import load_settings
from anpr.plate_format import PlateValidator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge-invalid", action="store_true",
                    help="удалить невалидные события целиком (вместе со снимками)")
    args = ap.parse_args()

    cfg = load_settings()
    v = PlateValidator(cfg["anpr"]["plate_regex"])
    db = cfg["paths"]["db"]
    if not os.path.exists(db):
        print(f"БД не найдена: {db}")
        return 0

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, plate_normalized, valid, region_uncertain, snapshot_path, full_path "
        "FROM vehicle_events").fetchall()

    changed = purged = 0
    for r in rows:
        pp = v.parse(r["plate_normalized"] or "")
        new_valid = 1 if pp.valid else 0
        new_unc = 1 if pp.region_uncertain else 0
        if args.purge_invalid and not pp.valid and not pp.region_uncertain:
            # мусор: тело не соответствует ни одному из типов — сносим событие и файлы
            for p in (r["snapshot_path"], r["full_path"]):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            conn.execute("DELETE FROM vehicle_events WHERE id=?", (r["id"],))
            purged += 1
            continue
        if new_valid != r["valid"] or new_unc != r["region_uncertain"]:
            conn.execute("UPDATE vehicle_events SET valid=?, region_uncertain=? WHERE id=?",
                         (new_valid, new_unc, r["id"]))
            changed += 1
            print(f"  {r['plate_normalized']}: valid {r['valid']}->{new_valid} "
                  f"region_unc {r['region_uncertain']}->{new_unc}")
    conn.commit()
    conn.close()
    print(f"Всего записей: {len(rows)}, перефлаговано: {changed}, удалено: {purged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
