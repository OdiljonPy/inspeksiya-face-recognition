# -*- coding: utf-8 -*-
r"""
backfill_gai_status.py — разовая проверка СТАРЫХ событий транспорта по базе ГАИ.

Автопроверка (gai_check_on_new) работает только для НОВЫХ событий; этот скрипт
проходит по истории: берёт уникальные номера без статуса, шлёт запрос на
integration.gai_url и пишет статус во ВСЕ события каждого номера.
Статусы: found | not_found (HTTP 404/500 или pResult!=1) | error (сервис недоступен).

Запуск (на сервере, где доступен сервис ГАИ):
  python scripts/backfill_gai_status.py               # только непроверенные
  python scripts/backfill_gai_status.py --retry-errors  # + перепроверить error
  python scripts/backfill_gai_status.py --all           # перепроверить ВСЁ
"""
import os
import sys
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import load_settings
from anpr.gai_check import check_plate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-errors", action="store_true",
                    help="перепроверить и события со статусом error")
    ap.add_argument("--all", action="store_true", help="перепроверить все события")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="пауза между запросами, сек (не молотить сервис)")
    args = ap.parse_args()

    cfg = load_settings()
    icfg = cfg.get("integration", {}) or {}
    url = icfg.get("gai_url", "")
    if not url:
        print("integration.gai_url не настроен в settings.yaml")
        return 1
    timeout = float(icfg.get("gai_timeout", 12))
    db = cfg["paths"]["db"]
    if not os.path.exists(db):
        print(f"БД не найдена: {db}")
        return 0

    if args.all:
        cond = "1=1"
    elif args.retry_errors:
        cond = "(gai_status IS NULL OR gai_status='' OR gai_status='error')"
    else:
        cond = "(gai_status IS NULL OR gai_status='')"

    conn = sqlite3.connect(db)
    plates = [r[0] for r in conn.execute(
        f"SELECT DISTINCT plate_normalized FROM vehicle_events WHERE {cond} "
        "AND plate_normalized != ''")]
    print(f"Номеров к проверке: {len(plates)} (url: {url})")

    stats = {"found": 0, "not_found": 0, "error": 0}
    for i, plate in enumerate(plates, 1):
        status = check_plate(url, plate, timeout)
        stats[status] += 1
        n = conn.execute("UPDATE vehicle_events SET gai_status=? WHERE plate_normalized=?",
                         (status, plate)).rowcount
        conn.commit()
        print(f"[{i}/{len(plates)}] {plate}: {status} (событий: {n})")
        time.sleep(args.delay)
    conn.close()
    print(f"Готово. found={stats['found']} not_found={stats['not_found']} error={stats['error']}")
    if stats["error"]:
        print("Были ошибки сервиса — повтори позже с --retry-errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
