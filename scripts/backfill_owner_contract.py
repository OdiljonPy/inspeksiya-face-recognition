# -*- coding: utf-8 -*-
r"""
backfill_owner_contract.py — разовое заполнение owner_type / owner_inn / has_contract
для СТАРЫХ событий транспорта (новые заполняются автоматически в main.py).

УСТАРЕЛ (18.07.2026): main.py при старте сам дозаполняет старые события sweep-ом
(integration.gai_backfill_on_start) и копит полные данные ГАИ/soliq в plate_info.
Скрипт оставлен для ручного прогона без перезапуска сервиса.

Два прохода:
  1. ОФФЛАЙН: базовый owner_type по формату тела номера ("A123BC" -> shaxsiy,
     "123ABC" -> yuridik). Не требует внешних сервисов, работает и на dev.
  2. СЕРВИСЫ (пропускается с --no-gai): по уникальным номерам запрос в ГАИ
     (integration.gai_url) — уточняет owner_type (pOwnerType, kompaniya по ИНН
     генподрядчика объекта) и owner_inn; затем сверка с налогом
     (integration.facturas_url) -> has_contract по каждому объекту номера.

Запуск (проход 2 — на сервере, где доступны сервисы):
  python scripts/backfill_owner_contract.py             # только незаполненные
  python scripts/backfill_owner_contract.py --all       # пересчитать всё
  python scripts/backfill_owner_contract.py --no-gai    # только оффлайн-проход
"""
import os
import sys
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import load_settings, load_objects
from anpr.plate_format import PlateValidator, owner_type_from_body, OWNER_KOMPANIYA
from anpr.gai_check import fetch_plate, owner_from_gai, contract_period, check_contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="пересчитать все события (иначе — только незаполненные)")
    ap.add_argument("--no-gai", action="store_true",
                    help="только оффлайн-проход по формату номера (без сервисов)")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="пауза между запросами к сервисам, сек")
    args = ap.parse_args()

    cfg = load_settings()
    db = cfg["paths"]["db"]
    if not os.path.exists(db):
        print(f"БД не найдена: {db}")
        return 0
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    # --- проход 1: owner_type по формату тела номера (оффлайн) ---
    validator = PlateValidator(cfg["anpr"]["plate_regex"])
    cond = "1=1" if args.all else "(owner_type IS NULL OR owner_type = '')"
    rows = conn.execute(
        f"SELECT DISTINCT plate_normalized FROM vehicle_events WHERE {cond} "
        "AND plate_normalized != ''").fetchall()
    filled = 0
    for r in rows:
        plate = r["plate_normalized"]
        ot = owner_type_from_body(validator.parse(plate).body)
        if not ot:
            continue
        # не трогаем записи, уже уточнённые ГАИ (owner_inn заполнен)
        filled += conn.execute(
            "UPDATE vehicle_events SET owner_type=? WHERE plate_normalized=? "
            "AND (owner_inn IS NULL OR owner_inn='')" + ("" if args.all else
            " AND (owner_type IS NULL OR owner_type='')"),
            (ot, plate)).rowcount
    conn.commit()
    print(f"Проход 1 (формат номера): обновлено событий: {filled}")

    if args.no_gai:
        conn.close()
        return 0

    # --- проход 2: уточнение через ГАИ + сверка с налогом ---
    icfg = cfg.get("integration", {}) or {}
    gai_url = icfg.get("gai_url", "")
    if not gai_url:
        print("integration.gai_url не настроен — проход 2 пропущен")
        conn.close()
        return 0
    facturas_url = icfg.get("facturas_url", "")
    timeout = float(icfg.get("gai_timeout", 12))
    months = int(icfg.get("facturas_months", 3))
    objs = {o["id"]: o for o in load_objects()}

    cond = "1=1" if args.all else \
        "(owner_inn IS NULL OR owner_inn='' OR has_contract IS NULL)"
    plates = [r[0] for r in conn.execute(
        f"SELECT DISTINCT plate_normalized FROM vehicle_events WHERE {cond} "
        "AND plate_normalized != ''")]
    print(f"Проход 2 (ГАИ + налог): номеров к проверке: {len(plates)}")
    start_s, end_s = contract_period(months)
    stats = {"found": 0, "not_found": 0, "error": 0, "contracts": 0}
    for i, plate in enumerate(plates, 1):
        status, data = fetch_plate(gai_url, plate, timeout)
        stats[status] += 1
        conn.execute("UPDATE vehicle_events SET gai_status=? WHERE plate_normalized=?",
                     (status, plate))
        if status != "found" or not data:
            conn.commit()
            print(f"[{i}/{len(plates)}] {plate}: gai={status}")
            time.sleep(args.delay)
            continue
        # владелец по объекту каждого события этого номера
        oids = [r[0] for r in conn.execute(
            "SELECT DISTINCT object_id FROM vehicle_events WHERE plate_normalized=?",
            (plate,))]
        note = []
        for oid in oids:
            obj = objs.get(oid or "default", {})
            ot, inn = owner_from_gai(data, str(obj.get("construction_inn") or ""))
            if ot or inn:
                conn.execute("UPDATE vehicle_events SET owner_type=?, owner_inn=? "
                             "WHERE plate_normalized=? AND object_id IS ?",
                             (ot or None, inn or None, plate, oid))
            hc = None
            if (facturas_url and inn.isdigit() and ot != OWNER_KOMPANIYA):
                buyers = [str(obj.get("zakazchik_inn") or ""),
                          str(obj.get("construction_inn") or "")]
                hc, _facturas = check_contract(facturas_url, inn, buyers, start_s, end_s, timeout)
                if hc is not None:
                    conn.execute("UPDATE vehicle_events SET has_contract=? "
                                 "WHERE plate_normalized=? AND object_id IS ?",
                                 (hc, plate, oid))
                    stats["contracts"] += hc
            note.append(f"{oid}:{ot or '?'}{'' if hc is None else f'/договор={hc}'}")
        conn.commit()
        print(f"[{i}/{len(plates)}] {plate}: gai=found {' '.join(note)}")
        time.sleep(args.delay)
    conn.close()
    print(f"Готово. found={stats['found']} not_found={stats['not_found']} "
          f"error={stats['error']} с фактурами={stats['contracts']}")
    if stats["error"]:
        print("Были ошибки сервиса ГАИ — повтори позже (незаполненные добираются без --all).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
