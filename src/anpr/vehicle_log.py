# -*- coding: utf-8 -*-
"""
vehicle_log.py — Этап 3 ANPR. SQLite-таблица vehicle_events + анти-дребезг.

Таблица по ТЗ: vehicle_events(id, timestamp, camera_id, zone, plate_text,
plate_normalized, confidence, snapshot_path). Доп. поля для интерим-режима:
valid (соответствие формату РУз) и region_uncertain (регион ненадёжен).

Анти-дребезг: одну машину не логируем чаще раза в dedup_seconds на камеру.
Ключ дедупа — ТЕЛО номера (надёжная часть), а не полная строка, чтобы «плавающий»
регион не плодил дубли.

Голосование в окне дедупа: пока машина в кадре, OCR даёт 5–20 чтений. Первое
логируем, а последующие чтения с БОЛЬШЕЙ уверенностью ОБНОВЛЯЮТ уже записанное
событие (текст/регион/валидность) — точность растёт без новой записи.

Использует тот же файл БД, что и события лиц (events.db), но отдельную таблицу.
"""
import os
import json
import time
import sqlite3
import threading


class VehicleLog:
    def __init__(self, db_path: str, dedup_seconds: float = 30.0):
        self.db_path = db_path
        self.dedup_seconds = float(dedup_seconds)
        self.lock = threading.Lock()
        # (cam_id, dedup_key) -> (ts первой записи, rowid, лучшая conf)
        self._last_seen: dict[tuple[str, str], tuple[float, int, float]] = {}

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        REAL    NOT NULL,
                camera_id        TEXT    NOT NULL,
                zone             TEXT,
                plate_text       TEXT,            -- сырой текст OCR
                plate_normalized TEXT    NOT NULL,-- UPPER без разделителей
                confidence       REAL,
                snapshot_path    TEXT,
                valid            INTEGER NOT NULL DEFAULT 0,  -- соответствует формату РУз
                region_uncertain INTEGER NOT NULL DEFAULT 0, -- интерим: регион ненадёжен
                object_id        TEXT DEFAULT 'default',     -- объект/стройплощадка (Задача 2)
                full_path        TEXT,                       -- полный кадр события (общий вид)
                gai_status       TEXT,                       -- проверка по базе ГАИ:
                                                             -- found | not_found | error | NULL (не проверялся)
                owner_type       TEXT,                       -- shaxsiy | yuridik | kompaniya | NULL (неизвестно)
                owner_inn        TEXT,                       -- ИНН владельца ТС (из ГАИ, только юрлица)
                has_contract     INTEGER                     -- сверка с налогом: 1=фактуры есть,
                                                             -- 0=нет, NULL=не проверялся/неприменимо
            )
        """)
        # миграция старых БД (идемпотентно)
        for stmt in ("ALTER TABLE vehicle_events ADD COLUMN object_id TEXT DEFAULT 'default'",
                     "ALTER TABLE vehicle_events ADD COLUMN full_path TEXT",
                     "ALTER TABLE vehicle_events ADD COLUMN gai_status TEXT",
                     "ALTER TABLE vehicle_events ADD COLUMN owner_type TEXT",
                     "ALTER TABLE vehicle_events ADD COLUMN owner_inn TEXT",
                     "ALTER TABLE vehicle_events ADD COLUMN has_contract INTEGER"):
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_ts ON vehicle_events(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_cam ON vehicle_events(camera_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_plate ON vehicle_events(plate_normalized)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_obj ON vehicle_events(object_id)")
        # под фильтры дашборда (статус/ГАИ/владелец/договор) на больших базах
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_valid ON vehicle_events(valid)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_gai ON vehicle_events(gai_status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_owner ON vehicle_events(owner_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_contract ON vehicle_events(has_contract)")
        # Справочник по УНИКАЛЬНЫМ номерам: полный ответ ГАИ + сверка с налогом (soliq).
        # Наполняется фоновой проверкой (gai_check) и стартовым sweep-ом по старым
        # событиям; отдаётся интеграции в /api/v1/vehicles (details=1) и /info/{plate}.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS plate_info (
                plate_normalized TEXT PRIMARY KEY,
                gai_status    TEXT,     -- found | not_found | error
                gai_json      TEXT,     -- полный ответ ГАИ (JSON), только при found
                gai_checked   REAL,     -- unix ts проверки ГАИ
                owner_inn     TEXT,
                owner_name    TEXT,     -- pOwner из ГАИ
                soliq_json    TEXT,     -- {object_id: {has_contract, facturas, checked, ...}}
                soliq_checked REAL      -- unix ts последней сверки с налогом
            )
        """)
        self.conn.commit()

    def log(self, camera_id, zone, plate_text, plate_normalized, confidence,
            snapshot_path, dedup_key, valid, region_uncertain, ts=None, object_id="default",
            owner_type=""):
        """
        Записать событие с анти-дребезгом по (camera_id, dedup_key).
        Возвращает rowid НОВОЙ записи или None.
        Голосование: если в окне дедупа пришло чтение с большей уверенностью —
        обновляем текст/регион/валидность уже записанного события (rowid не возвращаем,
        чтобы вызывающий не сохранял дубль-снимки).
        """
        if ts is None:
            ts = time.time()
        key = (camera_id, dedup_key)
        with self.lock:
            self._prune(ts)
            prev = self._last_seen.get(key)
            if prev is not None and ts - prev[0] < self.dedup_seconds:
                first_ts, rowid, best_conf = prev
                if float(confidence) > best_conf:
                    # owner_type перезаписываем только пока запись НЕ обогащена данными
                    # ГАИ (owner_inn пуст) — фоновая проверка авторитетнее формата номера
                    self.conn.execute(
                        "UPDATE vehicle_events SET plate_text=?, plate_normalized=?, "
                        "confidence=?, valid=?, region_uncertain=?, "
                        "owner_type=CASE WHEN owner_inn IS NULL OR owner_inn='' "
                        "THEN ? ELSE owner_type END WHERE id=?",
                        (plate_text, plate_normalized, float(confidence),
                         1 if valid else 0, 1 if region_uncertain else 0,
                         owner_type or None, rowid),
                    )
                    self.conn.commit()
                    self._last_seen[key] = (first_ts, rowid, float(confidence))
                return None
            cur = self.conn.execute(
                "INSERT INTO vehicle_events (timestamp, camera_id, zone, plate_text, "
                "plate_normalized, confidence, snapshot_path, valid, region_uncertain, "
                "object_id, owner_type) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (ts, camera_id, zone, plate_text, plate_normalized, float(confidence),
                 snapshot_path, 1 if valid else 0, 1 if region_uncertain else 0, object_id,
                 owner_type or None),
            )
            self.conn.commit()
            self._last_seen[key] = (ts, cur.lastrowid, float(confidence))
            return cur.lastrowid

    def _prune(self, now: float):
        """Не дать _last_seen расти бесконечно: выкинуть ключи старше окна дедупа."""
        if len(self._last_seen) < 512:
            return
        dead = [k for k, v in self._last_seen.items() if now - v[0] >= self.dedup_seconds]
        for k in dead:
            del self._last_seen[k]

    def update_region(self, rowid: int, plate_normalized: str, valid: bool):
        """Дописать восстановленный регион (второй OCR-проход): номер + флаги."""
        with self.lock:
            self.conn.execute(
                "UPDATE vehicle_events SET plate_normalized=?, valid=?, region_uncertain=0 "
                "WHERE id=?", (plate_normalized, 1 if valid else 0, rowid))
            self.conn.commit()

    def set_gai_status(self, rowid: int, status: str):
        """Записать результат проверки по базе ГАИ (found|not_found|error)."""
        with self.lock:
            self.conn.execute("UPDATE vehicle_events SET gai_status=? WHERE id=?",
                              (status, rowid))
            self.conn.commit()

    def set_owner(self, rowid: int, owner_type: str, owner_inn: str = ""):
        """Дописать тип владельца (уточнён данными ГАИ) и его ИНН."""
        with self.lock:
            self.conn.execute("UPDATE vehicle_events SET owner_type=?, owner_inn=? WHERE id=?",
                              (owner_type or None, owner_inn or None, rowid))
            self.conn.commit()

    def set_contract(self, rowid: int, has_contract):
        """Результат сверки с налогом: 1=фактуры есть, 0=нет, None=не проверялся."""
        with self.lock:
            self.conn.execute("UPDATE vehicle_events SET has_contract=? WHERE id=?",
                              (has_contract, rowid))
            self.conn.commit()

    # -------- обновления по НОМЕРУ (все события; для фоновой проверки и sweep) --------
    def set_gai_status_plate(self, plate: str, status: str):
        with self.lock:
            self.conn.execute("UPDATE vehicle_events SET gai_status=? WHERE plate_normalized=?",
                              (status, plate))
            self.conn.commit()

    def set_owner_plate(self, plate: str, object_id: str, owner_type: str, owner_inn: str):
        """Тип/ИНН владельца всем событиям номера НА ОБЪЕКТЕ (kompaniya зависит от объекта)."""
        with self.lock:
            self.conn.execute(
                "UPDATE vehicle_events SET owner_type=?, owner_inn=? "
                "WHERE plate_normalized=? AND object_id IS ?",
                (owner_type or None, owner_inn or None, plate, object_id))
            self.conn.commit()

    def set_contract_plate(self, plate: str, object_id: str, has_contract):
        with self.lock:
            self.conn.execute(
                "UPDATE vehicle_events SET has_contract=? "
                "WHERE plate_normalized=? AND object_id IS ?",
                (has_contract, plate, object_id))
            self.conn.commit()

    # -------- справочник plate_info (полный ответ ГАИ + сверка soliq) --------
    def upsert_gai_info(self, plate: str, status: str, data: dict | None):
        """Сохранить результат проверки ГАИ по номеру (полный JSON при found)."""
        gai_json = json.dumps(data, ensure_ascii=False) if data else None
        owner_inn = str((data or {}).get("pOrganizationInn") or "") or None
        owner_name = str((data or {}).get("pOwner") or "") or None
        with self.lock:
            self.conn.execute(
                "INSERT INTO plate_info (plate_normalized, gai_status, gai_json, "
                "gai_checked, owner_inn, owner_name) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(plate_normalized) DO UPDATE SET gai_status=excluded.gai_status, "
                "gai_json=COALESCE(excluded.gai_json, plate_info.gai_json), "
                "gai_checked=excluded.gai_checked, "
                "owner_inn=COALESCE(excluded.owner_inn, plate_info.owner_inn), "
                "owner_name=COALESCE(excluded.owner_name, plate_info.owner_name)",
                (plate, status, gai_json, time.time(), owner_inn, owner_name))
            self.conn.commit()

    def upsert_soliq_info(self, plate: str, object_id: str, entry: dict):
        """
        Дописать результат сверки с налогом по объекту в plate_info.soliq_json:
        entry = {"has_contract": 1|0|None, "facturas": [...], ...}. Ключ — object_id.
        """
        with self.lock:
            row = self.conn.execute(
                "SELECT soliq_json FROM plate_info WHERE plate_normalized=?",
                (plate,)).fetchone()
            try:
                soliq = json.loads(row[0]) if row and row[0] else {}
            except (ValueError, TypeError):
                soliq = {}
            entry = dict(entry)
            entry["checked"] = time.time()
            soliq[object_id or "default"] = entry
            self.conn.execute(
                "INSERT INTO plate_info (plate_normalized, soliq_json, soliq_checked) "
                "VALUES (?,?,?) ON CONFLICT(plate_normalized) DO UPDATE SET "
                "soliq_json=excluded.soliq_json, soliq_checked=excluded.soliq_checked",
                (plate, json.dumps(soliq, ensure_ascii=False), time.time()))
            self.conn.commit()

    def pending_checks(self) -> list[tuple[str, str]]:
        """
        (plate, object_id) СТАРЫХ событий, которым не хватает проверки ГАИ/soliq:
        нет строки в plate_info, статус error, события без gai_status, или
        для found-номеров с ИНН не сверен налог по этому объекту.
        Используется стартовым sweep-ом (gai_check.sweep_old).
        """
        with self.lock:
            pairs = self.conn.execute(
                "SELECT plate_normalized, object_id, "
                "SUM(CASE WHEN gai_status IS NULL OR gai_status='' THEN 1 ELSE 0 END) miss "
                "FROM vehicle_events WHERE plate_normalized != '' "
                "GROUP BY plate_normalized, object_id").fetchall()
            info = {r[0]: (r[1], r[2], r[3]) for r in self.conn.execute(
                "SELECT plate_normalized, gai_status, owner_inn, soliq_json FROM plate_info")}
        out = []
        for plate, object_id, miss in pairs:
            pi = info.get(plate)
            if pi is None or pi[0] in (None, "", "error") or miss:
                out.append((plate, object_id))
                continue
            if pi[0] == "found" and pi[1] and str(pi[1]).isdigit():
                try:
                    soliq = json.loads(pi[2]) if pi[2] else {}
                except (ValueError, TypeError):
                    soliq = {}
                if (object_id or "default") not in soliq:
                    out.append((plate, object_id))
        return out

    def set_full(self, rowid: int, full_path: str):
        """Дописать путь к полному кадру события."""
        with self.lock:
            self.conn.execute("UPDATE vehicle_events SET full_path=? WHERE id=?",
                              (full_path, rowid))
            self.conn.commit()

    def set_snapshot(self, rowid: int, snapshot_path: str):
        """Дописать путь к кропу после его сохранения (кроп пишем только для залогированных)."""
        with self.lock:
            self.conn.execute("UPDATE vehicle_events SET snapshot_path=? WHERE id=?",
                              (snapshot_path, rowid))
            self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.close()
