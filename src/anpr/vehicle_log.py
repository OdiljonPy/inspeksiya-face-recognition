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
