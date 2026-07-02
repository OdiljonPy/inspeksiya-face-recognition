# -*- coding: utf-8 -*-
"""
vehicle_log.py — Этап 3 ANPR. SQLite-таблица vehicle_events + анти-дребезг.

Таблица по ТЗ: vehicle_events(id, timestamp, camera_id, zone, plate_text,
plate_normalized, confidence, snapshot_path). Доп. поля для интерим-режима:
valid (соответствие формату РУз) и region_uncertain (регион ненадёжен).

Анти-дребезг: одну машину не логируем чаще раза в dedup_seconds на камеру.
Ключ дедупа — ТЕЛО номера (надёжная часть), а не полная строка, чтобы «плавающий»
регион не плодил дубли.

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
        self._last_seen: dict[tuple[str, str], float] = {}  # (cam_id, dedup_key) -> ts

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
                object_id        TEXT DEFAULT 'default'      -- объект/стройплощадка (Задача 2)
            )
        """)
        # миграция старых БД (идемпотентно)
        try:
            self.conn.execute("ALTER TABLE vehicle_events ADD COLUMN object_id TEXT DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_ts ON vehicle_events(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_cam ON vehicle_events(camera_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_plate ON vehicle_events(plate_normalized)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_veh_obj ON vehicle_events(object_id)")
        self.conn.commit()

    def log(self, camera_id, zone, plate_text, plate_normalized, confidence,
            snapshot_path, dedup_key, valid, region_uncertain, ts=None, object_id="default"):
        """
        Записать событие с анти-дребезгом по (camera_id, dedup_key).
        Возвращает rowid новой записи или None (если задедуплено).
        """
        if ts is None:
            ts = time.time()
        key = (camera_id, dedup_key)
        with self.lock:
            last = self._last_seen.get(key, 0.0)
            if ts - last < self.dedup_seconds:
                return None
            self._last_seen[key] = ts
            cur = self.conn.execute(
                "INSERT INTO vehicle_events (timestamp, camera_id, zone, plate_text, "
                "plate_normalized, confidence, snapshot_path, valid, region_uncertain, object_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, camera_id, zone, plate_text, plate_normalized, float(confidence),
                 snapshot_path, 1 if valid else 0, 1 if region_uncertain else 0, object_id),
            )
            self.conn.commit()
            return cur.lastrowid

    def set_snapshot(self, rowid: int, snapshot_path: str):
        """Дописать путь к кропу после его сохранения (кроп пишем только для залогированных)."""
        with self.lock:
            self.conn.execute("UPDATE vehicle_events SET snapshot_path=? WHERE id=?",
                              (snapshot_path, rowid))
            self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.close()
