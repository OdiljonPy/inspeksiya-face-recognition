# -*- coding: utf-8 -*-
"""
events.py — Этап 4. Логирование событий в SQLite + анти-дребезг.

Событие = факт появления человека на камере: время, camera_id, зона, ID человека,
score, признак нового лица, путь к снимку (из галереи — снимок один на ID).

Анти-дребезг: не логировать одного и того же человека на одной камере чаще,
чем раз в dedup_seconds (новые лица логируем всегда).
"""
import os
import time
import sqlite3
import threading


class EventLog:
    def __init__(self, db_path: str, dedup_seconds: float = 30.0):
        self.db_path = db_path
        self.dedup_seconds = float(dedup_seconds)
        self.lock = threading.Lock()
        self._last_seen: dict[tuple[str, str], float] = {}  # (cam_id, label) -> ts

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # SQLite из нескольких потоков: одно соединение + check_same_thread=False + lock
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL    NOT NULL,
                camera_id  TEXT    NOT NULL,
                zone       TEXT,
                person     TEXT    NOT NULL,
                score      REAL,
                is_new     INTEGER NOT NULL DEFAULT 0,
                crop_path  TEXT
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_cam ON events(camera_id)")
        self.conn.commit()

    def log(self, camera_id: str, zone: str, person: str, score: float,
            is_new: bool, crop_path: str, ts: float | None = None) -> bool:
        """
        Записать событие с учётом анти-дребезга. Возвращает True, если записали.
        Новые лица (is_new) логируются всегда.
        """
        if ts is None:
            ts = time.time()
        key = (camera_id, person)
        with self.lock:
            if not is_new:
                last = self._last_seen.get(key, 0.0)
                if ts - last < self.dedup_seconds:
                    return False
            self._last_seen[key] = ts
            self.conn.execute(
                "INSERT INTO events (ts, camera_id, zone, person, score, is_new, crop_path) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, camera_id, zone, person, float(score), 1 if is_new else 0, crop_path),
            )
            self.conn.commit()
        return True

    def close(self):
        with self.lock:
            self.conn.close()
