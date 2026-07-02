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
                crop_path  TEXT,
                full_path  TEXT,             -- полный кадр события (общий вид)
                q_det      REAL,             -- метрики качества лица (Задача 1)
                q_px       REAL,
                q_blur     REAL,
                q_yaw      REAL,
                object_id  TEXT DEFAULT 'default'  -- объект/стройплощадка (Задача 2)
            )
        """)
        # миграция старых БД: добавить недостающие колонки (идемпотентно)
        for col, typ in (("full_path", "TEXT"), ("q_det", "REAL"), ("q_px", "REAL"),
                         ("q_blur", "REAL"), ("q_yaw", "REAL"),
                         ("object_id", "TEXT DEFAULT 'default'")):
            try:
                self.conn.execute(f"ALTER TABLE events ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_cam ON events(camera_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person ON events(person)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_obj ON events(object_id)")
        # композитные индексы под агрегацию аналитики (Задача 3)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_obj_ts ON events(object_id, ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person_ts ON events(person, ts)")
        self.conn.commit()

    def log(self, camera_id: str, zone: str, person: str, score: float,
            is_new: bool, crop_path: str, ts: float | None = None,
            q_det=None, q_px=None, q_blur=None, q_yaw=None, object_id: str = "default"):
        """
        Записать событие с учётом анти-дребезга. Возвращает rowid новой записи
        или None (если задедуплено). Новые лица (is_new) логируются всегда.
        q_* — метрики качества лица; object_id — объект/стройплощадка.
        """
        if ts is None:
            ts = time.time()
        key = (camera_id, person)
        with self.lock:
            if not is_new:
                last = self._last_seen.get(key, 0.0)
                if ts - last < self.dedup_seconds:
                    return None
            self._last_seen[key] = ts
            cur = self.conn.execute(
                "INSERT INTO events (ts, camera_id, zone, person, score, is_new, crop_path, "
                "q_det, q_px, q_blur, q_yaw, object_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, camera_id, zone, person, float(score), 1 if is_new else 0, crop_path,
                 q_det, q_px, q_blur, q_yaw, object_id),
            )
            self.conn.commit()
            return cur.lastrowid

    def set_full(self, rowid: int, full_path: str):
        """Дописать путь к полному кадру (сохраняем только для залогированных событий)."""
        with self.lock:
            self.conn.execute("UPDATE events SET full_path=? WHERE id=?", (full_path, rowid))
            self.conn.commit()

    def set_crop(self, rowid: int, crop_path: str):
        """Дописать путь к снимку лица (для LOW_QUALITY — пишем кроп только при логировании)."""
        with self.lock:
            self.conn.execute("UPDATE events SET crop_path=? WHERE id=?", (crop_path, rowid))
            self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.close()
