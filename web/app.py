# -*- coding: utf-8 -*-
r"""
web/app.py — Этап 5. FastAPI-дашборд (localhost).

Показывает:
  - таблицу последних событий с миниатюрами лиц + фильтр по камере;
  - вкладку «Галерея ID» со всеми уникальными людьми (снимок + статистика).

Данные берём из events.db (SQLite) и data/gallery/ (meta.json + faces/).
Снимки отдаются статикой из папки галереи.

Запуск:
  .\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000
  затем открыть http://127.0.0.1:8000
"""
import os
import sys
import json
import sqlite3

# чтобы импортировать config из src/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import load_settings, load_cameras
from gallery import Gallery

cfg = load_settings()
DB_PATH = cfg["paths"]["db"]
GALLERY_DIR = cfg["gallery"]["dir"]
if not os.path.isabs(GALLERY_DIR):
    GALLERY_DIR = os.path.normpath(os.path.join(_ROOT, GALLERY_DIR))
FACES_DIR = os.path.join(GALLERY_DIR, "faces")
META_PATH = os.path.join(GALLERY_DIR, "meta.json")
PLATES_DIR = cfg["paths"]["plates"]
if not os.path.isabs(PLATES_DIR):
    PLATES_DIR = os.path.normpath(os.path.join(_ROOT, PLATES_DIR))
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(PLATES_DIR, exist_ok=True)

app = FastAPI(title="Face Recognition + ANPR Dashboard")
# миниатюры лиц и номеров
app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")
app.mount("/plates", StaticFiles(directory=PLATES_DIR), name="plates")


def _plate_url(path: str) -> str:
    return "/plates/" + os.path.basename(path) if path else ""


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _face_url(crop_path: str) -> str:
    """Из пути снимка делаем URL /faces/<имя>."""
    if not crop_path:
        return ""
    return "/faces/" + os.path.basename(crop_path)


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(TEMPLATES, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/cameras")
def api_cameras():
    """Список камер для фильтра: из cameras.yaml + те, что реально встречались в событиях."""
    cams = {c["id"]: c.get("zone", "") for c in load_cameras()}
    if os.path.exists(DB_PATH):
        with _db() as conn:
            for tbl in ("events", "vehicle_events"):
                try:
                    for r in conn.execute(f"SELECT DISTINCT camera_id, zone FROM {tbl}"):
                        cams.setdefault(r["camera_id"], r["zone"] or "")
                except sqlite3.OperationalError:
                    pass  # таблицы может ещё не быть
    return [{"id": k, "zone": v} for k, v in cams.items()]


@app.get("/api/events")
def api_events(camera: str = Query("", description="фильтр по camera_id"),
               limit: int = Query(100, ge=1, le=1000)):
    if not os.path.exists(DB_PATH):
        return JSONResponse([])
    q = ("SELECT id, ts, camera_id, zone, person, score, is_new, crop_path "
         "FROM events")
    params = []
    if camera:
        q += " WHERE camera_id = ?"
        params.append(camera)
    q += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    out = []
    with _db() as conn:
        for r in conn.execute(q, params):
            out.append({
                "id": r["id"],
                "ts": r["ts"],
                "camera_id": r["camera_id"],
                "zone": r["zone"],
                "person": r["person"],
                "score": round(r["score"], 3) if r["score"] is not None else None,
                "is_new": bool(r["is_new"]),
                "face_url": _face_url(r["crop_path"]),
            })
    return out


@app.get("/api/vehicle_events")
def api_vehicle_events(camera: str = Query("", description="фильтр по camera_id"),
                       q: str = Query("", description="поиск по номеру (подстрока)"),
                       limit: int = Query(100, ge=1, le=1000)):
    if not os.path.exists(DB_PATH):
        return JSONResponse([])
    sql = ("SELECT id, timestamp, camera_id, zone, plate_text, plate_normalized, "
           "confidence, snapshot_path, valid, region_uncertain FROM vehicle_events")
    where, params = [], []
    if camera:
        where.append("camera_id = ?"); params.append(camera)
    if q:
        where.append("plate_normalized LIKE ?"); params.append(f"%{q.upper()}%")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp DESC LIMIT ?"; params.append(limit)

    out = []
    try:
        with _db() as conn:
            for r in conn.execute(sql, params):
                out.append({
                    "id": r["id"], "ts": r["timestamp"],
                    "camera_id": r["camera_id"], "zone": r["zone"],
                    "plate_text": r["plate_text"], "plate": r["plate_normalized"],
                    "confidence": round(r["confidence"], 3) if r["confidence"] is not None else None,
                    "plate_url": _plate_url(r["snapshot_path"]),
                    "valid": bool(r["valid"]), "region_uncertain": bool(r["region_uncertain"]),
                })
    except sqlite3.OperationalError:
        return JSONResponse([])   # таблицы ещё нет
    return out


@app.get("/api/gallery")
def api_gallery():
    """Все уникальные ID из meta.json + счётчик событий по каждому."""
    if not os.path.exists(META_PATH):
        return []
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    counts = {}
    if os.path.exists(DB_PATH):
        with _db() as conn:
            for r in conn.execute("SELECT person, COUNT(*) c FROM events GROUP BY person"):
                counts[r["person"]] = r["c"]

    out = []
    for idn in meta.get("identities", []):
        out.append({
            "label": idn["label"],
            "face_url": _face_url(idn["crop_path"]),
            "first_seen": idn["first_seen"],
            "last_seen": idn["last_seen"],
            "n_emb": idn["n_emb"],
            "events": counts.get(idn["label"], 0),
        })
    out.sort(key=lambda x: x["label"])
    return out


@app.delete("/api/gallery/{label}")
def api_delete_person(label: str):
    """
    Удалить человека из галереи + ВСЕ его события лиц (таблица events).
    Снимок из галереи удаляется вместе с личностью.
    Внимание: если сейчас запущен src\\main.py, он держит галерею в памяти и может
    перезаписать удаление при следующем сохранении — удаляйте как обслуживание.
    """
    removed = Gallery(cfg).delete_identity(label)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"ID {label} не найден в галерее")
    events_deleted = 0
    if os.path.exists(DB_PATH):
        with _db() as conn:
            cur = conn.execute("DELETE FROM events WHERE person = ?", (label,))
            events_deleted = cur.rowcount
            conn.commit()
    return {"deleted": label, "events_deleted": events_deleted}


@app.delete("/api/vehicle/{plate}")
def api_delete_vehicle(plate: str):
    """Удалить ВСЕ события транспорта по номеру + их кропы из data/plates/."""
    plate = plate.upper()
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Нет базы событий")
    with _db() as conn:
        rows = conn.execute(
            "SELECT snapshot_path FROM vehicle_events WHERE plate_normalized = ?",
            (plate,)).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail=f"Номер {plate} не найден")
        for r in rows:
            sp = r["snapshot_path"]
            if sp and os.path.exists(sp):
                try:
                    os.remove(sp)
                except OSError:
                    pass
        cur = conn.execute("DELETE FROM vehicle_events WHERE plate_normalized = ?", (plate,))
        conn.commit()
    return {"deleted": plate, "events_deleted": cur.rowcount}
