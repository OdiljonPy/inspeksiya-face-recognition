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
import time
import sqlite3
import threading

# чтобы импортировать config из src/ и live из web/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # каталог web/ (live.py)

from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from config import load_settings, load_cameras, load_objects
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
FULL_DIR = cfg["paths"].get("full", "data/full")
if not os.path.isabs(FULL_DIR):
    FULL_DIR = os.path.normpath(os.path.join(_ROOT, FULL_DIR))
LOWQ_DIR = cfg["paths"].get("lowq", "data/lowq")
if not os.path.isabs(LOWQ_DIR):
    LOWQ_DIR = os.path.normpath(os.path.join(_ROOT, LOWQ_DIR))
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(PLATES_DIR, exist_ok=True)
os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(LOWQ_DIR, exist_ok=True)

app = FastAPI(title="Face Recognition + ANPR Dashboard")
# миниатюры лиц, номеров, полные кадры и снимки LOW_QUALITY
app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")
app.mount("/plates", StaticFiles(directory=PLATES_DIR), name="plates")
app.mount("/full", StaticFiles(directory=FULL_DIR), name="full")
app.mount("/lowq", StaticFiles(directory=LOWQ_DIR), name="lowq")


@app.middleware("http")
async def _no_cache_images(request, call_next):
    """Запрет кэша на миниатюры: файл person_0001.jpg мог смениться на другого человека."""
    resp = await call_next(request)
    if request.url.path.startswith(("/faces/", "/plates/", "/full/", "/lowq/")):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def _full_url(path: str) -> str:
    return _versioned("/full", FULL_DIR, path)


def _versioned(url_prefix: str, base_dir: str, path: str) -> str:
    """
    URL миниатюры с cache-busting ?v=<mtime>. Когда файл перезаписан (тот же ID —
    другой человек), mtime меняется -> URL меняется -> браузер грузит свежую картинку.
    """
    if not path:
        return ""
    name = os.path.basename(path)
    try:
        v = int(os.path.getmtime(os.path.join(base_dir, name)))
    except OSError:
        v = 0
    return f"{url_prefix}/{name}?v={v}"


def _plate_url(path: str) -> str:
    return _versioned("/plates", PLATES_DIR, path)


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _face_url(crop_path: str) -> str:
    """
    URL миниатюры лица. LOW_QUALITY-снимки лежат в data/lowq, снимки ID — в
    data/gallery/faces. Маршрутизируем по фактической папке файла.
    """
    if not crop_path:
        return ""
    name = os.path.basename(crop_path)
    norm = crop_path.replace("\\", "/")
    if "/lowq/" in norm or os.path.exists(os.path.join(LOWQ_DIR, name)):
        return _versioned("/lowq", LOWQ_DIR, crop_path)
    return _versioned("/faces", FACES_DIR, crop_path)




def _ensure_schema():
    """Гарантировать новые колонки (если БД создана старой версией кода)."""
    if not os.path.exists(DB_PATH):
        return
    with _db() as c:
        for stmt in ("ALTER TABLE events ADD COLUMN full_path TEXT",
                     "ALTER TABLE events ADD COLUMN uncertain INTEGER NOT NULL DEFAULT 0",
                     "ALTER TABLE vehicle_events ADD COLUMN full_path TEXT",
                     "ALTER TABLE vehicle_events ADD COLUMN object_id TEXT DEFAULT 'default'"):
            try:
                c.execute(stmt)
                c.commit()
            except sqlite3.OperationalError:
                pass  # колонка/таблица уже есть или таблицы ещё нет


_ensure_schema()


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(TEMPLATES, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/objects")
def api_objects():
    """Список объектов (стройплощадок) для фильтра."""
    objs = []
    if os.path.exists(DB_PATH):
        with _db() as conn:
            try:
                for r in conn.execute("SELECT id, name, address FROM objects ORDER BY name"):
                    objs.append({"id": r["id"], "name": r["name"] or r["id"],
                                 "address": r["address"] or ""})
            except sqlite3.OperationalError:
                pass
    if not objs:                              # fallback из конфига
        for o in load_objects():
            objs.append({"id": o["id"], "name": o.get("name", o["id"]),
                         "address": o.get("address", "")})
    return objs


@app.get("/api/cameras")
def api_cameras(object: str = Query("", description="фильтр по объекту")):
    """Камеры для фильтра (+ object_id). Если задан object — только его камеры."""
    cams = {}
    for c in load_cameras():
        cams[c["id"]] = {"zone": c.get("zone", ""), "object_id": c.get("object_id", "default")}
    if os.path.exists(DB_PATH):
        with _db() as conn:
            for tbl in ("events", "vehicle_events"):
                try:
                    for r in conn.execute(f"SELECT DISTINCT camera_id, zone, object_id FROM {tbl}"):
                        cams.setdefault(r["camera_id"], {"zone": r["zone"] or "",
                                                         "object_id": r["object_id"] or "default"})
                except sqlite3.OperationalError:
                    pass
    out = [{"id": k, "zone": v["zone"], "object_id": v["object_id"]} for k, v in cams.items()]
    if object:
        out = [c for c in out if c["object_id"] == object]
    return out


@app.get("/api/events")
def api_events(camera: str = Query("", description="фильтр по camera_id"),
               object: str = Query("", description="фильтр по объекту"),
               limit: int = Query(100, ge=1, le=1000)):
    if not os.path.exists(DB_PATH):
        return JSONResponse([])
    q = ("SELECT id, ts, camera_id, zone, person, score, is_new, crop_path, full_path, "
         "uncertain FROM events")
    where, params = [], []
    if camera:
        where.append("camera_id = ?"); params.append(camera)
    if object:
        where.append("object_id = ?"); params.append(object)
    if where:
        q += " WHERE " + " AND ".join(where)
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
                "uncertain": bool(r["uncertain"]),
                "face_url": _face_url(r["crop_path"]),
                "full_url": _full_url(r["full_path"]),
            })
    return out


@app.get("/api/vehicle_events")
def api_vehicle_events(camera: str = Query("", description="фильтр по camera_id"),
                       object: str = Query("", description="фильтр по объекту"),
                       q: str = Query("", description="поиск по номеру (подстрока)"),
                       limit: int = Query(100, ge=1, le=1000)):
    if not os.path.exists(DB_PATH):
        return JSONResponse([])
    sql = ("SELECT id, timestamp, camera_id, zone, plate_text, plate_normalized, "
           "confidence, snapshot_path, valid, region_uncertain, full_path FROM vehicle_events")
    where, params = [], []
    if camera:
        where.append("camera_id = ?"); params.append(camera)
    if object:
        where.append("object_id = ?"); params.append(object)
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
                    "full_url": _full_url(r["full_path"]),
                    "valid": bool(r["valid"]), "region_uncertain": bool(r["region_uncertain"]),
                })
    except sqlite3.OperationalError:
        return JSONResponse([])   # таблицы ещё нет
    return out


@app.get("/api/gallery")
def api_gallery(object: str = Query("", description="фильтр по объекту")):
    """Уникальные ID из meta.json + счётчик событий. object -> только те, кто там появлялся."""
    if not os.path.exists(META_PATH):
        return []
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    counts = {}
    on_object = None            # множество label, появлявшихся на объекте
    if os.path.exists(DB_PATH):
        with _db() as conn:
            for r in conn.execute("SELECT person, COUNT(*) c FROM events GROUP BY person"):
                counts[r["person"]] = r["c"]
            if object:
                on_object = set()
                try:
                    for r in conn.execute(
                            "SELECT DISTINCT person FROM events WHERE object_id = ?", (object,)):
                        on_object.add(r["person"])
                except sqlite3.OperationalError:
                    on_object = None

    out = []
    for idn in meta.get("identities", []):
        if on_object is not None and idn["label"] not in on_object:
            continue            # на этом объекте не появлялся
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


# ============================ АНАЛИТИКА (Задача 3) ============================
_PERIODS = {"day": 86400, "week": 7 * 86400, "month": 30 * 86400}
_UNIDENT = ("Unknown", "LOW_QUALITY")   # неопознанные (уникальность не определяется)


def _object_names() -> dict:
    names = {}
    if os.path.exists(DB_PATH):
        with _db() as conn:
            try:
                for r in conn.execute("SELECT id, name FROM objects"):
                    names[r["id"]] = r["name"] or r["id"]
            except sqlite3.OperationalError:
                pass
    for o in load_objects():
        names.setdefault(o["id"], o.get("name", o["id"]))
    return names


@app.get("/api/analytics")
def api_analytics(period: str = Query("day"), object: str = Query("")):
    """
    По каждому объекту за период: уникальных ЛЮДЕЙ (distinct ID, без Unknown/LOW_QUALITY)
    и число событий Неопознанных (уникальность неизвестных не определяется).
    """
    if not os.path.exists(DB_PATH):
        return {"period": period, "rows": []}
    frm = time.time() - _PERIODS.get(period, _PERIODS["day"])
    ph = ",".join("?" * len(_UNIDENT))
    sql = (f"SELECT object_id, "
           f"COUNT(DISTINCT CASE WHEN person NOT IN ({ph}) THEN person END) AS uniq, "
           f"SUM(CASE WHEN person IN ({ph}) THEN 1 ELSE 0 END) AS unident "
           f"FROM events WHERE ts >= ?")
    params = list(_UNIDENT) + list(_UNIDENT) + [frm]
    if object:
        sql += " AND object_id = ?"; params.append(object)
    sql += " GROUP BY object_id"
    names = _object_names()
    rows = []
    with _db() as conn:
        for r in conn.execute(sql, params):
            rows.append({"object_id": r["object_id"],
                         "object_name": names.get(r["object_id"], r["object_id"]),
                         "unique_people": r["uniq"] or 0,
                         "unidentified": r["unident"] or 0})
    rows.sort(key=lambda x: -x["unique_people"])
    return {"period": period, "rows": rows}


@app.get("/api/person/{label}")
def api_person(label: str, period: str = Query("month")):
    """Карточка человека: объекты появления, всего, по дням, последние снимки."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="нет базы")
    frm = time.time() - _PERIODS.get(period, _PERIODS["month"])
    names = _object_names()

    # снимок из галереи
    face_url = ""
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            for idn in json.load(f).get("identities", []):
                if idn["label"] == label:
                    face_url = _face_url(idn["crop_path"]); break

    with _db() as conn:
        by_object = [{"object_id": r["object_id"],
                      "object_name": names.get(r["object_id"], r["object_id"]),
                      "count": r["c"]}
                     for r in conn.execute(
                         "SELECT object_id, COUNT(*) c FROM events WHERE person=? AND ts>=? "
                         "GROUP BY object_id ORDER BY c DESC", (label, frm))]
        total = sum(o["count"] for o in by_object)
        by_day = [{"day": r["d"], "count": r["c"]}
                  for r in conn.execute(
                      "SELECT date(ts,'unixepoch','localtime') d, COUNT(*) c FROM events "
                      "WHERE person=? AND ts>=? GROUP BY d ORDER BY d", (label, frm))]
        last = [{"ts": r["ts"], "camera_id": r["camera_id"], "zone": r["zone"],
                 "object_id": r["object_id"],
                 "object_name": names.get(r["object_id"], r["object_id"]),
                 "face_url": _face_url(r["crop_path"]), "full_url": _full_url(r["full_path"])}
                for r in conn.execute(
                    "SELECT ts, camera_id, zone, object_id, crop_path, full_path FROM events "
                    "WHERE person=? ORDER BY ts DESC LIMIT 12", (label,))]
    return {"label": label, "period": period, "face_url": face_url, "total": total,
            "by_object": by_object, "by_day": by_day, "last": last}


# ==================== API ИНТЕГРАЦИИ (v1) — для внешних систем ====================
# По object_id: список событий лиц (/api/v1/faces), уникальные люди (/api/v1/persons)
# и события транспорта (/api/v1/vehicles). Все — с фильтрами и фильтром по дате.
# Даты: unix timestamp, "YYYY-MM-DD" или "YYYY-MM-DDTHH:MM:SS" (локальное время сервера).
# URL снимков — абсолютные (можно открывать с другого хоста).

def _parse_ts(s: str, end_of_day: bool = False) -> float | None:
    """Строка даты из query -> unix ts. Пустая строка -> None (фильтр не задан)."""
    if not s:
        return None
    s = s.strip()
    try:
        return float(s)                      # unix timestamp
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d" and end_of_day:
                # date_to без времени = включительно весь день
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt.timestamp()
        except ValueError:
            continue
    raise HTTPException(status_code=422,
                        detail=f"неверный формат даты: {s!r} "
                               "(ожидается unix ts, YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS)")


def _iso(ts) -> str:
    return datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds") if ts else ""


def _abs(request: Request, rel_url: str) -> str:
    """Относительный URL снимка -> абсолютный (для внешних потребителей API)."""
    return (str(request.base_url).rstrip("/") + rel_url) if rel_url else ""


def _gallery_face_urls() -> dict:
    """label -> относительный URL снимка из галереи."""
    out = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            for idn in json.load(f).get("identities", []):
                out[idn["label"]] = _face_url(idn["crop_path"])
    return out


@app.get("/api/v1/faces")
def api_v1_faces(request: Request,
                 object_id: str = Query("", description="фильтр по объекту"),
                 camera_id: str = Query("", description="фильтр по камере"),
                 person: str = Query("", description="фильтр по ID человека (person_0001)"),
                 date_from: str = Query("", description="unix ts | YYYY-MM-DD | YYYY-MM-DDTHH:MM:SS"),
                 date_to: str = Query("", description="то же; YYYY-MM-DD — включительно"),
                 exclude_uncertain: int = Query(0, description="1 = убрать события «серой зоны»"),
                 exclude_unidentified: int = Query(0, description="1 = убрать Unknown/LOW_QUALITY"),
                 limit: int = Query(100, ge=1, le=1000),
                 offset: int = Query(0, ge=0)):
    """API 1: события ЛИЦ. Сортировка — новые сверху. total — всего под фильтром."""
    if not os.path.exists(DB_PATH):
        return {"total": 0, "limit": limit, "offset": offset, "items": []}
    where, params = [], []
    if object_id:
        where.append("object_id = ?"); params.append(object_id)
    if camera_id:
        where.append("camera_id = ?"); params.append(camera_id)
    if person:
        where.append("person = ?"); params.append(person)
    frm, to = _parse_ts(date_from), _parse_ts(date_to, end_of_day=True)
    if frm is not None:
        where.append("ts >= ?"); params.append(frm)
    if to is not None:
        where.append("ts <= ?"); params.append(to)
    if exclude_uncertain:
        where.append("uncertain = 0")
    if exclude_unidentified:
        ph = ",".join("?" * len(_UNIDENT))
        where.append(f"person NOT IN ({ph})"); params.extend(_UNIDENT)
    cond = (" WHERE " + " AND ".join(where)) if where else ""
    names = _object_names()
    with _db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM events{cond}", params).fetchone()[0]
        rows = conn.execute(
            "SELECT id, ts, camera_id, zone, person, score, is_new, uncertain, "
            f"crop_path, full_path, object_id FROM events{cond} "
            "ORDER BY ts DESC LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
    items = [{
        "id": r["id"], "ts": r["ts"], "datetime": _iso(r["ts"]),
        "object_id": r["object_id"],
        "object_name": names.get(r["object_id"], r["object_id"]),
        "camera_id": r["camera_id"], "zone": r["zone"],
        "person": r["person"],
        "score": round(r["score"], 3) if r["score"] is not None else None,
        "is_new": bool(r["is_new"]), "uncertain": bool(r["uncertain"]),
        "face_url": _abs(request, _face_url(r["crop_path"])),
        "full_url": _abs(request, _full_url(r["full_path"])),
    } for r in rows]
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/api/v1/persons")
def api_v1_persons(request: Request,
                   object_id: str = Query("", description="фильтр по объекту"),
                   date_from: str = Query(""), date_to: str = Query(""),
                   limit: int = Query(100, ge=1, le=1000),
                   offset: int = Query(0, ge=0)):
    """API 1а: УНИКАЛЬНЫЕ люди на объекте за период (агрегация событий по person)."""
    if not os.path.exists(DB_PATH):
        return {"total": 0, "limit": limit, "offset": offset, "items": []}
    ph = ",".join("?" * len(_UNIDENT))
    where, params = [f"person NOT IN ({ph})"], list(_UNIDENT)
    if object_id:
        where.append("object_id = ?"); params.append(object_id)
    frm, to = _parse_ts(date_from), _parse_ts(date_to, end_of_day=True)
    if frm is not None:
        where.append("ts >= ?"); params.append(frm)
    if to is not None:
        where.append("ts <= ?"); params.append(to)
    cond = " WHERE " + " AND ".join(where)
    faces = _gallery_face_urls()
    with _db() as conn:
        total = conn.execute(
            f"SELECT COUNT(DISTINCT person) FROM events{cond}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT person, COUNT(*) events, MIN(ts) first_seen, MAX(ts) last_seen, "
            f"GROUP_CONCAT(DISTINCT camera_id) cams FROM events{cond} "
            "GROUP BY person ORDER BY last_seen DESC LIMIT ? OFFSET ?",
            params + [limit, offset]).fetchall()
    items = [{
        "person": r["person"], "events": r["events"],
        "first_seen": r["first_seen"], "first_seen_dt": _iso(r["first_seen"]),
        "last_seen": r["last_seen"], "last_seen_dt": _iso(r["last_seen"]),
        "cameras": (r["cams"] or "").split(","),
        "face_url": _abs(request, faces.get(r["person"], "")),
    } for r in rows]
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/api/v1/vehicles")
def api_v1_vehicles(request: Request,
                    object_id: str = Query("", description="фильтр по объекту"),
                    camera_id: str = Query("", description="фильтр по камере"),
                    plate: str = Query("", description="поиск по номеру (подстрока)"),
                    valid: str = Query("", description="'1' — только валидные РУз, '0' — только невалидные"),
                    date_from: str = Query(""), date_to: str = Query(""),
                    limit: int = Query(100, ge=1, le=1000),
                    offset: int = Query(0, ge=0)):
    """API 2: события ТРАНСПОРТА. region/body разбираются из номера на лету."""
    if not os.path.exists(DB_PATH):
        return {"total": 0, "limit": limit, "offset": offset, "items": []}
    where, params = [], []
    if object_id:
        where.append("object_id = ?"); params.append(object_id)
    if camera_id:
        where.append("camera_id = ?"); params.append(camera_id)
    if plate:
        where.append("plate_normalized LIKE ?"); params.append(f"%{plate.upper()}%")
    if valid in ("0", "1"):
        where.append("valid = ?"); params.append(int(valid))
    frm, to = _parse_ts(date_from), _parse_ts(date_to, end_of_day=True)
    if frm is not None:
        where.append("timestamp >= ?"); params.append(frm)
    if to is not None:
        where.append("timestamp <= ?"); params.append(to)
    cond = (" WHERE " + " AND ".join(where)) if where else ""
    names = _object_names()
    try:
        with _db() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM vehicle_events{cond}", params).fetchone()[0]
            rows = conn.execute(
                "SELECT id, timestamp, camera_id, zone, plate_text, plate_normalized, "
                f"confidence, snapshot_path, full_path, valid, region_uncertain, object_id "
                f"FROM vehicle_events{cond} "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
    except sqlite3.OperationalError:
        return {"total": 0, "limit": limit, "offset": offset, "items": []}   # таблицы ещё нет
    items = []
    for r in rows:
        pp = _plate_validator().parse(r["plate_normalized"] or "")
        items.append({
            "id": r["id"], "ts": r["timestamp"], "datetime": _iso(r["timestamp"]),
            "object_id": r["object_id"],
            "object_name": names.get(r["object_id"], r["object_id"]),
            "camera_id": r["camera_id"], "zone": r["zone"],
            "plate": pp.normalized or r["plate_normalized"],
            "plate_raw": r["plate_text"],
            "region": pp.region, "body": pp.body,
            "valid": bool(r["valid"]), "region_uncertain": bool(r["region_uncertain"]),
            "confidence": round(r["confidence"], 3) if r["confidence"] is not None else None,
            "plate_url": _abs(request, _plate_url(r["snapshot_path"])),
            "full_url": _abs(request, _full_url(r["full_path"])),
        })
    return {"total": total, "limit": limit, "offset": offset, "items": items}


_PLATE_VALIDATOR = None


def _plate_validator():
    """Ленивый PlateValidator (разбор region/body в API транспорта)."""
    global _PLATE_VALIDATOR
    if _PLATE_VALIDATOR is None:
        from anpr.plate_format import PlateValidator
        _PLATE_VALIDATOR = PlateValidator(cfg["anpr"]["plate_regex"])
    return _PLATE_VALIDATOR


# ============================ LIVE (просмотр камеры с боксами) ============================
_live = None
_live_lock = threading.Lock()


def _get_live():
    """Ленивый LiveManager: движки грузятся при первом запросе live."""
    global _live
    if _live is None:
        with _live_lock:
            if _live is None:
                from live import LiveManager
                _live = LiveManager(cfg)
    return _live


@app.get("/live/stream/{cam_id}")
def live_stream(cam_id: str, det: int = Query(0, description="det_size лица: 0=деф, 640/960/1280/1600")):
    """MJPEG-поток одной камеры с боксами. Запускается по запросу (клику)."""
    lm = _get_live()
    if not lm.start(cam_id, det=det):
        raise HTTPException(status_code=404, detail=f"камера {cam_id} не найдена в cameras.yaml")

    def gen():
        boundary = b"--frame"
        while True:
            jpg = lm.get_jpeg(cam_id)      # обновляет last_access -> watchdog держит поток
            if jpg is not None:
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/live/snapshot/{cam_id}")
def live_snapshot(cam_id: str, det: int = Query(0)):
    lm = _get_live()
    if not lm.start(cam_id, det=det):
        raise HTTPException(status_code=404, detail="камера не найдена")
    for _ in range(50):                     # подождать первый кадр
        jpg = lm.get_jpeg(cam_id)
        if jpg:
            return Response(content=jpg, media_type="image/jpeg")
        time.sleep(0.1)
    return Response(status_code=503)


@app.post("/live/stop")
def live_stop(cam_id: str = Query("")):
    """Остановить live: конкретную камеру (cam_id) или все (пусто)."""
    _get_live().stop(cam_id or None)
    return {"stopped": cam_id or "all"}
