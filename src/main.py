# -*- coding: utf-8 -*-
r"""
main.py — запуск всех камер из cameras.yaml (лица + ANPR по режиму камеры).

Поток на камеру (чтение + reconnect) -> общая очередь -> один inference-поток
(общий GPU-движок) -> по режиму камеры (face/plate/both): лица и/или номера.
Модули лиц и ANPR собираются ТОЛЬКО если есть камеры соответствующего режима.

Запуск:
  python src\main.py
  python src\main.py --cameras config\cameras_test.yaml --max-seconds 12 --quiet
"""
import os
import sys
import time
import queue
import argparse
import threading

from config import load_settings, load_cameras
from face_engine import FaceEngine
from gallery import Gallery
from events import EventLog
from camera_worker import CameraWorker
from inference_worker import InferenceWorker, FrameResult

from anpr.engine import AnprEngine
from anpr.plate_format import PlateValidator
from anpr.vehicle_log import VehicleLog


def make_face_handler(event_log: EventLog, quiet: bool):
    """Колбэк лиц: печать + запись события (с анти-дребезгом)."""
    def on_result(r: FrameResult):
        if not r.faces:
            return
        lines = []
        for f in r.faces:
            wrote = event_log.log(r.cam_id, r.zone, f.label, f.score,
                                  f.is_new, f.crop_path, ts=time.time())
            tag = "NEW" if f.is_new else ("LOG" if wrote else "...")
            if f.is_new or wrote or not quiet:
                lines.append(f"{f.label}:{f.score:.2f}[{tag}]")
        if lines:
            print(f"[{r.cam_id}/{r.zone}] faces={len(r.faces)} {' '.join(lines)} "
                  f"lat={r.latency_ms:.0f}ms")
    return on_result


def make_plate_handler(quiet: bool):
    """Колбэк ANPR: печать распознанных номеров (запись в БД — внутри pipeline)."""
    def on_plate(cam_id, zone, results, infer_ms, latency_ms):
        shown = []
        for r in results:
            if quiet and not r["logged"]:
                continue
            flag = "OK" if r["valid"] else ("регион?" if r["region_uncertain"] else "невалид")
            tag = "LOG" if r["logged"] else "..."
            shown.append(f"{r['normalized']}({flag})[{tag}]")
        if shown:
            print(f"[{cam_id}/{zone}] PLATE {' '.join(shown)} "
                  f"lat={latency_ms:.0f}ms infer={infer_ms:.0f}ms")
    return on_plate


def stats_printer(workers, infer: InferenceWorker, q, gallery, vehicle_log,
                  stop_event, period: float = 5.0):
    while not stop_event.is_set():
        end = time.time() + period
        while time.time() < end and not stop_event.is_set():
            time.sleep(0.2)
        if stop_event.is_set():
            break
        print("\n================= СТАТУС КАМЕР =================")
        print(f"{'cam':8s} {'zone':18s} {'mode':5s} {'conn':5s} {'fps':>5s} "
              f"{'read':>7s} {'drop':>6s} {'recon':>5s}  err")
        for w in workers:
            s = w.stats
            mode = w.cam.get("mode", "face")
            print(f"{s.cam_id:8s} {s.zone[:18]:18s} {mode:5s} {'OK' if s.connected else '--':5s} "
                  f"{s.fps_ema:5.1f} {s.frames_read:7d} {s.drops:6d} {s.reconnects:5d}  {s.last_error}")
        # лица и ANPR — отдельно (по ТЗ ANPR логируется отдельно)
        face_line = (f"ЛИЦА: кадров={infer.processed} avg_infer={infer.infer_ms_ema:.1f}ms "
                     f"ID={gallery.count() if gallery else 0}") if gallery else "ЛИЦА: выкл"
        anpr_line = (f"ANPR: кадров={infer.anpr_processed} avg_infer={infer.anpr_infer_ms_ema:.1f}ms"
                     if infer.anpr_engine else "ANPR: выкл")
        print(f"queue={q.qsize():3d}  {face_line}  |  {anpr_line}")
        print("===============================================\n")


def main():
    ap = argparse.ArgumentParser(description="Камеры: лица + ANPR по режиму")
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--queue-size", type=int, default=50)
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_settings()
    cameras = load_cameras(args.cameras)
    if not cameras:
        print("Нет камер в cameras.yaml.")
        return 1

    modes = {c["id"]: c.get("mode", "face") for c in cameras}
    need_face = any(m in ("face", "both") for m in modes.values())
    need_anpr = any(m in ("plate", "both") for m in modes.values())
    print(f"Камер: {len(cameras)}. Режимы: "
          f"{sum(m in ('face','both') for m in modes.values())} c лицами, "
          f"{sum(m in ('plate','both') for m in modes.values())} c ANPR.")

    # --- Лица (если есть face/both) ---
    engine = gallery = event_log = None
    if need_face:
        print("Инициализация движка лиц...")
        engine = FaceEngine(
            model_name=cfg["recognition"]["model_name"],
            det_size=(cfg["recognition"]["det_size"], cfg["recognition"]["det_size"]),
            ctx_id=cfg["gpu"]["ctx_id"],
            allowed_modules=["detection", "recognition"],
        )
        FaceEngine.warmup(engine, size=cfg["recognition"]["det_size"])
        gallery = Gallery(cfg)
        event_log = EventLog(cfg["paths"]["db"], dedup_seconds=cfg["events"]["dedup_seconds"])
        print(f"  лица GPU={engine.on_gpu}, ID в галерее={gallery.count()}")

    # --- ANPR (если есть plate/both) ---
    anpr_engine = anpr_validator = vehicle_log = None
    plates_dir = cfg["paths"]["plates"]
    if need_anpr:
        print("Инициализация движка ANPR...")
        anpr_engine = AnprEngine(cfg)
        anpr_validator = PlateValidator(cfg["anpr"]["plate_regex"])
        vehicle_log = VehicleLog(cfg["paths"]["db"], dedup_seconds=cfg["anpr"]["dedup_seconds"])
        print(f"  ANPR GPU={anpr_engine.on_gpu}")

    q = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()

    workers = [CameraWorker(cam, q, cfg, stop_event) for cam in cameras]
    infer = InferenceWorker(
        q, engine, gallery, cfg, stop_event,
        make_face_handler(event_log, args.quiet) if need_face else (lambda r: None),
        cam_modes=modes,
        anpr_engine=anpr_engine, anpr_validator=anpr_validator,
        vehicle_log=vehicle_log, plates_dir=plates_dir,
        on_plate=make_plate_handler(args.quiet) if need_anpr else None,
    )
    stats_t = threading.Thread(target=stats_printer,
                               args=(workers, infer, q, gallery, vehicle_log, stop_event),
                               daemon=True, name="stats")

    print("Старт потоков...\n")
    infer.start()
    for w in workers:
        w.start()
    stats_t.start()

    t_start = time.time()
    try:
        while True:
            time.sleep(0.3)
            if args.max_seconds and (time.time() - t_start) >= args.max_seconds:
                print(f"\nДостигнут --max-seconds={args.max_seconds}.")
                break
    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C...")
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=3)
        infer.join(timeout=3)
        if gallery:
            gallery.save()
        if event_log:
            event_log.close()
        if vehicle_log:
            vehicle_log.close()

    print("Остановлено.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
