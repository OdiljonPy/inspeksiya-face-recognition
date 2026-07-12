# -*- coding: utf-8 -*-
r"""
eval_quality.py — оценка качества распознавания на тестовых клипах/фото БЕЗ прода.

Гоняет полный конвейер (движок -> трекер -> галерея / ANPR -> parse -> лог) на
изолированных данных (data/_eval/*, прод data/gallery и events.db НЕ трогает).

Метрики лиц (на клип):
  - детекции, распределение det_score / px / frontality / blur;
  - создано ID (идеал: 1 человек в клипе = 1 ID), матчи, uncertain;
  - ПРОХОД 2 по тому же клипу: новых ID быть НЕ должно (стабильность матчинга).

Метрики ANPR:
  - события, valid, region_uncertain, восстановления региона (fix/ocr), апдейты-голосования.

Запуск: python scripts\eval_quality.py [--det 640] [--videos data\_cam_known.mp4 ...]
"""
import os
import sys
import glob
import time
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np

from config import load_settings
from face_engine import FaceEngine
from gallery import Gallery, frontality, blur_var
from tracker import CameraTracker

EVAL_DIR = os.path.join("data", "_eval")


def pctl(vals, p):
    return float(np.percentile(vals, p)) if vals else 0.0


def eval_face_video(engine, cfg, path, min_det):
    """Два прохода по клипу со СВЕЖЕЙ галереей. Возвращает dict метрик."""
    name = os.path.splitext(os.path.basename(path))[0]
    gal_dir = os.path.join(EVAL_DIR, f"gallery_{name}")
    shutil.rmtree(gal_dir, ignore_errors=True)
    eval_cfg = {**cfg, "gallery": {**cfg["gallery"], "dir": gal_dir}}
    g = Gallery(eval_cfg)

    stats = {"video": name, "frames": 0, "detections": 0,
             "det_scores": [], "px": [], "front": [], "blur": [],
             "pass1": {}, "pass2": {}}

    for pass_no in (1, 2):
        tr = CameraTracker(g, eval_cfg)
        ids_before = g.count()
        results, uncertain, new_ids = 0, 0, 0
        cap = cv2.VideoCapture(path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if pass_no == 1:
                stats["frames"] += 1
            faces = engine.detect(frame)
            faces = [f for f in faces if float(f.det_score) >= min_det]
            if pass_no == 1:
                for f in faces:
                    stats["detections"] += 1
                    stats["det_scores"].append(float(f.det_score))
                    x1, y1, x2, y2 = f.bbox
                    stats["px"].append(float(min(x2 - x1, y2 - y1)))
                    stats["front"].append(frontality(f.kps))
                    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                    stats["blur"].append(blur_var(crop) if crop.size else 0.0)
            for r in tr.update(faces, frame, time.time()):
                results += 1
                uncertain += int(r.uncertain)
                new_ids += int(r.is_new)
        cap.release()
        stats[f"pass{pass_no}"] = {
            "ids_created": g.count() - ids_before, "results": results,
            "uncertain": uncertain, "new_id_events": new_ids,
        }
    stats["total_ids"] = g.count()
    stats["emb_total"] = int(g.embeddings.shape[0])
    return stats


def print_face_stats(s):
    d = s
    print(f"\n=== {d['video']} ===")
    print(f"кадров {d['frames']}, детекций {d['detections']}")
    if d["det_scores"]:
        print(f"  det_score: p10={pctl(d['det_scores'],10):.2f} med={pctl(d['det_scores'],50):.2f} p90={pctl(d['det_scores'],90):.2f}")
        print(f"  px:        p10={pctl(d['px'],10):.0f} med={pctl(d['px'],50):.0f} p90={pctl(d['px'],90):.0f}")
        print(f"  frontal:   med={pctl(d['front'],50):.2f}   blur: med={pctl(d['blur'],50):.0f}")
    for p in ("pass1", "pass2"):
        pp = d[p]
        print(f"  {p}: ID создано={pp['ids_created']} событий={pp['results']} "
              f"uncertain={pp['uncertain']}")
    print(f"  итог: ID={d['total_ids']} эмбеддингов={d['emb_total']}")
    # флаги проблем
    if d["pass2"]["ids_created"] > 0:
        print("  [!] ПРОБЛЕМА: проход 2 создал новые ID — матчинг нестабилен")


def eval_anpr(cfg, video, img_glob):
    from anpr.engine import AnprEngine
    from anpr.plate_format import PlateValidator
    from anpr.vehicle_log import VehicleLog
    from anpr.pipeline import process_frame
    from anpr.region_ocr import RegionOCR

    db = os.path.join(EVAL_DIR, "anpr.db")
    if os.path.exists(db):
        os.remove(db)
    plates_dir = os.path.join(EVAL_DIR, "plates")
    full_dir = os.path.join(EVAL_DIR, "full")
    shutil.rmtree(plates_dir, ignore_errors=True)
    shutil.rmtree(full_dir, ignore_errors=True)

    eng = AnprEngine(cfg)
    v = PlateValidator(cfg["anpr"]["plate_regex"])
    ro = RegionOCR() if cfg["anpr"].get("region_ocr") else None
    vl = VehicleLog(db, dedup_seconds=cfg["anpr"]["dedup_seconds"])
    min_conf = cfg["anpr"]["min_ocr_confidence"]
    min_px = int(cfg["anpr"].get("min_plate_px", 0))

    frames = plates_seen = 0
    t0 = time.time()
    sources = []
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            sources.append((f"video:{frames}", fr, frames * 0.5))
            frames += 1
        cap.release()
    for p in sorted(glob.glob(img_glob)):
        img = cv2.imread(p)
        if img is not None:
            sources.append((os.path.basename(p), img, 10000.0 + len(sources)))

    for tag, frame, ts in sources:
        res = process_frame(eng, v, vl, frame, "eval", "eval", plates_dir,
                            min_conf, ts=ts, object_id="eval",
                            full_dir=full_dir, region_ocr=ro, min_plate_px=min_px)
        plates_seen += len(res)

    rows = vl.conn.execute(
        "SELECT plate_normalized, confidence, valid, region_uncertain FROM vehicle_events").fetchall()
    vl.close()
    print(f"\n=== ANPR ({len(sources)} кадров, {(time.time()-t0):.1f}s) ===")
    print(f"чтений выше порога: {plates_seen}, событий в БД: {len(rows)}")
    for r in rows:
        flag = "OK" if r[1] and r[2] == 0 else ""
        print(f"  {r[0]:12s} conf={r[1]:.2f} valid={r[2]} region_unc={r[3]}")
    n_unc = sum(1 for r in rows if r[3])
    n_valid = sum(1 for r in rows if r[2])
    print(f"итог: valid={n_valid}/{len(rows)}, region_uncertain={n_unc}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", type=int, default=640)
    ap.add_argument("--videos", nargs="*", default=[
        "data/_cam_known.mp4", "data/_cam_unknown.mp4", "data/_cam_both.mp4"])
    ap.add_argument("--anpr-video", default="data/_cam_plate.mp4")
    ap.add_argument("--anpr-imgs", default="data/anpr_test/img*.png")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-anpr", action="store_true")
    args = ap.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)
    cfg = load_settings()

    if not args.no_face:
        print(f"Движок лиц det_size={args.det}...")
        engine = FaceEngine(det_size=(args.det, args.det),
                            ctx_id=cfg["gpu"]["ctx_id"],
                            allowed_modules=["detection", "recognition"])
        FaceEngine.warmup(engine, size=args.det)
        min_det = cfg["recognition"]["min_det_score"]
        for vpath in args.videos:
            if os.path.exists(vpath):
                print_face_stats(eval_face_video(engine, cfg, vpath, min_det))
            else:
                print(f"[skip] нет файла {vpath}")

    if not args.no_anpr:
        eval_anpr(cfg, args.anpr_video, args.anpr_imgs)


if __name__ == "__main__":
    main()
