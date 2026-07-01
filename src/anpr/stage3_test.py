# -*- coding: utf-8 -*-
"""stage3_test.py — проверка Этапа 3: лог в SQLite + кропы + дедуп (офлайн по фото)."""
import os
import sys
import glob

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import sqlite3
import cv2
from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()

from config import load_settings
from anpr.engine import AnprEngine
from anpr.plate_format import PlateValidator
from anpr.vehicle_log import VehicleLog
from anpr.pipeline import process_frame

cfg = load_settings()
eng = AnprEngine(cfg)
val = PlateValidator(cfg["anpr"]["plate_regex"])
db = os.path.join(_SRC, "..", "data", "_anpr_test.db")
db = os.path.normpath(db)
plates_dir = os.path.join(_SRC, "..", "data", "plates")
plates_dir = os.path.normpath(plates_dir)

if os.path.exists(db):
    os.remove(db)
vlog = VehicleLog(db, dedup_seconds=cfg["anpr"]["dedup_seconds"])

base = os.path.join(_SRC, "..", "data", "anpr_test")
files = sorted(glob.glob(os.path.join(base, "*.png")))
for i, f in enumerate(files):
    img = cv2.imread(f)
    res = process_frame(eng, val, vlog, img, f"cam{i%2+1:02d}", "Тест-зона",
                        plates_dir, cfg["anpr"]["min_ocr_confidence"])
    for r in res:
        print(f"{os.path.basename(f)}: {r['normalized']} valid={r['valid']} "
              f"рег?={r['region_uncertain']} logged={r['logged']}")

print("\n--- ДЕДУП: повтор первого фото на той же камере сразу ---")
res = process_frame(eng, val, vlog, cv2.imread(files[0]), "cam01", "Тест-зона",
                    plates_dir, cfg["anpr"]["min_ocr_confidence"])
print("logged (ожидаем False):", [r["logged"] for r in res])

c = sqlite3.connect(db)
print("\nвсего строк vehicle_events:", c.execute("select count(*) from vehicle_events").fetchone()[0])
print("кропов в data/plates:", len(glob.glob(os.path.join(plates_dir, "*.jpg"))))
print("\nстроки таблицы:")
for row in c.execute("select camera_id, plate_normalized, round(confidence,2), valid, "
                     "region_uncertain, snapshot_path from vehicle_events order by id"):
    print("  ", row[:5], os.path.basename(row[5] or ""))
