# -*- coding: utf-8 -*-
r"""
stage2_anpr_stream.py — Этап 2 ANPR. Один источник: детекция номера -> plate_text в консоль.

Переиспускает помощники чтения из модуля лиц (stage1_single_stream): open_capture,
resize_to_width. Источник:
  --source rtsp://... | путь к видео | 0 (вебка)
  --dir <папка с фото>   офлайн-демо по изображениям (у нас есть тест-фото)

Интерим-логика: тело номера надёжно, регион — best-effort (флаг region?).

Запуск:
  python src\anpr\stage2_anpr_stream.py --dir data\anpr_test
  python src\anpr\stage2_anpr_stream.py --source "rtsp://admin:admin123@192.168.100.99:554/cam/realmonitor?channel=1&subtype=0"
"""
import os
import sys
import time
import glob
import argparse

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import cv2
from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()

from config import load_settings
from anpr.engine import AnprEngine
from anpr.plate_format import PlateValidator
from stage1_single_stream import open_capture, resize_to_width   # переиспользуем чтение

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def handle_frame(engine, validator, frame, min_conf, src_tag):
    """Распознать номера в кадре и напечатать. Вернуть число валидных/принятых."""
    plates = engine.predict(frame)
    n = 0
    for p in plates:
        if p.ocr_conf < min_conf:
            continue
        pp = validator.parse(p.text)
        n += 1
        flag = "OK" if pp.valid else ("регион?" if pp.region_uncertain else "невалид")
        print(f"  [{src_tag}] plate='{pp.normalized}' (тело={pp.body} регион={pp.region}) "
              f"conf={p.ocr_conf:.2f} det={p.det_conf:.2f} -> {flag}")
    if not plates:
        print(f"  [{src_tag}] номер не найден")
    return n


def run_dir(engine, validator, folder, min_conf):
    paths = []
    for ext in IMG_EXT:
        paths += glob.glob(os.path.join(folder, ext))
    paths = sorted(set(paths))
    if not paths:
        print(f"Нет изображений в {folder}")
        return 1
    print(f"Офлайн-демо по {len(paths)} изображениям:")
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        handle_frame(engine, validator, img, min_conf, os.path.basename(path))
    return 0


def run_stream(engine, validator, source, width, target_fps, min_conf, max_frames):
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"ОШИБКА: не открыть источник {source}. Для теста: --dir data\\anpr_test")
        return 1
    is_file = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    print(f"Источник: {source} | {'файл' if is_file else 'live'} | цель ~{target_fps} fps")
    min_interval = 1.0 / max(0.1, target_fps)
    last = 0.0
    processed = 0
    fail = 0
    try:
        while True:
            if not cap.grab():
                if is_file:
                    break
                fail += 1
                if fail > 50:
                    print("Поток прервался."); break
                time.sleep(0.02); continue
            fail = 0
            now = time.time()
            if now - last < min_interval:
                continue
            last = now
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            small, _ = resize_to_width(frame, width)
            t0 = time.time()
            handle_frame(engine, validator, small, min_conf, "cam")
            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
    print(f"Обработано кадров: {processed}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Этап 2 ANPR: один источник -> plate_text")
    ap.add_argument("--source", default="", help="rtsp/видео/0")
    ap.add_argument("--dir", default="", help="папка с изображениями (офлайн-демо)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_settings()
    print("Инициализация AnprEngine...")
    engine = AnprEngine(cfg)
    validator = PlateValidator(cfg["anpr"]["plate_regex"])
    min_conf = cfg["anpr"]["min_ocr_confidence"]
    print(f"ANPR на GPU = {engine.on_gpu}, порог OCR = {min_conf}\n")

    if args.dir:
        return run_dir(engine, validator, args.dir, min_conf)
    if args.source:
        return run_stream(engine, validator, args.source,
                          cfg["recognition"]["width"], cfg["anpr"].get("target_fps",
                          cfg["recognition"]["target_fps"]), min_conf, args.max_frames)
    print("Укажи --source или --dir")
    return 1


if __name__ == "__main__":
    sys.exit(main())
