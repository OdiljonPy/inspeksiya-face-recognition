# -*- coding: utf-8 -*-
r"""
check_anpr.py — Этап 0 ANPR. Диагностика провайдера + прогон на изображении.

Печатает: версию onnxruntime и providers, реально ли fast-alpr использует GPU,
и (если задан --image) распознанный номер + confidence, сохраняет кроп с разметкой.

Запуск:
  .\.venv\Scripts\python.exe src\anpr\check_anpr.py --image D:\path\car.jpg
  .\.venv\Scripts\python.exe src\anpr\check_anpr.py           # только инициализация/GPU
"""
import os
import sys
import time
import argparse

# доступ к src/ (config, gpu_setup) и к пакету anpr
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import glob
import cv2
from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()

from config import load_settings
from anpr.engine import AnprEngine
from anpr.plate_format import PlateValidator

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _process_image(engine, validator, path, out_dir):
    img = cv2.imread(path)
    if img is None:
        print(f"  ! не читается: {path}")
        return
    plates = engine.predict(img)
    name = os.path.basename(path)
    if not plates:
        print(f"  {name:28s} -> номер не найден")
    for p in plates:
        norm = validator.normalize(p.text)
        valid = validator.is_valid(norm)
        print(f"  {name:28s} -> '{p.text}'  norm='{norm}'  "
              f"ocr={p.ocr_conf:.3f} det={p.det_conf:.3f}  "
              f"формат_РУз={'OK' if valid else 'нет'}")
        if p.bbox:
            x1, y1, x2, y2 = p.bbox
            color = (0, 200, 0) if valid else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, f"{norm} {p.ocr_conf:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    out = os.path.join(out_dir, "out_" + os.path.basename(path))
    cv2.imwrite(out, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="путь к тест-изображению с авто")
    ap.add_argument("--dir", default="", help="папка с изображениями (пакетный прогон)")
    ap.add_argument("--cpu", action="store_true", help="принудительно CPU")
    args = ap.parse_args()

    print("=" * 60)
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}, device={ort.get_device()}")
    print(f"providers: {ort.get_available_providers()}")
    print("=" * 60)

    cfg = load_settings()
    print(f"Инициализация AnprEngine (detector={cfg['anpr']['detector_model']}, "
          f"ocr={cfg['anpr']['ocr_model']})...")
    print("(при первом запуске модели скачиваются)")
    t0 = time.time()
    engine = AnprEngine(cfg, prefer_gpu=not args.cpu)
    print(f"Готово за {time.time()-t0:.1f}s. ANPR на GPU = {engine.on_gpu}")

    validator = PlateValidator(cfg["anpr"]["plate_regex"])

    if not args.image and not args.dir:
        print("\nИзображение не задано. Движок инициализирован, GPU проверен.")
        return 0

    out_dir = os.path.normpath(os.path.join(_SRC, "..", "data", "anpr_test_out"))
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    if args.image:
        paths.append(args.image)
    if args.dir:
        for ext in IMG_EXT:
            paths += glob.glob(os.path.join(args.dir, ext))
    paths = sorted(set(paths))
    if not paths:
        print(f"Нет изображений в {args.dir or args.image}")
        return 1

    print(f"\nПрогон по {len(paths)} изображениям (порог OCR в проде = "
          f"{cfg['anpr']['min_ocr_confidence']}):")
    # прогрев
    engine.predict(cv2.imread(paths[0]))
    for path in paths:
        _process_image(engine, validator, path, out_dir)
    print(f"\nРазмеченные кадры: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
