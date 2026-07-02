# -*- coding: utf-8 -*-
r"""
tune_quality.py — прогон фильтра качества по папке снимков лиц.

Печатает метрики каждого лица (det_score, размер px, blur, yaw-asym) и вердикт
PASS/FAIL по текущим порогам из config — чтобы подобрать пороги под реальные
кадры с камер.

Запуск:
  python src\tune_quality.py --dir data\lowq
  python src\tune_quality.py --dir data\gallery\faces --save   # + разметка в data\quality_out
"""
import os
import sys
import glob
import argparse

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

import cv2
from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()

from config import load_settings
from face_engine import FaceEngine
from face_quality import FaceQuality

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def detect_robust(engine, img):
    """Детекция с паддинг-фолбэком (тесные кропы лиц)."""
    faces = engine.detect(img)
    if faces:
        return faces
    h, w = img.shape[:2]
    pad = int(0.4 * max(h, w))
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return engine.detect(padded)


def largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    ap = argparse.ArgumentParser(description="Подбор порогов фильтра качества")
    ap.add_argument("--dir", required=True, help="папка со снимками лиц")
    ap.add_argument("--save", action="store_true", help="сохранять размеченные кадры")
    args = ap.parse_args()

    cfg = load_settings()
    fq = FaceQuality(cfg)
    engine = FaceEngine(det_size=(640, 640), allowed_modules=["detection"])

    paths = []
    for ext in IMG_EXT:
        paths += glob.glob(os.path.join(args.dir, ext))
    paths = sorted(set(paths))
    if not paths:
        print(f"Нет изображений в {args.dir}")
        return 1

    out_dir = os.path.join(_SRC, "..", "data", "quality_out")
    if args.save:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Пороги: det>={fq.min_det} px>={fq.min_px:.0f} blur>={fq.min_blur:.0f} "
          f"asym<={fq.max_asym}\n")
    print(f"{'файл':32s} {'det':>5s} {'px':>6s} {'blur':>7s} {'yaw':>5s}  вердикт")
    npass = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        faces = detect_robust(engine, img)
        name = os.path.basename(p)
        if not faces:
            print(f"{name:32s} {'--':>5s} {'--':>6s} {'--':>7s} {'--':>5s}  нет лица")
            continue
        f = largest(faces)
        q = fq.assess(f, img, scale=1.0)
        verdict = "PASS" if q.passed else f"FAIL ({q.reason})"
        npass += q.passed
        print(f"{name:32s} {q.det_score:5.2f} {q.width_px:6.0f} {q.blur:7.1f} "
              f"{q.yaw_asym:5.2f}  {verdict}")
        if args.save:
            x1, y1, x2, y2 = f.bbox.astype(int)
            color = (0, 200, 0) if q.passed else (0, 0, 230)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, verdict, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, "q_" + name), img)

    print(f"\nИтого: {npass}/{len(paths)} прошли фильтр.")
    if args.save:
        print(f"Размеченные кадры: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
