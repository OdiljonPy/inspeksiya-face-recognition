# -*- coding: utf-8 -*-
r"""
stage2_recognize.py — Этап 2. Распознавание на одном источнике.

Для каждого найденного лица: эмбеддинг -> поиск в FAISS -> "Имя" или "Unknown".
Имя/score выводятся в консоль и рисуются на кадре (зелёный = свой, красный = Unknown).

Источник как на Этапе 1:
  --source rtsp://... | 0 | video.mp4 | selftest

Перед запуском нужен индекс: сначала python src\enroll.py

Запуск:
  python src\stage2_recognize.py --source "rtsp://192.168.100.133/live"
  python src\stage2_recognize.py --source selftest
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np

from config import load_settings, index_paths
from face_engine import FaceEngine
from recognizer import Recognizer
# переиспользуем низкоуровневые помощники Этапа 1
from stage1_single_stream import open_capture, resize_to_width, draw_hud


def draw_named_faces(frame, faces, recognizer: Recognizer, min_det_score: float,
                     console: bool = False) -> list[str]:
    """Распознать и нарисовать каждое лицо. Вернуть список строк для лога."""
    lines = []
    for f in faces:
        if float(getattr(f, "det_score", 0.0)) < min_det_score:
            continue
        m = recognizer.identify(f.normed_embedding)
        x1, y1, x2, y2 = f.bbox.astype(int)
        color = (0, 200, 0) if m.matched else (0, 0, 230)   # зелёный/красный (BGR)
        label = f"{m.name} {m.score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        line = f"{m.name:>12s}  score={m.score:.3f}  (best={m.best_name or '-'})"
        lines.append(line)
        if console:
            print("   " + line)
    return lines


def build_recognizer(cfg) -> Recognizer:
    index_path, labels_path = index_paths(cfg)
    return Recognizer.from_files(index_path, labels_path,
                                 threshold=cfg["recognition"]["cosine_threshold"])


# --------------------------------------------------------------------------- #
def run_selftest(engine, recognizer, cfg) -> int:
    """Офлайн-проверка: распознать Тома Хэнкса и группу из t1."""
    from insightface.data import get_image as ins_get_image
    from enroll import detect_robust   # тот же фолбэк с паддингом для тесных кропов
    width = cfg["recognition"]["width"]
    min_ds = cfg["recognition"]["min_det_score"]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    for tag in ("Tom_Hanks_54745", "t1"):
        img = ins_get_image(tag)
        if img is None:
            print(f"  ! нет тест-изображения {tag}")
            continue
        small, _ = resize_to_width(img, width)
        faces = detect_robust(engine, small)
        print(f"\n[{tag}] лиц={len(faces)}:")
        draw_named_faces(small, faces, recognizer, min_ds, console=True)
        draw_hud(small, [f"SELFTEST {tag}  faces={len(faces)}  thr={recognizer.threshold}"])
        out = os.path.join(out_dir, f"stage2_{tag}.jpg")
        cv2.imwrite(out, small)
        print(f"  -> {out}")
    return 0


def run_stream(engine, recognizer, cfg, source, headless, max_frames) -> int:
    width = cfg["recognition"]["width"]
    target_fps = cfg["recognition"]["target_fps"]
    min_ds = cfg["recognition"]["min_det_score"]

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"ОШИБКА: не открыть источник {source}. Для теста: --source selftest")
        return 1

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    is_file = frame_count and frame_count > 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"Источник: {source} | {'файл' if is_file else 'live'} | thr={recognizer.threshold} "
          f"| в базе {len(recognizer.names)} чел.")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    processed = 0
    min_interval = 1.0 / max(0.1, target_fps)
    last_proc = 0.0
    stride = max(1, int(round((src_fps or target_fps) / max(0.1, target_fps))))
    idx = 0
    fail = 0

    try:
        while True:
            if is_file:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("EOF (файл закончился).")
                    break
                idx += 1
                if (idx - 1) % stride != 0:
                    continue
            else:
                if not cap.grab():
                    fail += 1
                    if fail > 50:
                        print("Поток прервался. Авто-reconnect будет на Этапе 3.")
                        break
                    time.sleep(0.02); continue
                fail = 0
                now = time.time()
                if now - last_proc < min_interval:
                    continue
                last_proc = now
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue

            small, _ = resize_to_width(frame, width)
            t0 = time.time()
            faces = engine.detect(small)
            det_ms = (time.time() - t0) * 1000
            processed += 1

            print(f"[{processed:4d}] лиц={len(faces)} detect={det_ms:.1f}ms")
            draw_named_faces(small, faces, recognizer, min_ds, console=True)
            draw_hud(small, [f"faces={len(faces)} detect={det_ms:.1f}ms "
                             f"GPU={engine.on_gpu} thr={recognizer.threshold}"])

            if headless:
                cv2.imwrite(os.path.join(out_dir, "stage2_last.jpg"), small)
            else:
                cv2.imshow("Stage2 - recognition (q to quit)", small)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
    print(f"Готово. Обработано кадров: {processed}.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Этап 2: распознавание на одном источнике")
    ap.add_argument("--source", default="selftest")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_settings()
    print("Загрузка базы и инициализация движка...")
    try:
        recognizer = build_recognizer(cfg)
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}")
        return 1

    engine = FaceEngine(
        model_name=cfg["recognition"]["model_name"],
        det_size=(cfg["recognition"]["det_size"], cfg["recognition"]["det_size"]),
        ctx_id=cfg["gpu"]["ctx_id"],
        allowed_modules=["detection", "recognition"],
    )
    FaceEngine.warmup(engine, size=cfg["recognition"]["det_size"])
    print(f"GPU = {engine.on_gpu}, людей в базе: {len(recognizer.names)} "
          f"({', '.join(recognizer.names)})")

    if args.source == "selftest":
        return run_selftest(engine, recognizer, cfg)
    return run_stream(engine, recognizer, cfg, args.source, args.headless, args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
