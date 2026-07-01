# -*- coding: utf-8 -*-
r"""
stage1_single_stream.py — Этап 1.

Чтение ОДНОГО видеопотока + детекция лиц с отрисовкой боксов.
Источник гибкий:
  --source rtsp://user:pass@ip:554/...   RTSP-камера (FFmpeg, TCP-транспорт)
  --source 0                              вебкамера (индекс устройства)
  --source D:\path\video.mp4              локальный видеофайл
  --source selftest                       офлайн-самопроверка на встроенном
                                          тест-изображении InsightFace (без камеры/окна)

Ключевые приёмы (по ТЗ):
  - НЕ обрабатываем каждый кадр: берём ~target-fps кадров/сек (frame skip по времени);
    лишние кадры дренируем grab(), чтобы не копилась задержка RTSP.
  - Перед детекцией ресайз до ширины ~960 px.
  - Логируем FPS детекции и латентность.

Запуск (на машине с экраном — откроется окно с боксами):
  python src\stage1_single_stream.py --source "rtsp://admin:admin123@12.6.0.8:8554/cam/realmonitor?channel=1&subtype=0"
  python src\stage1_single_stream.py --source 0
  python src\stage1_single_stream.py --source selftest        # проверка без камеры
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np

from face_engine import FaceEngine


# ---- RTSP через FFmpeg: TCP-транспорт + таймауты (ставить ДО VideoCapture) ----
# stimeout/timeout — в микросекундах; TCP надёжнее UDP при потерях на стройке.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;5000000|buffer_size;1024000",
)


def resize_to_width(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Ресайз до заданной ширины с сохранением пропорций. Возвращает (кадр, scale)."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame, 1.0
    scale = width / float(w)
    new = cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return new, scale


def draw_faces(frame: np.ndarray, faces) -> np.ndarray:
    """Нарисовать боксы и det_score по найденным лицам."""
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        score = float(getattr(f, "det_score", 0.0))
        label = f"face {score:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Полупрозрачная панель с метриками FPS/латентности."""
    y = 22
    for ln in lines:
        cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 220, 255), 1, cv2.LINE_AA)
        y += 24


def open_capture(source: str) -> cv2.VideoCapture:
    """Открыть источник. '0','1'... -> вебкамера; иначе RTSP/файл через FFmpeg."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))          # вебкамера (DirectShow по умолч.)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    # Минимизируем внутренний буфер (не на всех бэкендах срабатывает)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


# --------------------------------------------------------------------------- #
#  Режим самопроверки: офлайн, без камеры и без окна                          #
# --------------------------------------------------------------------------- #
def run_selftest(engine: FaceEngine, width: int) -> int:
    """Прогон детекции на встроенном тест-изображении InsightFace. Сохраняет результат."""
    from insightface.data import get_image as ins_get_image
    img = ins_get_image("t1")  # групповое фото, идёт в комплекте insightface (офлайн)
    if img is None:
        print("ОШИБКА: не удалось загрузить тест-изображение InsightFace 't1'.")
        return 1

    small, _ = resize_to_width(img, width)

    t = time.time()
    faces = engine.detect(small)
    dt = (time.time() - t) * 1000

    draw_faces(small, faces)
    draw_hud(small, [f"SELFTEST  faces={len(faces)}  detect={dt:.1f} ms",
                     f"GPU={engine.on_gpu}"])

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage1_selftest.jpg")
    cv2.imwrite(out_path, small)

    print(f"SELFTEST OK: найдено лиц = {len(faces)}, детекция {dt:.1f} ms, GPU={engine.on_gpu}")
    print(f"Аннотированный кадр сохранён: {out_path}")
    return 0 if len(faces) > 0 else 2


# --------------------------------------------------------------------------- #
#  Основной цикл по потоку                                                     #
# --------------------------------------------------------------------------- #
def _process_and_show(engine, frame, width, headless, out_dir, processed,
                      fps_ema, extra_hud=""):
    """Общий шаг: ресайз -> детекция -> отрисовка -> показ/сохранение. Возвращает (detect_ms, faces, quit)."""
    small, _ = resize_to_width(frame, width)
    t0 = time.time()
    faces = engine.detect(small)
    detect_ms = (time.time() - t0) * 1000

    draw_faces(small, faces)
    draw_hud(small, [
        f"faces={len(faces)}  detect={detect_ms:.1f} ms  "
        f"fps={0.0 if fps_ema is None else fps_ema:.1f}{extra_hud}",
        f"GPU={engine.on_gpu}  src={width}px",
    ])

    quit_now = False
    if headless:
        cv2.imwrite(os.path.join(out_dir, "stage1_last.jpg"), small)
    else:
        cv2.imshow("Stage1 - face detection (q to quit)", small)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            quit_now = True
    return detect_ms, faces, quit_now


def run_stream(engine: FaceEngine, source: str, width: int, target_fps: float,
               headless: bool, max_frames: int) -> int:
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"ОШИБКА: не удалось открыть источник: {source}")
        print("Проверь доступность RTSP/файла. Для теста без камеры: --source selftest")
        return 1

    # Файл (есть число кадров) vs live-поток (RTSP/вебка). Логика frame-skip разная.
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    is_file = frame_count and frame_count > 0

    print(f"Источник открыт: {source}")
    print(f"Тип: {'видеофайл' if is_file else 'live-поток (RTSP/вебка)'}, "
          f"исходный fps≈{src_fps:.1f}. Цель ~{target_fps} детекций/сек, ресайз до {width}px. "
          f"Окно: {'нет (headless)' if headless else 'есть, выход — q'}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)

    processed = 0
    fps_ema = None
    t_prev_proc = None

    def update_fps(now):
        nonlocal fps_ema, t_prev_proc
        if t_prev_proc is not None:
            inst = 1.0 / max(1e-6, now - t_prev_proc)
            fps_ema = inst if fps_ema is None else 0.8 * fps_ema + 0.2 * inst
        t_prev_proc = now

    try:
        if is_file:
            # ---- ФАЙЛ: пропуск по счётчику кадров (файл не real-time) ----
            stride = max(1, int(round((src_fps or target_fps) / max(0.1, target_fps))))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Видео закончилось (EOF) — это норма для файла.")
                    break
                idx += 1
                if (idx - 1) % stride != 0:
                    continue                      # пропускаем кадр
                now = time.time(); update_fps(now)
                processed += 1
                detect_ms, faces, quit_now = _process_and_show(
                    engine, frame, width, headless, out_dir, processed, fps_ema)
                if processed % 10 == 0 or processed == 1:
                    print(f"[{processed:4d}] frame#{idx} faces={len(faces)} "
                          f"detect={detect_ms:5.1f}ms fps={0.0 if fps_ema is None else fps_ema:4.1f}")
                if quit_now or (max_frames and processed >= max_frames):
                    break
        else:
            # ---- LIVE: дренируем буфер grab(), decode по таймеру (низкая задержка) ----
            min_interval = 1.0 / max(0.1, target_fps)
            last_proc = 0.0
            fail_reads = 0
            while True:
                if not cap.grab():
                    fail_reads += 1
                    if fail_reads > 50:
                        print("Поток прервался (нет кадров 50+ раз). Этап 3 добавит авто-reconnect.")
                        break
                    time.sleep(0.02)
                    continue
                fail_reads = 0
                now = time.time()
                if now - last_proc < min_interval:
                    continue                      # frame skip по времени
                last_proc = now
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
                update_fps(now)
                processed += 1
                detect_ms, faces, quit_now = _process_and_show(
                    engine, frame, width, headless, out_dir, processed, fps_ema)
                if processed % 10 == 0 or processed == 1:
                    print(f"[{processed:4d}] faces={len(faces)} detect={detect_ms:5.1f}ms "
                          f"fps={0.0 if fps_ema is None else fps_ema:4.1f}")
                if quit_now or (max_frames and processed >= max_frames):
                    break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    print(f"Готово. Обработано кадров: {processed}.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Этап 1: один поток + детекция лиц")
    ap.add_argument("--source", default="selftest",
                    help="rtsp://... | путь к видео | индекс вебки (0) | selftest")
    ap.add_argument("--width", type=int, default=960, help="ресайз по ширине перед детекцией")
    ap.add_argument("--target-fps", type=float, default=3.0, help="сколько кадров/сек детектировать")
    ap.add_argument("--det-size", type=int, default=640, help="размер входа детектора (квадрат)")
    ap.add_argument("--headless", action="store_true",
                    help="без окна (кадры пишутся в data\\stage1_last.jpg)")
    ap.add_argument("--max-frames", type=int, default=0, help="остановиться после N кадров (0 = без лимита)")
    args = ap.parse_args()

    print("Инициализация FaceEngine (buffalo_l, только детекция)...")
    engine = FaceEngine(det_size=(args.det_size, args.det_size),
                        allowed_modules=["detection"])
    FaceEngine.warmup(engine, size=args.det_size)
    print(f"FaceEngine готов. GPU = {engine.on_gpu}")

    if args.source == "selftest":
        rc = run_selftest(engine, args.width)
    else:
        rc = run_stream(engine, args.source, args.width, args.target_fps,
                        args.headless, args.max_frames)
    sys.exit(rc)


if __name__ == "__main__":
    main()
