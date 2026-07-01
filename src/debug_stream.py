# -*- coding: utf-8 -*-
r"""
debug_stream.py — живой просмотр детекций в браузере (для headless-сервера).

Открывает ОДНУ камеру, рисует боксы ЛИЦ (SCRFD) и НОМЕРОВ (fast-alpr) с их
размерами в пикселях и отдаёт MJPEG-поток по HTTP. Нужен, чтобы понять, почему
объектовые камеры не распознают: обычно лицо/номер на кадре слишком мелкие
(далеко/угол/низкое разрешение) — на стриме это сразу видно.

Запуск на сервере:
  python src/debug_stream.py --camera cam05 --port 8091
  python src/debug_stream.py --source "rtsp://user:pass@ip:554/..." --port 8091 --recognize
Затем открыть в браузере:  http://<IP-сервера>:8091

Не мешает main.py: это отдельный инструмент, поднимает свои модели на GPU.
"""
import os
import sys
import time
import argparse
import threading

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, Response

from config import load_settings, load_cameras
from face_engine import FaceEngine
from stage1_single_stream import open_capture, resize_to_width

# --- общее состояние (последний annotated-кадр) ---
_state = {"jpeg": None, "stats": "инициализация…"}
_lock = threading.Lock()


def _draw_faces(frame, engine, gallery, min_det):
    """Боксы лиц + размер в px (+ ID, если есть галерея). Возвращает счётчики."""
    faces = engine.detect(frame)
    n_ok = 0
    for f in faces:
        det = float(getattr(f, "det_score", 0.0))
        x1, y1, x2, y2 = f.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        weak = det < min_det
        color = (0, 165, 255) if weak else (0, 220, 0)   # оранжевый = слишком слабое
        if not weak:
            n_ok += 1
        label = f"face {w}x{h}px {det:.2f}"
        if gallery is not None and not weak:
            try:
                ident, score = gallery.identify(f.normed_embedding)
                if ident is not None and score >= gallery.match_threshold:
                    label = f"{ident.label} {score:.2f} ({w}x{h})"
            except Exception:
                pass
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return len(faces), n_ok


def _draw_plates(frame, anpr, validator, min_conf):
    """Боксы номеров + текст + размер. Возвращает счётчики."""
    plates = anpr.predict(frame)
    n_ok = 0
    for p in plates:
        if not p.bbox:
            continue
        x1, y1, x2, y2 = p.bbox
        w, h = x2 - x1, y2 - y1
        weak = p.ocr_conf < min_conf
        color = (0, 165, 255) if weak else (255, 120, 0)  # синий = принят
        if not weak:
            n_ok += 1
        norm = validator.normalize(p.text)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{norm} {p.ocr_conf:.2f} {w}x{h}px", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return len(plates), n_ok


def _hud(frame, lines):
    y = 18
    for ln in lines:
        cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 230, 255), 1, cv2.LINE_AA)
        y += 22


def capture_loop(cfg, source, width, det_size, target_fps, do_faces, do_plates,
                 face_engine, anpr, validator, gallery, stop=None):
    if stop is None:
        stop = threading.Event()
    min_det = cfg["recognition"]["min_det_score"]
    min_conf = cfg["anpr"]["min_ocr_confidence"]
    fps_ema, t_prev = 0.0, None
    last_proc = 0.0
    min_interval = 1.0 / max(0.1, target_fps)

    while not stop.is_set():
        cap = open_capture(source)
        if not cap.isOpened():
            with _lock:
                _state["stats"] = f"НЕ ОТКРЫТ источник: {source} (повтор через 3с)"
                _blank(source)
            time.sleep(3)
            continue
        is_file = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        fails = 0
        while not stop.is_set():
            if not cap.grab():
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                fails += 1
                if fails > 50:
                    break
                time.sleep(0.02)
                continue
            fails = 0
            now = time.time()
            if now - last_proc < min_interval:
                continue
            last_proc = now
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]
            if width and width > 0:
                small, scale = resize_to_width(frame, width)
            else:
                small, scale = frame, 1.0          # без ресайза — нативное разрешение
            t0 = time.time()
            nf = nf_ok = npl = npl_ok = 0
            if do_faces:
                nf, nf_ok = _draw_faces(small, face_engine, gallery, min_det)
            if do_plates:
                npl, npl_ok = _draw_plates(small, anpr, validator, min_conf)
            infer_ms = (time.time() - t0) * 1000

            if t_prev is not None:
                inst = 1.0 / max(1e-6, now - t_prev)
                fps_ema = inst if fps_ema == 0 else 0.8 * fps_ema + 0.2 * inst
            t_prev = now

            # ВНИМАНИЕ: cv2.putText не рисует кириллицу -> HUD только ASCII
            resized_txt = (f"{small.shape[1]}x{small.shape[0]}"
                           if (width and width > 0) else "NATIVE (no resize)")
            _hud(small, [
                f"src {orig_w}x{orig_h} -> {resized_txt}  det_size={det_size}  fps={fps_ema:.1f}  infer={infer_ms:.0f}ms",
                f"faces={nf} (ok={nf_ok})   plates={npl} (ok={npl_ok})",
                "orange box = too small/weak to recognize",
            ])
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _lock:
                    _state["jpeg"] = buf.tobytes()
                    _state["stats"] = (f"{orig_w}x{orig_h} faces={nf}/{nf_ok} "
                                       f"plates={npl}/{npl_ok} fps={fps_ema:.1f}")
        cap.release()


def _blank(msg):
    import numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "NO SIGNAL", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        _state["jpeg"] = buf.tobytes()


app = FastAPI(title="Debug stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return ("<html><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
            "<div style='padding:8px'>Debug stream — боксы лиц и номеров. "
            "Оранжевый = слишком мелко/слабо.</div>"
            "<img src='/stream' style='width:100%;max-width:1280px;display:block;margin:auto'>"
            "</body></html>")


@app.get("/stream")
def stream():
    def gen():
        boundary = b"--frame"
        while True:
            with _lock:
                jpg = _state["jpeg"]
            if jpg is not None:
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot")
def snapshot():
    with _lock:
        jpg = _state["jpeg"]
    if jpg is None:
        return Response(status_code=503)
    return Response(content=jpg, media_type="image/jpeg")


def resolve_source(args, cfg):
    if args.source:
        return args.source
    for c in load_cameras():
        if c["id"] == args.camera:
            return c["rtsp"]
    print(f"Камера {args.camera} не найдена в cameras.yaml")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Живой просмотр детекций (MJPEG)")
    ap.add_argument("--source", default="", help="RTSP/файл")
    ap.add_argument("--camera", default="", help="id камеры из cameras.yaml")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--width", type=int, default=0,
                    help="ресайз по ширине; 0 = БЕЗ ресайза (нативное разрешение) — для дебага")
    ap.add_argument("--det-size", type=int, default=1600,
                    help="вход детектора лиц (SCRFD); больше = ловит мелкие лица, но медленнее")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-plates", action="store_true")
    ap.add_argument("--recognize", action="store_true", help="подписывать лица ID из галереи")
    args = ap.parse_args()
    if not args.source and not args.camera:
        print("Укажи --camera <id> или --source <rtsp>")
        return 1

    cfg = load_settings()
    source = resolve_source(args, cfg)
    width = args.width            # 0 = без ресайза (нативное разрешение)
    det_size = args.det_size
    do_faces = not args.no_faces
    do_plates = not args.no_plates

    face_engine = anpr = validator = gallery = None
    if do_faces:
        face_engine = FaceEngine(
            model_name=cfg["recognition"]["model_name"],
            det_size=(det_size, det_size),
            ctx_id=cfg["gpu"]["ctx_id"],
            allowed_modules=["detection", "recognition"] if args.recognize else ["detection"],
        )
        if args.recognize:
            from gallery import Gallery
            gallery = Gallery(cfg)
    if do_plates:
        from anpr.engine import AnprEngine
        from anpr.plate_format import PlateValidator
        anpr = AnprEngine(cfg)
        validator = PlateValidator(cfg["anpr"]["plate_regex"])

    print(f"Источник: {source}\nФейсы={do_faces} Номера={do_plates} "
          f"width={'NATIVE' if not width else width} det_size={det_size} "
          f"GPU_faces={getattr(face_engine,'on_gpu',None)} GPU_anpr={getattr(anpr,'on_gpu',None)}")
    print(f"Открой в браузере: http://<IP-сервера>:{args.port}")

    stop = threading.Event()
    # именованные аргументы — устойчиво к порядку/числу параметров
    t = threading.Thread(target=capture_loop, kwargs=dict(
        cfg=cfg, source=source, width=width, det_size=det_size,
        target_fps=cfg["recognition"]["target_fps"], do_faces=do_faces, do_plates=do_plates,
        face_engine=face_engine, anpr=anpr, validator=validator, gallery=gallery, stop=stop),
        daemon=True)
    t.start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
