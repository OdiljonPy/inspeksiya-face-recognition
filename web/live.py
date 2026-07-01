# -*- coding: utf-8 -*-
"""
live.py — живой просмотр камер с боксами в дашборде (по клику / мозаикой).

- Ленивая инициализация движков (грузятся при первом запросе live).
- НЕСКОЛЬКО камер одновременно (для мозаики), каждая — свой поток захвата+детекции.
- Пул движков лиц по det_size (выбор качества из UI).
- Авто-стоп: поток, который никто не читает > IDLE_TIMEOUT сек, останавливается
  (освобождаем камеру и GPU). Лимит одновременных камер MAX_LIVE.
"""
import time
import threading

import cv2

from config import load_settings, load_cameras
import draw_overlay

IDLE_TIMEOUT = 12.0      # сек без чтения -> остановить захват
MAX_LIVE = 6             # максимум одновременных live-камер (защита GPU)
DEFAULT_DET = 1280
ALLOWED_DET = (640, 960, 1280, 1600)


class LiveManager:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_settings()
        self.lock = threading.Lock()
        self.cam_map = {c["id"]: c for c in load_cameras()}

        self._ready = False
        self._eng_lock = threading.Lock()
        self.face_engines = {}          # det_size -> FaceEngine (ленивый пул)
        self.anpr = self.validator = self.gallery = None

        self.workers = {}               # cam_id -> dict(cam, det, stop, thread, jpeg, last, status)

        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    # ---------- ленивые движки ----------
    def _ensure_common(self):
        if self.anpr is not None:
            return
        with self._eng_lock:
            if self.anpr is not None:
                return
            from gpu_setup import enable_onnx_cuda
            enable_onnx_cuda()
            from anpr.engine import AnprEngine
            from anpr.plate_format import PlateValidator
            from gallery import Gallery
            self.anpr = AnprEngine(self.cfg)
            self.validator = PlateValidator(self.cfg["anpr"]["plate_regex"])
            self.gallery = Gallery(self.cfg)

    def _face_engine(self, det: int):
        with self._eng_lock:
            if det not in self.face_engines:
                from face_engine import FaceEngine
                e = FaceEngine(
                    model_name=self.cfg["recognition"]["model_name"],
                    det_size=(det, det),
                    ctx_id=self.cfg["gpu"]["ctx_id"],
                    allowed_modules=["detection", "recognition"],
                )
                FaceEngine.warmup(e, size=det)
                self.face_engines[det] = e
            return self.face_engines[det]

    def cameras(self):
        return [{"id": c["id"], "zone": c.get("zone", ""), "mode": c.get("mode", "face")}
                for c in self.cam_map.values()]

    # ---------- управление ----------
    def start(self, cam_id: str, det: int = 0) -> bool:
        if cam_id not in self.cam_map:
            return False
        cam = self.cam_map[cam_id]
        det = int(det) if det in ALLOWED_DET else int(cam.get("det_size", DEFAULT_DET))
        if det not in ALLOWED_DET:
            det = DEFAULT_DET
        self._ensure_common()
        engine = self._face_engine(det)

        with self.lock:
            w = self.workers.get(cam_id)
            if w and w["thread"].is_alive() and w["det"] == det:
                w["last"] = time.time()
                return True
            if w:                                   # перезапуск (сменился det)
                w["stop"].set()
                self.workers.pop(cam_id, None)
            # лимит одновременных: вытесняем самую старую по доступу
            if len(self.workers) >= MAX_LIVE:
                oldest = min(self.workers, key=lambda k: self.workers[k]["last"])
                self.workers[oldest]["stop"].set()
                self.workers.pop(oldest, None)
            stop = threading.Event()
            wd = {"cam": cam, "det": det, "stop": stop, "jpeg": None,
                  "last": time.time(), "status": ""}
            t = threading.Thread(target=self._loop, args=(wd, engine), daemon=True)
            wd["thread"] = t
            self.workers[cam_id] = wd
            t.start()
            return True

    def get_jpeg(self, cam_id: str):
        w = self.workers.get(cam_id)
        if not w:
            return None
        w["last"] = time.time()
        return w["jpeg"]

    def stop(self, cam_id: str = None):
        with self.lock:
            if cam_id:
                w = self.workers.pop(cam_id, None)
                if w:
                    w["stop"].set()
            else:
                for w in self.workers.values():
                    w["stop"].set()
                self.workers.clear()

    def _watchdog(self):
        while True:
            time.sleep(3)
            with self.lock:
                for cid in [c for c, w in self.workers.items()
                            if time.time() - w["last"] > IDLE_TIMEOUT
                            or not w["thread"].is_alive()]:
                    self.workers[cid]["stop"].set()
                    self.workers.pop(cid, None)

    # ---------- захват + детекция одной камеры ----------
    def _loop(self, w, engine):
        from stage1_single_stream import open_capture, resize_to_width
        cam = w["cam"]
        stop = w["stop"]
        rtsp = cam["rtsp"]
        width = int(cam.get("width", 0))
        target_fps = float(cam.get("fps", self.cfg["recognition"]["target_fps"]))
        min_det = self.cfg["recognition"]["min_det_score"]
        min_conf = self.cfg["anpr"]["min_ocr_confidence"]
        mode = cam.get("mode", "face")
        do_face = mode in ("face", "both")
        do_plate = mode in ("plate", "both")
        min_interval = 1.0 / max(0.1, target_fps)
        fps_ema, t_prev, last_proc = 0.0, None, 0.0

        while not stop.is_set():
            cap = open_capture(rtsp)
            if not cap.isOpened():
                self._blank(w, f"NO SIGNAL: {cam['id']}")
                if stop.wait(3):
                    break
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
                oh, ow = frame.shape[:2]
                if width and width > 0 and ow > width:
                    frame, _ = resize_to_width(frame, width)
                t0 = time.time()
                nf = nf_ok = npl = npl_ok = 0
                try:
                    if do_face:
                        nf, nf_ok = draw_overlay.draw_faces(frame, engine, self.gallery, min_det)
                    if do_plate:
                        npl, npl_ok = draw_overlay.draw_plates(frame, self.anpr, self.validator, min_conf)
                except Exception as e:
                    draw_overlay.hud(frame, [f"INFER ERROR: {str(e)[:60]}"])
                infer_ms = (time.time() - t0) * 1000
                if t_prev is not None:
                    inst = 1.0 / max(1e-6, now - t_prev)
                    fps_ema = inst if fps_ema == 0 else 0.8 * fps_ema + 0.2 * inst
                t_prev = now
                draw_overlay.hud(frame, [
                    f"{cam['id']} [{mode}]  src {ow}x{oh}  det={w['det']}  fps={fps_ema:.1f}  infer={infer_ms:.0f}ms",
                    f"faces={nf} (ok={nf_ok})   plates={npl} (ok={npl_ok})",
                ])
                enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if enc:
                    w["jpeg"] = buf.tobytes()
            cap.release()

    def _blank(self, w, msg):
        import numpy as np
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, msg, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        enc, buf = cv2.imencode(".jpg", img)
        if enc:
            w["jpeg"] = buf.tobytes()
