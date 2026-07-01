# -*- coding: utf-8 -*-
"""
inference_worker.py — Единый поток инференса на общий GPU-движок.

Маршрутизация по РЕЖИМУ камеры (face/plate/both):
  - face: детекция лиц + трекинг + авто-галерея (стабилизация ID);
  - plate: ANPR (детекция номера + OCR + лог в vehicle_events с дедупом);
  - both: и то, и другое.
Камеры с режимом без лиц не гоняют распознавание лиц, и наоборот — экономим GPU.

FPS/задержка для лиц и для ANPR считаются ОТДЕЛЬНО (по ТЗ ANPR-модуля).
"""
import time
import queue
import threading

from camera_worker import FrameItem
from tracker import CameraTracker
from results import FaceResult, FrameResult   # noqa: F401 (реэкспорт)
from anpr.pipeline import process_frame


class InferenceWorker(threading.Thread):
    def __init__(self, frame_queue, engine, gallery, settings, stop_event, on_result,
                 cam_modes=None, anpr_engine=None, anpr_validator=None,
                 vehicle_log=None, plates_dir=None, on_plate=None):
        super().__init__(daemon=True, name="inference")
        self.q = frame_queue
        self.engine = engine                 # FaceEngine (или None, если лиц нет нигде)
        self.gallery = gallery
        self.cfg = settings
        self.min_det = settings["recognition"]["min_det_score"]
        self.stop_event = stop_event
        self.on_result = on_result
        self.cam_modes = cam_modes or {}

        # ANPR-компоненты (могут быть None, если plate-камер нет)
        self.anpr_engine = anpr_engine
        self.anpr_validator = anpr_validator
        self.vehicle_log = vehicle_log
        self.plates_dir = plates_dir
        self.anpr_min_conf = settings["anpr"]["min_ocr_confidence"]
        self.on_plate = on_plate

        self.trackers: dict[str, CameraTracker] = {}
        # отдельная статистика
        self.processed = 0          # кадров с лицами
        self.infer_ms_ema = 0.0
        self.anpr_processed = 0     # кадров с ANPR
        self.anpr_infer_ms_ema = 0.0

    def _mode(self, cam_id: str) -> str:
        return self.cam_modes.get(cam_id, "face")

    def _tracker(self, cam_id: str) -> CameraTracker:
        t = self.trackers.get(cam_id)
        if t is None:
            t = CameraTracker(self.gallery, self.cfg)
            self.trackers[cam_id] = t
        return t

    def run(self):
        while not self.stop_event.is_set():
            try:
                item: FrameItem = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            mode = self._mode(item.cam_id)
            do_face = mode in ("face", "both") and self.engine is not None
            do_plate = mode in ("plate", "both") and self.anpr_engine is not None

            # -------- ЛИЦА --------
            if do_face:
                t0 = time.time()
                faces = self.engine.detect(item.frame)
                faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= self.min_det]
                results = self._tracker(item.cam_id).update(faces, item.frame, item.capture_ts)
                now = time.time()
                infer_ms = (now - t0) * 1000
                self.infer_ms_ema = infer_ms if self.infer_ms_ema == 0 else \
                    0.9 * self.infer_ms_ema + 0.1 * infer_ms
                self.processed += 1
                self.on_result(FrameResult(
                    cam_id=item.cam_id, zone=item.zone, faces=results,
                    infer_ms=infer_ms, latency_ms=(now - item.capture_ts) * 1000,
                    frame=item.frame,
                ))

            # -------- НОМЕРА (ANPR) --------
            if do_plate:
                t0 = time.time()
                plate_res = process_frame(
                    self.anpr_engine, self.anpr_validator, self.vehicle_log,
                    item.frame, item.cam_id, item.zone, self.plates_dir,
                    self.anpr_min_conf, ts=item.capture_ts,
                )
                now = time.time()
                infer_ms = (now - t0) * 1000
                self.anpr_infer_ms_ema = infer_ms if self.anpr_infer_ms_ema == 0 else \
                    0.9 * self.anpr_infer_ms_ema + 0.1 * infer_ms
                self.anpr_processed += 1
                if self.on_plate and plate_res:
                    self.on_plate(item.cam_id, item.zone, plate_res,
                                  infer_ms, (now - item.capture_ts) * 1000)
