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
from stage1_single_stream import resize_to_width


class InferenceWorker(threading.Thread):
    def __init__(self, frame_queue, face_engines, gallery, settings, stop_event, on_result,
                 cam_modes=None, cam_det=None, cam_width=None, cam_roi=None,
                 anpr_engine=None, anpr_validator=None,
                 vehicle_log=None, plates_dir=None, on_plate=None,
                 veh_full_dir=None, region_ocr=None):
        super().__init__(daemon=True, name="inference")
        self.q = frame_queue
        # пул движков лиц по det_size: {640: FaceEngine, 1280: FaceEngine, ...}
        self.face_engines = face_engines or {}
        self.gallery = gallery
        self.cfg = settings
        self.min_det = settings["recognition"]["min_det_score"]
        self.stop_event = stop_event
        self.on_result = on_result
        self.cam_modes = cam_modes or {}
        # per-camera параметры (с дефолтами из settings)
        self.cam_det = cam_det or {}          # cam_id -> det_size
        self.cam_width = cam_width or {}       # cam_id -> width (0 = без ресайза)
        self.cam_roi = cam_roi or {}           # cam_id -> [x1,y1,x2,y2] (зона обработки)
        self.default_det = settings["recognition"]["det_size"]

        # ANPR-компоненты (могут быть None, если plate-камер нет)
        self.anpr_engine = anpr_engine
        self.anpr_validator = anpr_validator
        self.vehicle_log = vehicle_log
        self.plates_dir = plates_dir
        self.anpr_min_conf = settings["anpr"]["min_ocr_confidence"]
        self.anpr_min_px = int(settings["anpr"].get("min_plate_px", 0))
        self.veh_full_dir = veh_full_dir       # куда писать полный кадр события транспорта
        self.region_ocr = region_ocr           # второй OCR-проход региона (или None)
        self.on_plate = on_plate

        self.trackers: dict[str, CameraTracker] = {}
        # отдельная статистика
        self.processed = 0          # кадров с лицами
        self.infer_ms_ema = 0.0
        self.anpr_processed = 0     # кадров с ANPR
        self.anpr_infer_ms_ema = 0.0

    def _mode(self, cam_id: str) -> str:
        return self.cam_modes.get(cam_id, "face")

    def _face_engine(self, cam_id: str):
        ds = self.cam_det.get(cam_id, self.default_det)
        return self.face_engines.get(ds) or next(iter(self.face_engines.values()), None)

    def _tracker(self, cam_id: str) -> CameraTracker:
        t = self.trackers.get(cam_id)
        if t is None:
            t = CameraTracker(self.gallery, self.cfg)
            self.trackers[cam_id] = t
        return t

    def _apply_roi(self, cam_id: str, frame):
        """Кроп зоны обработки (roi из cameras.yaml). Пиксели НЕ масштабируются,
        поэтому все замеры качества/px остаются в исходном масштабе."""
        roi = self.cam_roi.get(cam_id)
        if not roi:
            return frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(roi[0])), max(0, int(roi[1]))
        x2, y2 = min(w, int(roi[2])), min(h, int(roi[3]))
        if x2 - x1 < 32 or y2 - y1 < 32:      # битый roi — не роняем обработку
            return frame
        return frame[y1:y2, x1:x2]

    def run(self):
        while not self.stop_event.is_set():
            try:
                item: FrameItem = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            mode = self._mode(item.cam_id)
            do_face = mode in ("face", "both") and self.face_engines
            do_plate = mode in ("plate", "both") and self.anpr_engine is not None

            # подхватить внешние изменения галереи (напр. удаление из дашборда)
            if do_face and self.gallery is not None:
                self.gallery.maybe_reload()

            # ROI-кроп (зона прохода/ворот) — общий для лиц и ANPR
            roi_frame = self._apply_roi(item.cam_id, item.frame)

            # -------- ЛИЦА --------
            if do_face:
                t0 = time.time()
                engine = self._face_engine(item.cam_id)
                # per-camera ресайз (0 = без ресайза, нативное); det_size задаёт движок
                w = self.cam_width.get(item.cam_id, 0)
                frame = roi_frame
                scale = 1.0                       # frame_w/original_w (для размера лица в исходных px)
                if w and w > 0 and frame.shape[1] > w:
                    frame, scale = resize_to_width(frame, w)
                faces = engine.detect(frame)
                faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= self.min_det]
                results = self._tracker(item.cam_id).update(faces, frame, item.capture_ts, scale)
                now = time.time()
                infer_ms = (now - t0) * 1000
                self.infer_ms_ema = infer_ms if self.infer_ms_ema == 0 else \
                    0.9 * self.infer_ms_ema + 0.1 * infer_ms
                self.processed += 1
                self.on_result(FrameResult(
                    cam_id=item.cam_id, zone=item.zone, faces=results,
                    infer_ms=infer_ms, latency_ms=(now - item.capture_ts) * 1000,
                    frame=frame, object_id=item.object_id,
                ))

            # -------- НОМЕРА (ANPR) --------
            if do_plate:
                t0 = time.time()
                plate_res = process_frame(
                    self.anpr_engine, self.anpr_validator, self.vehicle_log,
                    roi_frame, item.cam_id, item.zone, self.plates_dir,
                    self.anpr_min_conf, ts=item.capture_ts, object_id=item.object_id,
                    full_dir=self.veh_full_dir, region_ocr=self.region_ocr,
                    min_plate_px=self.anpr_min_px,
                )
                now = time.time()
                infer_ms = (now - t0) * 1000
                self.anpr_infer_ms_ema = infer_ms if self.anpr_infer_ms_ema == 0 else \
                    0.9 * self.anpr_infer_ms_ema + 0.1 * infer_ms
                self.anpr_processed += 1
                if self.on_plate and plate_res:
                    self.on_plate(item.cam_id, item.zone, plate_res,
                                  infer_ms, (now - item.capture_ts) * 1000)
