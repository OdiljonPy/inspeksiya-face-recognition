# -*- coding: utf-8 -*-
"""
person_engine.py — детектор ЧЕЛОВЕКА (YOLOv8n, класс person из COCO) на onnxruntime.

Этап 1 архитектуры «person-first»: находим людей в кадре. Дальше (этапы 2-3)
лицо будет искаться только внутри бокса человека, а трек вестись по телу.

Стек тот же, что у лиц/ANPR: onnxruntime-gpu, БЕЗ torch в рантайме.
Модель: data/models/yolov8n.onnx (экспорт с HF-зеркала Ultralytics/YOLOv8;
GitHub-ассеты в регионе блокируются — не менять источник на github).

Вход 640x640 (letterbox), выход (1, 84, 8400): 4 бокса cxcywh + 80 классов.
Берём только класс 0 (person) + NMS.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()  # до импорта onnxruntime

import cv2
import numpy as np
import onnxruntime as ort


class PersonDetection:
    __slots__ = ("bbox", "conf")

    def __init__(self, bbox, conf):
        self.bbox = bbox      # (x1, y1, x2, y2) в координатах ИСХОДНОГО кадра
        self.conf = conf


class PersonEngine:
    def __init__(self, model_path: str, conf: float = 0.5, iou: float = 0.45,
                 input_size: int = 640, prefer_gpu: bool = True):
        if not os.path.isabs(model_path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.normpath(os.path.join(root, model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель человека не найдена: {model_path}. "
                "См. deploy/preload_models.py (HF-зеркало Ultralytics/YOLOv8).")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if prefer_gpu else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.on_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        self.conf = float(conf)
        self.iou = float(iou)
        self.size = int(input_size)

    def _letterbox(self, img):
        """Вписать кадр в квадрат size x size с сохранением пропорций (паддинг серым)."""
        h, w = img.shape[:2]
        r = min(self.size / w, self.size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, r, dx, dy

    def detect(self, frame_bgr) -> list[PersonDetection]:
        """Найти людей на кадре. Боксы — в координатах исходного кадра."""
        h, w = frame_bgr.shape[:2]
        canvas, r, dx, dy = self._letterbox(frame_bgr)
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0     # BGR->RGB, 0..1
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])  # (1,3,H,W)
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (84, 8400)

        boxes_c = out[:4].T          # (8400, 4) cx,cy,w,h в координатах letterbox
        person_conf = out[4]         # класс 0 = person
        keep = person_conf >= self.conf
        if not keep.any():
            return []
        boxes_c, confs = boxes_c[keep], person_conf[keep]

        # cxcywh -> xywh (для NMS) в координатах ИСХОДНОГО кадра
        x = (boxes_c[:, 0] - boxes_c[:, 2] / 2 - dx) / r
        y = (boxes_c[:, 1] - boxes_c[:, 3] / 2 - dy) / r
        bw = boxes_c[:, 2] / r
        bh = boxes_c[:, 3] / r
        rects = np.stack([x, y, bw, bh], axis=1)

        idx = cv2.dnn.NMSBoxes(rects.tolist(), confs.tolist(), self.conf, self.iou)
        result = []
        for i in np.array(idx).flatten():
            x1 = max(0, int(rects[i, 0])); y1 = max(0, int(rects[i, 1]))
            x2 = min(w, int(rects[i, 0] + rects[i, 2])); y2 = min(h, int(rects[i, 1] + rects[i, 3]))
            if x2 > x1 and y2 > y1:
                result.append(PersonDetection((x1, y1, x2, y2), float(confs[i])))
        return result

    def warmup(self, rounds: int = 2):
        dummy = (np.random.rand(480, 848, 3) * 255).astype(np.uint8)
        for _ in range(rounds):
            self.detect(dummy)
