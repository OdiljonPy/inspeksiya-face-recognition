# -*- coding: utf-8 -*-
"""
engine.py — обёртка над fast_alpr.ALPR (детектор номера + OCR).

Переиспользует то же GPU-решение sm_120, что и InsightFace: перед импортом
onnxruntime вызываем gpu_setup.enable_onnx_cuda(), затем отдаём fast-alpr
CUDA-провайдеры. Если CUDA недоступна — OCR/детекция уходят на CPU (по ТЗ допустимо).
"""
import os
import sys

# gpu_setup лежит в src/ (родитель пакета anpr). Гарантируем, что он в путях.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()  # до импорта onnxruntime/fast_alpr

import numpy as np
from fast_alpr import ALPR


class PlateDetection:
    """Унифицированный результат одного номера."""
    __slots__ = ("text", "ocr_conf", "det_conf", "bbox")

    def __init__(self, text, ocr_conf, det_conf, bbox):
        self.text = text              # сырой текст OCR
        self.ocr_conf = ocr_conf      # средняя уверенность OCR [0..1]
        self.det_conf = det_conf      # уверенность детектора номера [0..1]
        self.bbox = bbox              # (x1, y1, x2, y2)


class AnprEngine:
    def __init__(self, cfg: dict, prefer_gpu: bool = True):
        a = cfg["anpr"]
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if prefer_gpu else ["CPUExecutionProvider"])

        self.alpr = ALPR(
            detector_model=a["detector_model"],
            detector_conf_thresh=float(a["detector_conf"]),
            detector_providers=providers,
            ocr_model=a["ocr_model"],
            ocr_providers=providers,
        )
        self.on_gpu = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        """Понять, реально ли хоть одна сессия onnxruntime поехала на CUDA."""
        provs = []
        for obj in (getattr(self.alpr, "detector", None), getattr(self.alpr, "ocr", None)):
            for sess in _collect_sessions(obj):
                try:
                    provs += list(sess.get_providers())
                except Exception:
                    pass
        return "CUDAExecutionProvider" in provs

    def predict(self, image_bgr: np.ndarray) -> list[PlateDetection]:
        """Прогнать кадр через ALPR. Вернуть список PlateDetection."""
        raw = self.alpr.predict(image_bgr)
        out = []
        for r in raw:
            det = getattr(r, "detection", None)
            ocr = getattr(r, "ocr", None)
            bbox = None
            det_conf = 0.0
            if det is not None:
                bb = getattr(det, "bounding_box", None)
                if bb is not None:
                    bbox = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))
                det_conf = float(getattr(det, "confidence", 0.0))
            text = getattr(ocr, "text", "") if ocr is not None else ""
            ocr_conf = _mean_conf(getattr(ocr, "confidence", None)) if ocr is not None else 0.0
            out.append(PlateDetection(text, ocr_conf, det_conf, bbox))
        return out


def _mean_conf(conf) -> float:
    """OCR-уверенность может быть числом или списком (по символам) — усредняем."""
    if conf is None:
        return 0.0
    if isinstance(conf, (list, tuple)):
        vals = [float(c) for c in conf if c is not None]
        return sum(vals) / len(vals) if vals else 0.0
    try:
        return float(conf)
    except (TypeError, ValueError):
        return 0.0


def _collect_sessions(obj, depth: int = 0, seen: set | None = None) -> list:
    """
    Рекурсивно найти все onnxruntime.InferenceSession внутри объекта
    (в fast-alpr они лежат как detector.detector.model и ocr.ocr_model.model).
    """
    import onnxruntime as ort
    if seen is None:
        seen = set()
    out = []
    if obj is None or id(obj) in seen or depth > 4:
        return out
    seen.add(id(obj))
    if isinstance(obj, ort.InferenceSession):
        return [obj]
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            v = getattr(obj, name)
        except Exception:
            continue
        if isinstance(v, ort.InferenceSession):
            out.append(v)
        elif hasattr(v, "__dict__") and not callable(v):
            out += _collect_sessions(v, depth + 1, seen)
    return out
