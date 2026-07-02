# -*- coding: utf-8 -*-
"""
face_quality.py — фильтр КАЧЕСТВА лица ДО распознавания (не порог cosine!).

Отсекает лица, которые дают ложные совпадения: профиль / размытые / мелкие.
Только лицо, прошедшее ВСЕ проверки, идёт в FAISS-поиск (identify).

4 проверки (пороги — из config.face_quality, с дефолтами):
  1) det_score  — уверенность детекции SCRFD (face.det_score), деф. >= 0.65
  2) размер     — ширина bbox в px НА ИСХОДНОМ кадре (до ресайза), деф. >= 80
  3) фронтальность — по 5 kps: asym = max(d_нос-левглаз, d_нос-правглаз)/min(...);
                     чем больше — тем сильнее профиль. Деф. отбраковка при asym > 1.6
  4) резкость   — variance of Laplacian на кропе лица, деф. >= 100
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Quality:
    det_score: float
    width_px: float          # ширина лица на ИСХОДНОМ кадре, px
    blur: float              # variance of Laplacian (чем больше — чётче)
    yaw_asym: float          # асимметрия нос-глаза (1.0 = анфас, >1 = профиль)
    passed: bool
    reason: str              # какая проверка не прошла ("" если прошла)


def variance_of_laplacian(gray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def yaw_asymmetry(kps) -> float:
    """
    kps insightface: [левый глаз, правый глаз, нос, левый угол рта, правый угол рта].
    Возвращает max(d_нос-левглаз, d_нос-правглаз)/min(...). Анфас ~1.0, профиль >>1.
    """
    try:
        kps = np.asarray(kps, dtype=np.float32)
        le, re, nose = kps[0], kps[1], kps[2]
        d_left = float(np.linalg.norm(nose - le))
        d_right = float(np.linalg.norm(nose - re))
        lo, hi = min(d_left, d_right), max(d_left, d_right)
        if lo < 1e-3:
            return 999.0
        return hi / lo
    except Exception:
        return 1.0


class FaceQuality:
    """Оценка качества лица по порогам из конфига."""

    def __init__(self, cfg: dict):
        fq = cfg.get("face_quality", {}) or {}
        self.enabled = bool(fq.get("enabled", True))
        self.mode = str(fq.get("mode", "event"))          # event | ignore
        self.min_det = float(fq.get("min_det_score", 0.65))
        self.min_px = float(fq.get("min_width_px", 80))
        self.min_blur = float(fq.get("min_blur", 100.0))
        self.max_asym = float(fq.get("max_yaw_asym", 1.6))

    def assess(self, face, frame, scale: float = 1.0) -> Quality:
        """
        face  — insightface Face (bbox, det_score, kps).
        frame — кадр, НА КОТОРОМ детектили (возможно ресайзнутый).
        scale — коэффициент ресайза (frame_w/original_w); px пересчитываем в исходные.
        """
        det = float(getattr(face, "det_score", 0.0))
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        width_px = (x2 - x1) / max(scale, 1e-6)          # ширина на ИСХОДНОМ кадре

        # резкость на кропе лица (в координатах текущего frame)
        h, w = frame.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        blur = 0.0
        if cx2 > cx1 and cy2 > cy1:
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = variance_of_laplacian(gray)

        asym = yaw_asymmetry(getattr(face, "kps", None))

        # проверки по порядку — фиксируем первую непройденную
        reason = ""
        if det < self.min_det:
            reason = f"det<{self.min_det}"
        elif width_px < self.min_px:
            reason = f"px<{self.min_px:.0f}"
        elif asym > self.max_asym:
            reason = f"profile(asym>{self.max_asym})"
        elif blur < self.min_blur:
            reason = f"blur<{self.min_blur:.0f}"
        passed = reason == ""

        return Quality(det_score=round(det, 3), width_px=round(width_px, 1),
                       blur=round(blur, 1), yaw_asym=round(asym, 3),
                       passed=passed, reason=reason)
