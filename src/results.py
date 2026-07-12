# -*- coding: utf-8 -*-
"""results.py — общие датаклассы результата (чтобы не было циклических импортов)."""
from dataclasses import dataclass


@dataclass
class FaceResult:
    bbox: tuple
    label: str        # ID человека ("person_0001") или "LOW_QUALITY"
    score: float      # cosine со своим эмбеддингом в галерее
    is_new: bool      # True, если этот ID создан только что
    crop_path: str    # путь к снимку лица (один на ID; для LOW_QUALITY пустой — снимок пишется при логировании)
    # метрики качества (для подбора порогов и логирования)
    q_det: float = None
    q_px: float = None
    q_blur: float = None
    q_yaw: float = None
    # «серая зона» матчинга: ID присвоен по ближайшему, но уверенности мало
    uncertain: bool = False


@dataclass
class FrameResult:
    cam_id: str
    zone: str
    faces: list           # list[FaceResult]
    infer_ms: float
    latency_ms: float
    frame: object
    object_id: str = "default"
