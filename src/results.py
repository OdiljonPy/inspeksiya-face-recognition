# -*- coding: utf-8 -*-
"""results.py — общие датаклассы результата (чтобы не было циклических импортов)."""
from dataclasses import dataclass


@dataclass
class FaceResult:
    bbox: tuple
    label: str        # ID человека, напр. "person_0001"
    score: float      # cosine со своим эмбеддингом в галерее
    is_new: bool      # True, если этот ID создан только что
    crop_path: str    # путь к снимку лица (один на ID)


@dataclass
class FrameResult:
    cam_id: str
    zone: str
    faces: list           # list[FaceResult]
    infer_ms: float
    latency_ms: float
    frame: object
