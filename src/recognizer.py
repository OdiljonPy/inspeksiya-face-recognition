# -*- coding: utf-8 -*-
"""
recognizer.py — сопоставление эмбеддинга лица с базой известных людей.

Обёртка над загруженным FAISS-индексом + порогом cosine из конфига.
"""
from dataclasses import dataclass

import numpy as np

import faiss_index as fdb


@dataclass
class Match:
    name: str          # итоговое имя ("Unknown", если ниже порога)
    score: float       # cosine-сходство с лучшим кандидатом
    matched: bool      # прошёл ли порог
    best_name: str     # имя лучшего кандидата (даже если не прошёл порог)


class Recognizer:
    """Идентификация лица по нормализованному эмбеддингу (512-d ArcFace)."""

    def __init__(self, index, names: list[str], threshold: float = 0.5):
        self.index = index
        self.names = names
        self.threshold = float(threshold)

    @classmethod
    def from_files(cls, index_path: str, labels_path: str, threshold: float = 0.5):
        index, names = fdb.load_index(index_path, labels_path)
        return cls(index, names, threshold)

    def identify(self, normed_embedding: np.ndarray) -> Match:
        """
        normed_embedding — L2-нормализованный вектор (insightface .normed_embedding).
        Возвращает Match. Если база пуста — Unknown со score 0.
        """
        res = fdb.search(self.index, self.names, normed_embedding, top_k=1)
        if not res:
            return Match(name="Unknown", score=0.0, matched=False, best_name="")
        best_name, score = res[0]
        matched = score >= self.threshold
        return Match(
            name=best_name if matched else "Unknown",
            score=score,
            matched=matched,
            best_name=best_name,
        )
