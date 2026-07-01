# -*- coding: utf-8 -*-
"""
faiss_index.py — построение и поиск по базе эмбеддингов.

Используем cosine-сходство. Приём: L2-нормализуем векторы, тогда
inner product (IndexFlatIP) == cosine similarity. Порог берём из конфига.

Один вектор = один человек (усреднённый по его фото эмбеддинг).
labels.json хранит имена, выровненные по id векторов в индексе.
"""
import os
import json
import numpy as np
import faiss


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-нормализация по строкам (поддерживает (D,) и (N,D))."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v) + 1e-9
        return v / n
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n


def build_index(vectors: np.ndarray, names: list[str],
                index_path: str, labels_path: str) -> None:
    """
    Построить и сохранить индекс.
    vectors : (N, D) — ДОЛЖНЫ быть уже L2-нормализованы.
    names   : список из N имён, выровнен по строкам vectors.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D (N, D)")
    if vectors.shape[0] != len(names):
        raise ValueError("vectors и names должны совпадать по длине")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine на норм. векторах
    index.add(vectors)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)


def load_index(index_path: str, labels_path: str):
    """Загрузить (index, names). Бросает FileNotFoundError, если базы нет."""
    if not (os.path.exists(index_path) and os.path.exists(labels_path)):
        raise FileNotFoundError(
            f"Индекс не найден ({index_path}). Сначала запусти enroll.py.")
    index = faiss.read_index(index_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    return index, names


def search(index, names: list[str], query: np.ndarray, top_k: int = 1):
    """
    Поиск ближайших. query — (D,) или (1,D), ДОЛЖЕН быть L2-нормализован.
    Возвращает список [(name, score), ...] длиной до top_k.
    """
    q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
    scores, idx = index.search(q, top_k)
    out = []
    for s, i in zip(scores[0], idx[0]):
        if i < 0:
            continue
        out.append((names[i], float(s)))
    return out
