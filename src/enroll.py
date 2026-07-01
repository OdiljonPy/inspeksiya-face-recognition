# -*- coding: utf-8 -*-
r"""
enroll.py — Этап 2. Регистрация известных людей.

Кладёшь фото в data\known_faces\<Имя>\*.jpg (несколько ракурсов на человека).
Скрипт:
  1) детектирует лицо на каждом фото (берёт самое крупное, если их несколько);
  2) считает нормализованный эмбеддинг (512-d ArcFace);
  3) усредняет эмбеддинги по человеку и снова нормализует -> один вектор на персону;
  4) строит FAISS-индекс (cosine) и сохраняет faiss.index + labels.json.

Запуск:
  python src\enroll.py
"""
import os
import sys
import glob

import cv2
import numpy as np

from config import load_settings, index_paths
from face_engine import FaceEngine
import faiss_index as fdb

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def detect_robust(engine, img):
    """
    Детекция с фолбэком: если на тесном кропе (лицо во весь кадр) ничего не нашлось,
    добавляем чёрные поля и пробуем ещё раз. Возвращает список лиц.
    """
    faces = engine.detect(img)
    if faces:
        return faces
    h, w = img.shape[:2]
    pad = int(0.4 * max(h, w))
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return engine.detect(padded)


def pick_largest_face(faces):
    """Из списка лиц выбрать самое крупное по площади bbox."""
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)
    return max(faces, key=area)


def list_person_images(person_dir: str) -> list[str]:
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(person_dir, ext))
    return sorted(files)


def enroll(cfg: dict) -> int:
    known_dir = cfg["paths"]["known_faces"]
    if not os.path.isdir(known_dir):
        print(f"ОШИБКА: нет папки {known_dir}. Создай data\\known_faces\\<Имя>\\*.jpg")
        return 1

    persons = [d for d in sorted(os.listdir(known_dir))
               if os.path.isdir(os.path.join(known_dir, d))]
    if not persons:
        print(f"В {known_dir} нет папок-людей. Структура: known_faces\\<Имя>\\*.jpg")
        return 1

    print("Инициализация FaceEngine (детекция + распознавание)...")
    engine = FaceEngine(
        model_name=cfg["recognition"]["model_name"],
        det_size=(cfg["recognition"]["det_size"], cfg["recognition"]["det_size"]),
        ctx_id=cfg["gpu"]["ctx_id"],
        allowed_modules=["detection", "recognition"],
    )
    print(f"GPU = {engine.on_gpu}\n")

    names: list[str] = []
    vectors: list[np.ndarray] = []

    for person in persons:
        pdir = os.path.join(known_dir, person)
        images = list_person_images(pdir)
        if not images:
            print(f"[ПРОПУСК] {person}: нет изображений")
            continue

        embs = []
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ! {person}: не читается {os.path.basename(img_path)}")
                continue
            faces = detect_robust(engine, img)
            if not faces:
                print(f"  ! {person}: лицо не найдено на {os.path.basename(img_path)}")
                continue
            if len(faces) > 1:
                print(f"  ~ {person}: на {os.path.basename(img_path)} "
                      f"{len(faces)} лиц — беру самое крупное")
            face = pick_largest_face(faces)
            embs.append(face.normed_embedding)   # уже L2-нормализован

        if not embs:
            print(f"[ПРОПУСК] {person}: ни одного валидного лица")
            continue

        # средний эмбеддинг по человеку -> ре-нормализация
        mean_vec = fdb.l2_normalize(np.mean(np.stack(embs), axis=0))
        names.append(person)
        vectors.append(mean_vec)
        print(f"[OK] {person}: {len(embs)} фото -> 1 вектор")

    if not vectors:
        print("\nНе удалось зарегистрировать ни одного человека.")
        return 1

    mat = np.stack(vectors).astype(np.float32)
    index_path, labels_path = index_paths(cfg)
    fdb.build_index(mat, names, index_path, labels_path)

    print(f"\nГотово. В базе {len(names)} чел.: {', '.join(names)}")
    print(f"Индекс : {index_path}")
    print(f"Метки  : {labels_path}")
    return 0


if __name__ == "__main__":
    sys.exit(enroll(load_settings()))
