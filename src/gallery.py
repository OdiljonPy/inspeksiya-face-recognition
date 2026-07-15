# -*- coding: utf-8 -*-
"""
gallery.py — Этап 4. Динамическая авто-галерея лиц (open-set re-identification).

Логика:
  - видим лицо -> ищем ближайший эмбеддинг в галерее;
  - если score >= match_threshold -> это уже известный человек (тот же ID);
  - иначе -> НОВЫЙ человек: присваиваем уникальный ID (person_0001, ...),
    сохраняем снимок лица в папку ОДИН РАЗ, заносим эмбеддинг в базу.

На один ID храним 1 снимок, но до max_embeddings_per_id эмбеддингов (разные
ракурсы) — это заметно повышает узнаваемость при повороте головы.

Хранилище (в gallery.dir):
  faces/<ID>.jpg     — один снимок на человека
  embeddings.npy     — все эмбеддинги (M, 512), float32
  owners.npy         — для каждой строки эмбеддинга: индекс владельца-Identity (M,)
  meta.json          — список Identity + счётчик следующего ID

Поиск — FAISS IndexFlatIP по ВСЕМ эмбеддингам (макс-похожесть среди всех ракурсов).
"""
import os
import json
import time
import threading
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import faiss


@dataclass
class Identity:
    idx: int            # порядковый индекс в списке identities
    label: str          # человекочитаемый ID, напр. "person_0001"
    crop_path: str      # путь к сохранённому снимку (относительно проекта)
    first_seen: float
    last_seen: float
    n_emb: int = 0      # сколько эмбеддингов в базе у этого человека


class Gallery:
    def __init__(self, cfg: dict, dim: int = 512):
        g = cfg["gallery"]
        # gallery.dir уже сделан абсолютным в config.load_settings? нет — он не в paths.
        self.dir = g["dir"] if os.path.isabs(g["dir"]) else \
            os.path.normpath(os.path.join(_project_root(), g["dir"]))
        self.faces_dir = os.path.join(self.dir, "faces")
        self.match_threshold = float(g["match_threshold"])
        self.new_id_threshold = float(g["new_id_threshold"])
        self.max_emb = int(g["max_embeddings_per_id"])
        self.add_below = float(g["add_embedding_below"])
        self.crop_margin = float(g["crop_margin"])
        self.new_id_min_det = float(g["new_id_min_det_score"])
        self.new_id_min_px = int(g["new_id_min_face_px"])
        self.min_frontality = float(g["new_id_min_frontality"])
        self.min_blur = float(g["new_id_min_blur"])
        self.dim = dim

        self.lock = threading.Lock()
        self.identities: list[Identity] = []
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.owners = np.zeros((0,), dtype=np.int64)
        self.index = faiss.IndexFlatIP(dim)
        self._next_num = 1
        self._loaded_mtime = 0.0        # mtime meta.json на момент нашей последней записи/чтения

        os.makedirs(self.faces_dir, exist_ok=True)
        self.load()

    # ------------------------- персистентность -------------------------
    def _paths(self):
        return (os.path.join(self.dir, "embeddings.npy"),
                os.path.join(self.dir, "owners.npy"),
                os.path.join(self.dir, "meta.json"))

    def load(self):
        emb_p, own_p, meta_p = self._paths()
        if not os.path.exists(meta_p):
            return
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # meta.json мог быть записан другой версией кода (лишние поля, напр.
        # best_quality) — незнакомые ключи молча отбрасываем, а не падаем
        fields = Identity.__dataclass_fields__
        self.identities = [Identity(**{k: v for k, v in d.items() if k in fields})
                           for d in meta.get("identities", [])]
        self._next_num = meta.get("next_num", len(self.identities) + 1)
        self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
        self.owners = np.zeros((0,), dtype=np.int64)
        self.index = faiss.IndexFlatIP(self.dim)
        if os.path.exists(emb_p) and os.path.exists(own_p):
            self.embeddings = np.load(emb_p).astype(np.float32)
            self.owners = np.load(own_p).astype(np.int64)
            if self.embeddings.shape[0]:
                self.index.add(self.embeddings)
        try:
            self._loaded_mtime = os.path.getmtime(meta_p)
        except OSError:
            self._loaded_mtime = 0.0
        print(f"[gallery] загружено: {len(self.identities)} ID, "
              f"{self.embeddings.shape[0]} эмбеддингов")

    def maybe_reload(self):
        """
        Если meta.json изменён ДРУГИМ процессом (напр. дашборд удалил человека) —
        перечитать галерею с диска. Так удаление из дашборда «прилипает», а процесс
        распознавания не воскрешает удалённого. Свои же записи (save обновляет mtime)
        перечитку не триггерят. Дёшево: один stat.
        """
        _, _, meta_p = self._paths()
        try:
            m = os.path.getmtime(meta_p)
        except OSError:
            return
        if m != self._loaded_mtime:
            with self.lock:
                self.load()

    def save(self):
        emb_p, own_p, meta_p = self._paths()
        np.save(emb_p, self.embeddings)
        np.save(own_p, self.owners)
        meta = {
            "next_num": self._next_num,
            "identities": [asdict(i) for i in self.identities],
        }
        tmp = meta_p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, meta_p)  # атомарная замена (важно: meta читает дашборд)
        try:
            self._loaded_mtime = os.path.getmtime(meta_p)  # чтобы не перечитывать свою же запись
        except OSError:
            pass

    # ------------------------- основная логика -------------------------
    def identify(self, normed_emb: np.ndarray):
        """Вернуть (Identity|None, score) ближайшего совпадения."""
        if self.index.ntotal == 0:
            return None, 0.0
        q = np.ascontiguousarray(normed_emb.reshape(1, -1), dtype=np.float32)
        D, I = self.index.search(q, 1)
        score = float(D[0, 0])
        row = int(I[0, 0])
        if row < 0:
            return None, 0.0
        return self.identities[int(self.owners[row])], score

    def _crop_face(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * self.crop_margin), int(bh * self.crop_margin)
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _append_embedding(self, owner_idx: int, emb: np.ndarray):
        row = np.ascontiguousarray(emb.reshape(1, -1), dtype=np.float32)
        self.embeddings = np.vstack([self.embeddings, row])
        self.owners = np.concatenate([self.owners, [owner_idx]])
        self.index.add(row)
        self.identities[owner_idx].n_emb += 1

    def add_new(self, normed_emb, frame, bbox, ts: float) -> Identity:
        """Создать нового человека: ID + снимок (один раз) + первый эмбеддинг."""
        idx = len(self.identities)
        label = f"person_{self._next_num:04d}"
        self._next_num += 1

        crop = self._crop_face(frame, bbox)
        crop_path = os.path.join(self.faces_dir, f"{label}.jpg")
        if crop is not None:
            cv2.imwrite(crop_path, _enhance_gallery_crop(crop),
                        [cv2.IMWRITE_JPEG_QUALITY, 97])

        ident = Identity(idx=idx, label=label,
                         crop_path=os.path.relpath(crop_path, _project_root()),
                         first_seen=ts, last_seen=ts, n_emb=0)
        self.identities.append(ident)
        self._append_embedding(idx, normed_emb)
        self.save()
        return ident

    def maybe_add_embedding(self, ident: Identity, emb: np.ndarray, score: float, ts: float):
        """Добавить ещё один ракурс, если их мало и кадр достаточно «другой»."""
        ident.last_seen = ts
        if ident.n_emb < self.max_emb and score < self.add_below:
            self._append_embedding(ident.idx, emb)
            self.save()

    def get_by_label(self, label: str):
        for i in self.identities:
            if i.label == label:
                return i
        return None

    def delete_identity(self, label: str) -> str | None:
        """
        Удалить человека из галереи: его эмбеддинги, запись, снимок; переиндексировать.
        Возвращает label при успехе или None, если такого ID нет.
        ВАЖНО: owners хранит позиционный idx владельца — при удалении сдвигаем индексы.
        """
        with self.lock:
            ident = self.get_by_label(label)
            if ident is None:
                return None
            k = ident.idx

            # выкидываем строки эмбеддингов этого человека
            keep = self.owners != k
            self.embeddings = self.embeddings[keep]
            ow = self.owners[keep]
            # индексы владельцев после k сдвигаются на -1
            self.owners = np.where(ow > k, ow - 1, ow).astype(np.int64)

            # удаляем саму личность и переиндексируем остальные
            crop_path = ident.crop_path
            del self.identities[k]
            for i in range(k, len(self.identities)):
                self.identities[i].idx = i

            # пересобираем FAISS-индекс
            self.index = faiss.IndexFlatIP(self.dim)
            if self.embeddings.shape[0]:
                self.index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))

            # удаляем файл снимка
            abs_crop = crop_path if os.path.isabs(crop_path) else \
                os.path.join(_project_root(), crop_path)
            try:
                if os.path.exists(abs_crop):
                    os.remove(abs_crop)
            except OSError:
                pass

            self.save()
            return label

    def quality_ok_for_new(self, det_score: float, bbox, kps, frame) -> bool:
        """
        Можно ли по этому лицу заводить НОВЫЙ ID? Требуем:
        уверенную детекцию, крупный размер, фронтальность (не профиль), резкость.
        """
        if det_score < self.new_id_min_det:
            return False
        x1, y1, x2, y2 = bbox
        if min(x2 - x1, y2 - y1) < self.new_id_min_px:
            return False
        if frontality(kps) < self.min_frontality:
            return False
        if self.min_blur > 0:
            crop = self._crop_face(frame, bbox)
            if crop is None or blur_var(crop) < self.min_blur:
                return False
        return True

    def count(self) -> int:
        return len(self.identities)


def _enhance_gallery_crop(crop, target_min: int = 256, max_scale: float = 2.5):
    """
    Подготовить снимок лица для галереи. Кропы с камер мелкие (~100-180px),
    браузер растягивает их мыльно. Апскейлим сами качественным фильтром
    (LANCZOS4) + лёгкое повышение резкости. Деталей это не добавляет, но
    фото в дашборде выглядит заметно чище. Крупные кропы не трогаем.
    """
    h, w = crop.shape[:2]
    m = min(h, w)
    if m >= target_min:
        return crop
    s = min(max_scale, target_min / m)
    up = cv2.resize(crop, (int(round(w * s)), int(round(h * s))),
                    interpolation=cv2.INTER_LANCZOS4)
    # unsharp mask: мягко вернуть контуры после интерполяции
    blur = cv2.GaussianBlur(up, (0, 0), 1.2)
    return cv2.addWeighted(up, 1.35, blur, -0.35, 0)


# ----------------------------- утилиты качества -----------------------------
def frontality(kps) -> float:
    """
    Оценка фронтальности лица [0..1] по 5 ключевым точкам insightface
    (порядок: левый глаз, правый глаз, нос, левый угол рта, правый угол рта).
    Идея: у анфаса нос по горизонтали посередине между глазами; у профиля —
    смещён к одному глазу. 1 = анфас, 0 = профиль.
    """
    try:
        import numpy as _np
        kps = _np.asarray(kps, dtype=_np.float32)
        le_x, re_x, nose_x = kps[0, 0], kps[1, 0], kps[2, 0]
        span = re_x - le_x
        if abs(span) < 1e-3:
            return 0.0
        ratio = (nose_x - le_x) / span          # ~0.5 у анфаса
        return max(0.0, 1.0 - 2.0 * abs(ratio - 0.5))
    except Exception:
        return 1.0                              # не смогли оценить — не мешаем


def blur_var(crop) -> float:
    """Резкость = дисперсия Лапласиана (чем больше, тем чётче)."""
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 1e9


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
