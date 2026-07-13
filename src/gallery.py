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
    best_quality: float = 0.0  # оценка качества текущего снимка (best-shot)


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
        # крупные лица почти не бывают ложными -> det-порог мягче
        self.large_face_px = int(g.get("new_id_large_face_px", 100))
        self.large_face_det = float(g.get("new_id_large_face_det", self.new_id_min_det))
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
        self.identities = [Identity(**d) for d in meta.get("identities", [])]
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

    def add_new(self, normed_emb, frame, bbox, ts: float,
                det_score: float = 0.0, kps=None, scale: float = 1.0) -> Identity:
        """Создать нового человека: ID + снимок + первый эмбеддинг.
        Снимок дальше улучшается best-shot'ом (maybe_update_crop)."""
        idx = len(self.identities)
        label = f"person_{self._next_num:04d}"
        self._next_num += 1

        crop = self._crop_face(frame, bbox)
        crop_path = os.path.join(self.faces_dir, f"{label}.jpg")
        if crop is not None:
            cv2.imwrite(crop_path, crop)

        ident = Identity(idx=idx, label=label,
                         crop_path=os.path.relpath(crop_path, _project_root()),
                         first_seen=ts, last_seen=ts, n_emb=0,
                         best_quality=round(shot_quality(det_score, bbox, kps, frame, scale), 4))
        self.identities.append(ident)
        self._append_embedding(idx, normed_emb)
        self.save()
        return ident

    def own_score(self, ident: Identity, emb: np.ndarray) -> float:
        """Похожесть эмбеддинга на СВОИ эмбеддинги этого человека (max cosine)."""
        rows = self.embeddings[self.owners == ident.idx]
        if rows.shape[0] == 0:
            return 0.0
        return float(np.max(rows @ emb.reshape(-1)))

    def maybe_update_crop(self, ident: Identity, frame, bbox, det_score: float,
                          kps, scale: float = 1.0):
        """
        Best-shot: если текущий кадр ЗАМЕТНО лучше сохранённого снимка этого
        человека — перезаписать person_XXXX.jpg. Фото «дозревает» до лучшего
        ракурса, пока человек в кадре (дашборд подхватит через ?v=mtime).
        Гистерезис +15%, чтобы не молотить диск на каждом кадре.
        """
        q = shot_quality(det_score, bbox, kps, frame, scale)
        if q <= ident.best_quality * 1.15:
            return
        crop = self._crop_face(frame, bbox)
        if crop is None:
            return
        abs_path = os.path.join(self.faces_dir, f"{ident.label}.jpg")
        cv2.imwrite(abs_path, crop)
        ident.best_quality = round(q, 4)
        self.save()

    def maybe_add_embedding(self, ident: Identity, emb: np.ndarray, ts: float,
                            quality_ok: bool = True):
        """
        Добавить ещё один ракурс. Защита от «отравления» галереи:
          - quality_ok: кадр прошёл те же гейты качества, что и для нового ID
            (плохой кадр может удерживать ID трека, но в базу не попадает);
          - похожесть на СВОИ эмбеддинги >= match_threshold (раньше сравнивали
            с глобальным ближайшим — чужой мусор попадал в ID и «слипал» людей);
          - < add_below: слишком похожий ракурс не даёт новой информации.
        """
        ident.last_seen = ts
        if not quality_ok or ident.n_emb >= self.max_emb:
            return
        s = self.own_score(ident, emb)
        if s < self.match_threshold or s >= self.add_below:
            return
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

    def quality_ok_for_new(self, det_score: float, bbox, kps, frame,
                           scale: float = 1.0) -> bool:
        """
        Можно ли по этому лицу заводить НОВЫЙ ID? Требуем:
        уверенную детекцию, крупный размер, фронтальность (не профиль), резкость.
        scale — frame_w/original_w: размер лица меряем в ИСХОДНЫХ пикселях.
        Det-порог адаптивный: мелким лицам строго (ложные детекции почти всегда
        мелкие), крупным — мягче (new_id_large_face_det).
        """
        x1, y1, x2, y2 = bbox
        px = min(x2 - x1, y2 - y1) / max(scale, 1e-6)
        if px < self.new_id_min_px:
            return False
        need_det = self.large_face_det if px >= self.large_face_px else self.new_id_min_det
        if det_score < need_det:
            return False
        if frontality(kps) < self.min_frontality:
            return False
        if self.min_blur > 0:
            # резкость меряем по ТЕСНОМУ bbox (как face_quality.py): кроп с полями
            # (_crop_face) добавляет гладкий фон и занижает дисперсию Лапласиана
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(w, int(x2)), min(h, int(y2))
            if cx2 <= cx1 or cy2 <= cy1:
                return False
            if blur_var(frame[cy1:cy2, cx1:cx2]) < self.min_blur:
                return False
        return True

    def count(self) -> int:
        return len(self.identities)


# ----------------------------- утилиты качества -----------------------------
def shot_quality(det_score, bbox, kps, frame, scale: float = 1.0) -> float:
    """
    Композитная оценка кадра для best-shot: det_score × yaw-фронтальность ×
    размер × резкость (мягко). Абсолютное значение не важно — сравниваем кадры
    ОДНОГО человека между собой (det_score хорошо коррелирует с ракурсом:
    анфас ~0.86, опущенная голова ~0.71 — замерено на реальных кадрах).
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        px = min(x2 - x1, y2 - y1) / max(scale, 1e-6)
        s_px = min(px / 120.0, 1.0)                    # насыщение на 120px
        h, w = frame.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return 0.0
        blur = blur_var(frame[cy1:cy2, cx1:cx2])
        s_blur = 0.5 + 0.5 * min(blur / 150.0, 1.0)    # смаз штрафует максимум вдвое
        # float(): frontality может вернуть numpy-скаляр, а он ломает json.dump в save()
        return float(float(det_score) * frontality(kps) * s_px * s_blur)
    except Exception:
        return 0.0


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
