# -*- coding: utf-8 -*-
"""
tracker.py — Этап 4+. Лёгкий трекинг лиц по одной камере (IoU) + стабилизация ID.

Зачем: решение «по одному кадру» неустойчиво — при повороте головы/смазе эмбеддинг
плывёт, score падает, и появляется дубль ID. Трекер связывает лица между кадрами
по пересечению боксов и:
  - ДЕРЖИТ присвоенный ID, пока трек жив (плохие кадры не сбрасывают личность);
  - заводит НОВЫЙ ID только если кандидат стабильно виден несколько кадров
    И это фронтальное/чёткое лицо (гейт качества в Gallery);
  - в «серой зоне» (0.32..0.45) отдаёт ближайший существующий ID, а не новый.

Один трекер на камеру. Вызывается из единственного inference-потока, блокировки не нужны.
"""
from dataclasses import dataclass

from gallery import Gallery, frontality
from results import FaceResult
from face_quality import FaceQuality


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


class _Track:
    __slots__ = ("bbox", "label", "crop_path", "hits", "misses", "candidate_frames")

    def __init__(self, bbox):
        self.bbox = bbox
        self.label = None          # присвоенный ID (или None, пока не решили)
        self.crop_path = ""
        self.hits = 0
        self.misses = 0
        self.candidate_frames = 0  # сколько кадров держится как качественный «новый» кандидат


class CameraTracker:
    def __init__(self, gallery: Gallery, cfg: dict):
        self.g = gallery
        gg = cfg["gallery"]
        self.iou_thr = float(gg["track_iou"])
        self.max_misses = int(gg["track_max_misses"])
        self.confirm = int(gg["new_id_confirm_frames"])
        self.fq = FaceQuality(cfg)         # фильтр качества (Задача 1)
        self._scale = 1.0                  # коэффициент ресайза кадра (для размера в исходных px)
        self.tracks: list[_Track] = []

    def update(self, faces, frame, ts, scale: float = 1.0) -> list:
        """faces — список insightface Face (bbox, normed_embedding, det_score, kps).
        scale — frame_w/original_w (чтобы мерить размер лица в исходных пикселях)."""
        self._scale = scale
        dets = [tuple(int(v) for v in f.bbox) for f in faces]

        # --- связывание детекций с существующими треками (жадно по IoU) ---
        pairs = []
        for di, dbox in enumerate(dets):
            for ti, t in enumerate(self.tracks):
                i = _iou(dbox, t.bbox)
                if i >= self.iou_thr:
                    pairs.append((i, di, ti))
        pairs.sort(reverse=True)

        det2track: dict[int, int] = {}
        used_d, used_t = set(), set()
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            det2track[di] = ti
            used_d.add(di); used_t.add(ti)

        results = []
        matched_tracks = set()
        new_tracks: list[_Track] = []

        for di, f in enumerate(faces):
            if di in det2track:
                t = self.tracks[det2track[di]]
                matched_tracks.add(det2track[di])
            else:
                t = _Track(dets[di])
                new_tracks.append(t)
            t.bbox = dets[di]
            t.hits += 1
            t.misses = 0
            res = self._decide(t, f, frame, ts)
            if res is not None:
                results.append(res)

        # --- старение треков, которые не обновились в этом кадре ---
        survivors = []
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.misses += 1
                if t.misses > self.max_misses:
                    continue
            survivors.append(t)
        self.tracks = survivors + new_tracks
        return results

    def _decide(self, t: _Track, f, frame, ts):
        emb = f.normed_embedding
        # оценка качества один раз на лицо (метрики идут во все события)
        q = self.fq.assess(f, frame, self._scale)

        def fr(label, score, is_new, crop):
            return FaceResult(t.bbox, label, score, is_new, crop,
                              q.det_score, q.width_px, q.blur, q.yaw_asym)

        # 1) Трек уже знает свою личность -> держим ID (FAISS-гейт не применяем)
        if t.label is not None:
            own = self.g.get_by_label(t.label)
            if own is None:
                # личность удалили (напр. из дашборда) — сбрасываем трек, переидентифицируем
                t.label = None
                t.crop_path = ""
            else:
                ident, score = self.g.identify(emb)
                if frontality(f.kps) >= self.g.min_frontality:
                    self.g.maybe_add_embedding(own, emb, score, ts)
                return fr(t.label, score, False, t.crop_path)

        # ★ ФИЛЬТР КАЧЕСТВА — перед FAISS, только для НЕопознанных лиц ★
        if self.fq.enabled and not q.passed:
            if self.fq.mode == "ignore":
                return None
            # mode == "event": фиксируем как LOW_QUALITY (снимок пишется при логировании)
            return fr("LOW_QUALITY", 0.0, False, "")

        # 2) Личность ещё не присвоена — ищем в галерее
        ident, score = self.g.identify(emb)

        if ident is not None and score >= self.g.match_threshold:
            # уверенное совпадение -> закрепляем существующий ID
            t.label = ident.label
            t.crop_path = ident.crop_path
            self.g.maybe_add_embedding(ident, emb, score, ts)
            return fr(ident.label, score, False, ident.crop_path)

        # 3) Ниже порога. Можно ли это качественный кандидат в НОВЫЙ ID?
        good = self.g.quality_ok_for_new(float(getattr(f, "det_score", 0.0)), t.bbox, f.kps, frame)
        if score < self.g.new_id_threshold and good:
            t.candidate_frames += 1
            if t.candidate_frames >= self.confirm:
                ident = self.g.add_new(emb, frame, f.bbox, ts)
                t.label = ident.label
                t.crop_path = ident.crop_path
                return fr(ident.label, score, True, ident.crop_path)
            return None  # ещё не подтверждён — ничего не выдаём (не мигаем)

        # 4) «Серая зона»: НЕ заводим новый ID. Если рядом есть существующий — отдаём его.
        if ident is not None and score >= self.g.new_id_threshold:
            return fr(ident.label, score, False, ident.crop_path)
        return None
