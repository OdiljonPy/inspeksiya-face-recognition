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

import numpy as np

from gallery import Gallery, frontality, blur_var, aggregate_embeddings
from results import FaceResult
from face_quality import FaceQuality


def _mean_pairwise_cosine(embs) -> float:
    """Средний попарный cosine L2-норм. эмбеддингов (самосогласованность буфера)."""
    m = np.asarray(embs, dtype=np.float32)
    k = m.shape[0]
    if k < 2:
        return 1.0
    s = m @ m.T
    return float((s.sum() - np.trace(s)) / (k * (k - 1)))


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
    __slots__ = ("bbox", "label", "crop_path", "hits", "misses", "candidate_frames",
                 "te_embs", "te_best_w", "te_best_crop", "te_best_ts",
                 "te_first_ts", "te_last_score", "agg_embs")

    def __init__(self, bbox):
        self.bbox = bbox
        self.label = None          # присвоенный ID (или None, пока не решили)
        self.crop_path = ""
        self.hits = 0
        self.misses = 0
        self.candidate_frames = 0  # сколько кадров держится как качественный «новый» кандидат
        # --- track_enroll: буфер доказательств для отложенного создания ID ---
        self.te_embs = []          # [(вес качества, эмбеддинг)] кандидат-кадров
        self.te_best_w = 0.0       # лучший вес — его кадр станет фото галереи
        self.te_best_crop = None   # кроп лица лучшего кадра (с полями)
        self.te_best_ts = 0.0
        self.te_first_ts = 0.0     # ts первого кандидат-кадра (для max_wait)
        self.te_last_score = 0.0   # последний FAISS-score (для события is_new)
        # шаг 4: скользящий буфер ВСЕХ кадров трека для матчинга по агрегату
        self.agg_embs = []         # [(вес, эмбеддинг)], последние match_agg_frames


class CameraTracker:
    def __init__(self, gallery: Gallery, cfg: dict):
        self.g = gallery
        gg = cfg["gallery"]
        self.iou_thr = float(gg["track_iou"])
        self.max_misses = int(gg["track_max_misses"])
        self.confirm = int(gg["new_id_confirm_frames"])
        # track_enroll (шаг 1): отложенное создание ID из агрегата лучших кадров
        # трека. ВЫКЛЮЧЕНО по умолчанию — без секции в конфиге поведение старое.
        te = gg.get("track_enroll") or {}
        self.te_enabled = bool(te.get("enabled", False))
        self.te_topk = int(te.get("topk", 5))
        self.te_min_frames = int(te.get("min_frames", 3))
        self.te_max_wait = float(te.get("max_wait_seconds", 10.0))
        # шаг 2: самосогласованность буфера — настоящее лицо даёт похожие
        # эмбеддинги между кадрами, текстура/ложная детекция — случайные.
        self.te_min_consistency = float(te.get("min_consistency", 0.35))
        # шаг 3: кандидат подтверждается матчем ДРУГОГО трека не раньше чем через
        # gap после создания (иначе дробление трека подтверждало бы мгновенно).
        self.te_confirm_gap = float(te.get("confirm_min_gap_seconds", 300))
        # шаг 4: матчить неопознанный трек по СКОЛЬЗЯЩЕМУ АГРЕГАТУ его кадров,
        # а не по одиночному эмбеддингу (score стабилизируется на мелких лицах).
        # Отдельный под-флаг: включать ПОСЛЕ живой проверки шагов 1-3.
        self.te_match_agg = bool(te.get("match_by_aggregate", False))
        self.te_agg_frames = int(te.get("match_agg_frames", 5))
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
                    # track_enroll: трек умирает с достаточными доказательствами
                    # (человек быстро прошёл кадр) — доводим создание ID до конца
                    if self.te_enabled and t.label is None and \
                            len(t.te_embs) >= self.te_min_frames:
                        sc = t.te_last_score
                        ident = self._te_commit(t, ts)
                        if ident is not None:
                            results.append(FaceResult(t.bbox, ident.label, sc,
                                                      True, ident.crop_path))
                    continue
            survivors.append(t)
        self.tracks = survivors + new_tracks
        return results

    def _decide(self, t: _Track, f, frame, ts):
        emb = f.normed_embedding
        # фильтр качества считаем ТОЛЬКО если включён (иначе поведение как до Задачи 1)
        q = self.fq.assess(f, frame, self._scale) if self.fq.enabled else None

        def fr(label, score, is_new, crop):
            if q is None:
                return FaceResult(t.bbox, label, score, is_new, crop)
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
                    # best-shot: чёткий фронтальный кадр заменяет фото галереи
                    self.g.maybe_update_photo(own, frame, f.bbox,
                                              self._te_weight(f, frame))
                return fr(t.label, score, False, t.crop_path)

        # ★ ФИЛЬТР КАЧЕСТВА — перед FAISS, только для НЕопознанных лиц (если включён) ★
        if q is not None and not q.passed:
            if self.fq.mode == "ignore":
                return None
            # mode == "event": фиксируем как LOW_QUALITY (снимок пишется при логировании)
            return fr("LOW_QUALITY", 0.0, False, "")

        # 2) Личность ещё не присвоена — ищем в галерее.
        # Шаг 4 (под-флаг): ищем по скользящему агрегату кадров трека — одиночный
        # эмбеддинг на мелком лице шумит (0.19..0.86 у одного человека), среднее
        # last-K кадров стабильнее. Плохие кадры входят с малым весом.
        if self.te_enabled and self.te_match_agg:
            t.agg_embs.append((self._te_weight(f, frame),
                               np.array(emb, dtype=np.float32, copy=True)))
            if len(t.agg_embs) > self.te_agg_frames:
                t.agg_embs.pop(0)
            agg = aggregate_embeddings([e for _, e in t.agg_embs],
                                       [w for w, _ in t.agg_embs])
            ident, score = self.g.identify(agg if agg is not None else emb)
        else:
            ident, score = self.g.identify(emb)

        if ident is not None and score >= self.g.match_threshold:
            # уверенное совпадение -> закрепляем существующий ID
            t.label = ident.label
            t.crop_path = ident.crop_path
            # шаг 3: НОВЫЙ трек уверенно матчит кандидата спустя gap — подтверждаем
            if self.te_enabled and getattr(ident, "provisional", False) and \
                    (ts - ident.first_seen) >= self.te_confirm_gap:
                self.g.confirm_identity(ident)
            self.g.maybe_add_embedding(ident, emb, score, ts)
            if frontality(f.kps) >= self.g.min_frontality:
                # best-shot: чёткий фронтальный кадр заменяет фото галереи
                self.g.maybe_update_photo(ident, frame, f.bbox,
                                          self._te_weight(f, frame))
            return fr(ident.label, score, False, ident.crop_path)

        # 3) Ниже порога. Можно ли это качественный кандидат в НОВЫЙ ID?
        good = self.g.quality_ok_for_new(float(getattr(f, "det_score", 0.0)), t.bbox, f.kps, frame)
        if score < self.g.new_id_threshold and good:
            if self.te_enabled:
                # track_enroll: копим доказательства, ID создаём отложенно из агрегата
                return self._te_collect(t, f, emb, frame, ts, score, fr)
            t.candidate_frames += 1
            if t.candidate_frames >= self.confirm:
                ident = self.g.add_new(emb, frame, f.bbox, ts,
                                       photo_q=self._te_weight(f, frame))
                t.label = ident.label
                t.crop_path = ident.crop_path
                return fr(ident.label, score, True, ident.crop_path)
            return None  # ещё не подтверждён — ничего не выдаём (не мигаем)

        # 4) «Серая зона»: НЕ заводим новый ID. Если рядом есть существующий — отдаём его.
        if ident is not None and score >= self.g.new_id_threshold:
            return fr(ident.label, score, False, ident.crop_path)
        return None

    # ------------------- track_enroll: отложенное создание ID -------------------
    def _te_weight(self, f, frame) -> float:
        """Вес кадра-доказательства: det × фронтальность × нормированная резкость."""
        det = float(getattr(f, "det_score", 0.0))
        fro = frontality(f.kps)
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        b = blur_var(crop) if crop.size else 0.0
        return float(det * max(fro, 1e-3) * min(b / 100.0, 1.0))

    def _te_collect(self, t: _Track, f, emb, frame, ts, score, fr):
        """Кандидат-кадр в буфер трека; при наборе topk кадров (или по таймауту) —
        коммит: ID из взвешенного среднего эмбеддингов, фото — лучший кадр."""
        w = self._te_weight(f, frame)
        t.te_embs.append((w, np.array(emb, dtype=np.float32, copy=True)))
        if t.te_first_ts == 0.0:
            t.te_first_ts = ts
        t.te_last_score = score
        if w > t.te_best_w or t.te_best_crop is None:
            crop = self.g._crop_face(frame, f.bbox)
            if crop is not None:
                t.te_best_w, t.te_best_crop, t.te_best_ts = w, crop, ts
        if len(t.te_embs) >= self.te_topk or (ts - t.te_first_ts) >= self.te_max_wait:
            ident = self._te_commit(t, ts)
            if ident is not None:
                return fr(ident.label, score, True, ident.crop_path)
        return None  # копим доказательства — ничего не выдаём (не мигаем)

    def _te_commit(self, t: _Track, ts):
        """Создать ID из накопленных доказательств. None — доказательств мало."""
        embs = t.te_embs
        best_crop, best_ts, best_w = t.te_best_crop, t.te_best_ts, t.te_best_w
        self._te_reset(t)
        if len(embs) < self.te_min_frames or best_crop is None:
            return None
        top = sorted(embs, key=lambda x: x[0], reverse=True)[:self.te_topk]
        # шаг 2: гейт самосогласованности — средний попарный cosine кадров буфера.
        # Один человек за секунды с одной камеры проходит легко; текстура даёт
        # каждый кадр «другое лицо» -> ID не создаётся вовсе (главный источник мусора).
        if self.te_min_consistency > 0 and \
                _mean_pairwise_cosine([e for _, e in top]) < self.te_min_consistency:
            return None
        agg = aggregate_embeddings([e for _, e in top], [w for w, _ in top])
        if agg is None:
            return None
        ident = self.g.add_new_from_track(agg, best_crop, best_ts or ts,
                                          photo_q=best_w)
        t.label = ident.label
        t.crop_path = ident.crop_path
        return ident

    @staticmethod
    def _te_reset(t: _Track):
        t.te_embs = []
        t.te_best_w = 0.0
        t.te_best_crop = None
        t.te_best_ts = 0.0
        t.te_first_ts = 0.0
