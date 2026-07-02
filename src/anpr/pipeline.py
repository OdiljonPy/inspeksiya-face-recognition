# -*- coding: utf-8 -*-
"""
pipeline.py — Этап 3/4 ANPR. Общий обработчик кадра: детекция -> разбор -> кроп -> лог.

Используется и тестом Этапа 3, и интеграцией в 10 камер (Этап 4), чтобы не дублировать
логику. На вход — кадр + метаданные камеры; на выход — список распознанных номеров,
параллельно сохраняются кропы в data/plates/ и пишутся события (с дедупом).
"""
import os
import time

import cv2

from anpr.plate_format import PlateValidator


def _safe_name(s: str) -> str:
    return "".join(c for c in s if c.isalnum()) or "NA"


def process_frame(engine, validator: PlateValidator, vlog, frame, cam_id, zone,
                  plates_dir, min_conf, ts=None, save_crop=True, object_id="default") -> list[dict]:
    """
    Прогнать кадр через ANPR. Для каждого номера выше порога:
      - разобрать (регион/тело, флаги),
      - сохранить кроп номера в plates_dir (если save_crop),
      - залогировать с анти-дребезгом по ТЕЛУ номера.
    Возвращает список словарей с результатами (для печати/дашборда).
    """
    if ts is None:
        ts = time.time()
    out = []
    plates = engine.predict(frame)
    for p in plates:
        if p.ocr_conf < min_conf:
            continue
        pp = validator.parse(p.text)
        # ключ дедупа — надёжное тело номера (регион может «плавать»)
        dedup_key = pp.body or pp.normalized

        # Сначала пытаемся залогировать (дедуп внутри). Кроп пишем ТОЛЬКО если
        # событие реально записано — иначе плодили бы орфан-кропы для дублей.
        rowid = vlog.log(
            camera_id=cam_id, zone=zone, plate_text=p.text,
            plate_normalized=pp.normalized, confidence=p.ocr_conf,
            snapshot_path="", dedup_key=dedup_key,
            valid=pp.valid, region_uncertain=pp.region_uncertain, ts=ts,
            object_id=object_id,
        ) if vlog is not None else None
        logged = rowid is not None

        snapshot_path = ""
        if logged and save_crop and p.bbox:
            x1, y1, x2, y2 = p.bbox
            h, w = frame.shape[:2]
            pad = 6
            crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size:
                os.makedirs(plates_dir, exist_ok=True)
                fname = f"{int(ts*1000)}_{cam_id}_{_safe_name(pp.normalized)}.jpg"
                snapshot_path = os.path.join(plates_dir, fname)
                cv2.imwrite(snapshot_path, crop)
                vlog.set_snapshot(rowid, snapshot_path)

        out.append({
            "normalized": pp.normalized, "body": pp.body, "region": pp.region,
            "valid": pp.valid, "region_uncertain": pp.region_uncertain,
            "ocr_conf": p.ocr_conf, "det_conf": p.det_conf,
            "snapshot_path": snapshot_path, "logged": logged,
        })
    return out
