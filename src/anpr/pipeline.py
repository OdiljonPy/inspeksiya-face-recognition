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

from anpr.plate_format import PlateValidator, owner_type_from_body


def _safe_name(s: str) -> str:
    return "".join(c for c in s if c.isalnum()) or "NA"


def process_frame(engine, validator: PlateValidator, vlog, frame, cam_id, zone,
                  plates_dir, min_conf, ts=None, save_crop=True, object_id="default",
                  full_dir=None, region_ocr=None, min_plate_px=0,
                  require_body=True, gai_checker=None) -> list[dict]:
    """
    Прогнать кадр через ANPR. Для каждого номера выше порога:
      - разобрать (регион/тело, флаги),
      - сохранить кроп номера в plates_dir (если save_crop),
      - залогировать с анти-дребезгом по ТЕЛУ номера,
      - для НОВЫХ событий: полный кадр в full_dir (если задан) и, при битом
        регионе, второй OCR-проход регион-бокса (region_ocr).
    Возвращает список словарей с результатами (для печати/дашборда).
    """
    if ts is None:
        ts = time.time()
    out = []
    plates = engine.predict(frame)
    for p in plates:
        if p.ocr_conf < min_conf:
            continue
        # слишком мелкий номер — уверенный мусор OCR, не логируем
        if min_plate_px and p.bbox and (p.bbox[2] - p.bbox[0]) < min_plate_px:
            continue
        pp = validator.parse(p.text)
        # фильтр ложных срабатываний детектора (фары/решётки): OCR даёт текст с
        # conf>=порога, но он НЕ похож на тело номера РУз — не логируем
        if require_body and not pp.body_ok:
            continue
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
            # базовый тип владельца по формату тела ("A123BC"=физлицо, "123ABC"=юрлицо);
            # фоновая проверка ГАИ уточнит его (в т.ч. kompaniya по ИНН генподрядчика)
            owner_type=owner_type_from_body(pp.body),
        ) if vlog is not None else None
        logged = rowid is not None

        snapshot_path = ""
        normalized, valid, region_uncertain = pp.normalized, pp.valid, pp.region_uncertain
        region = pp.region
        if logged and p.bbox:
            x1, y1, x2, y2 = p.bbox
            h, w = frame.shape[:2]
            pad = 6
            crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size:
                # регион всё ещё битый -> второй OCR-проход по регион-боксу (дёшево:
                # только для новых событий, единицы вызовов в минуту)
                if region_uncertain and pp.body and region_ocr is not None and region_ocr.ok:
                    reg = region_ocr.read_region(crop)
                    if reg:
                        region = reg
                        normalized = reg + pp.body
                        valid = validator.is_valid(normalized)
                        region_uncertain = False
                        vlog.update_region(rowid, normalized, valid)
                if save_crop:
                    os.makedirs(plates_dir, exist_ok=True)
                    fname = f"{int(ts*1000)}_{cam_id}_{_safe_name(normalized)}.jpg"
                    snapshot_path = os.path.join(plates_dir, fname)
                    cv2.imwrite(snapshot_path, crop)
                    vlog.set_snapshot(rowid, snapshot_path)
        # НОВОЕ событие -> фоновая проверка по базе ГАИ (после коррекции региона,
        # чтобы отправить исправленный номер; сам запрос — в отдельном потоке).
        # object_id нужен проверке для kompaniya (ИНН генподрядчика) и сверки с налогом.
        if logged and gai_checker is not None:
            gai_checker.enqueue(rowid, normalized, object_id)
        # полный кадр события (общий вид машины) — только для новых событий
        if logged and full_dir:
            os.makedirs(full_dir, exist_ok=True)
            full_path = os.path.join(full_dir, f"{int(ts*1000)}_{cam_id}_veh.jpg")
            cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            vlog.set_full(rowid, full_path)

        out.append({
            "normalized": normalized, "body": pp.body, "region": region,
            "valid": valid, "region_uncertain": region_uncertain,
            "ocr_conf": p.ocr_conf, "det_conf": p.det_conf,
            "snapshot_path": snapshot_path, "logged": logged,
        })
    return out
