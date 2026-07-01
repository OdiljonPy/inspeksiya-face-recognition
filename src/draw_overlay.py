# -*- coding: utf-8 -*-
"""
draw_overlay.py — общая отрисовка боксов лиц/номеров (для debug_stream и live в дашборде).

ВНИМАНИЕ: cv2.putText не рисует кириллицу -> все подписи ASCII.
"""
import cv2


def draw_faces(frame, engine, gallery, min_det):
    """Боксы лиц + размер в px (+ ID, если есть галерея). Возвращает (всего, годных)."""
    faces = engine.detect(frame)
    n_ok = 0
    for f in faces:
        det = float(getattr(f, "det_score", 0.0))
        x1, y1, x2, y2 = f.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        weak = det < min_det
        color = (0, 165, 255) if weak else (0, 220, 0)   # оранжевый = слабо
        if not weak:
            n_ok += 1
        label = f"face {w}x{h}px {det:.2f}"
        if gallery is not None and not weak:
            try:
                ident, score = gallery.identify(f.normed_embedding)
                if ident is not None and score >= gallery.match_threshold:
                    label = f"{ident.label} {score:.2f} ({w}x{h})"
            except Exception:
                pass
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return len(faces), n_ok


def draw_plates(frame, anpr, validator, min_conf):
    """Боксы номеров + текст + размер. Возвращает (всего, годных)."""
    plates = anpr.predict(frame)
    n_ok = 0
    for p in plates:
        if not p.bbox:
            continue
        x1, y1, x2, y2 = p.bbox
        w, h = x2 - x1, y2 - y1
        weak = p.ocr_conf < min_conf
        color = (0, 165, 255) if weak else (255, 120, 0)
        if not weak:
            n_ok += 1
        norm = validator.normalize(p.text)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{norm} {p.ocr_conf:.2f} {w}x{h}px", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return len(plates), n_ok


def hud(frame, lines):
    y = 18
    for ln in lines:
        cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 230, 255), 1, cv2.LINE_AA)
        y += 22
