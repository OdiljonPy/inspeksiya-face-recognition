# -*- coding: utf-8 -*-
"""
region_ocr.py — второй OCR-проход ТОЛЬКО по регион-боксу номера.

На узбекских номерах регион (2 цифры) стоит слева и отделён разделителем.
Основная OCR-модель fast-alpr глобальная и часто бьёт регион (01 -> CI).
Здесь: кроп левой части номера + апскейл + rapidocr, из результата берём цифры.

Вызывается ТОЛЬКО для уже залогированных событий с region_uncertain — то есть
единицы раз в минуту, а не на каждый кадр. Если rapidocr не установлен,
модуль тихо выключается (self.ok = False).
"""
import re

import cv2

from anpr.plate_format import fix_region, normalize_plate


class RegionOCR:
    # Доля ширины номера слева, где лежит регион-бокс, зависит от ракурса/кропа.
    # Перебираем несколько значений — первое валидное чтение побеждает
    # (на реальных кропах 0.25 часто читается там, где 0.32 захватывает лишний символ).
    FRACS = (0.25, 0.32, 0.40)

    def __init__(self, upscale: int = 4):
        self.upscale = int(upscale)
        self._ocr = None
        self.ok = False
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self.ok = True
        except Exception as e:                   # rapidocr не установлен/не завёлся
            print(f"[region_ocr] rapidocr недоступен ({e}) — второй проход региона выключен")

    def read_region(self, plate_crop) -> str:
        """Прочитать регион с кропа НОМЕРА. Вернуть '01'..'99' или '' (не смогли)."""
        if not self.ok or plate_crop is None or getattr(plate_crop, "size", 0) == 0:
            return ""
        h, w = plate_crop.shape[:2]
        for frac in self.FRACS:
            rw = max(1, int(w * frac))
            crop = plate_crop[:, :rw]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (rw * self.upscale, h * self.upscale),
                              interpolation=cv2.INTER_CUBIC)
            try:
                # text_score ниже дефолтных 0.5: цифры валидируем сами (диапазон 01-99),
                # а строгий порог молча отбрасывал читаемый регион
                result, _ = self._ocr(crop, text_score=0.3)
            except Exception:
                return ""
            if not result:
                continue
            text = "".join(str(seg[1]) for seg in result)
            # 1) прямые цифры из результата
            digits = re.sub(r"\D", "", text)
            if len(digits) >= 2 and 1 <= int(digits[:2]) <= 99:
                return digits[:2]
            # 2) буквы, похожие на цифры (та же позиционная коррекция, что в plate_format)
            norm = normalize_plate(text)
            if len(norm) >= 2:
                reg = fix_region(norm[:2])
                if reg:
                    return reg
        return ""
