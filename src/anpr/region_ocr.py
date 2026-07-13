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

from anpr.plate_format import fix_region, normalize_plate, region_valid


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

    @staticmethod
    def _variants(crop):
        """Варианты препроцессинга: как есть + контраст/бинаризация (тёмные номера
        rapidocr в сыром виде не читает, а CLAHE+Otsu — читает)."""
        yield crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        _, binimg = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield cv2.cvtColor(binimg, cv2.COLOR_GRAY2BGR)

    def _try_read(self, img) -> str:
        """Один OCR-вызов + извлечение действующего кода региона ('' если не вышло)."""
        try:
            # text_score ниже дефолтных 0.5: результат валидируем сами по VALID_REGIONS,
            # а строгий порог молча отбрасывал читаемый регион
            result, _ = self._ocr(img, text_score=0.3)
        except Exception:
            return ""
        if not result:
            return ""
        text = "".join(str(seg[1]) for seg in result)
        # 1) прямые цифры из результата (только действующий код региона)
        digits = re.sub(r"\D", "", text)
        if region_valid(digits[:2]):
            return digits[:2]
        # 2) буквы, похожие на цифры (та же позиционная коррекция, что в plate_format)
        norm = normalize_plate(text)
        return fix_region(norm[:2]) if len(norm) >= 2 else ""

    def read_region(self, plate_crop) -> str:
        """Прочитать регион с кропа НОМЕРА. Вернуть действующий код или '' (не смогли)."""
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
            for variant in self._variants(crop):
                reg = self._try_read(variant)
                if reg:
                    return reg
        return ""
