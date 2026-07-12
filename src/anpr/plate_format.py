# -*- coding: utf-8 -*-
"""
plate_format.py — нормализация и валидация номерного знака (Узбекистан).

Шаблон формата вынесен в config (anpr.plate_regex), чтобы его можно было
поправить без изменения кода. Нормализатор приводит распознанный текст к
верхнему регистру и убирает всё, кроме латинских букв и цифр.
"""
import re
from dataclasses import dataclass


# Частые ошибки OCR кириллица->латиница и похожие глифы.
# Применяются ОПЦИОНАЛЬНО (см. PlateValidator.normalize, fix_confusions=False по умолчанию),
# т.к. агрессивная замена может навредить. Вынесено отдельно для прозрачности.
_CONFUSIONS = str.maketrans({
    "О": "O", "о": "O", "А": "A", "а": "A", "В": "B", "Е": "E", "е": "E",
    "К": "K", "М": "M", "Н": "H", "Р": "P", "С": "C", "с": "C", "Т": "T",
    "У": "Y", "Х": "X", "х": "X",
})


def normalize_plate(text: str, fix_confusions: bool = False) -> str:
    """UPPER + удалить всё, кроме [A-Z0-9]. Опционально чинит кириллицу->латиница."""
    if text is None:
        return ""
    if fix_confusions:
        text = text.translate(_CONFUSIONS)
    return re.sub(r"[^A-Z0-9]", "", text.upper())


@dataclass
class PlateParse:
    raw: str               # исходный текст OCR
    normalized: str        # UPPER без разделителей
    region: str            # первые 2 символа (регион)
    body: str              # остаток (тело номера)
    valid: bool            # полностью соответствует узбекскому regex
    region_uncertain: bool # регион не похож на 2 цифры (интерим: тело надёжно, регион — нет)
    region_fixed: bool = False  # регион восстановлен позиционной коррекцией букв->цифр


# Регион РУз — ВСЕГДА 2 цифры, а глобальная OCR-модель предпочитает буквы
# (01 -> CI/OI, 80 -> BO и т.п.). Позиционная коррекция ТОЛЬКО для регион-префикса:
# буквы, визуально похожие на цифры -> цифры. К телу номера НЕ применять.
_REGION_L2D = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "C": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8",
})


def fix_region(prefix: str) -> str:
    """
    Восстановить 2-значный регион из битого OCR-префикса ("CI" -> "01", "S0" -> "50").
    Возвращает исправленный регион или "" (если восстановить надёжно нельзя).
    """
    if len(prefix) != 2:
        return ""
    fixed = prefix.translate(_REGION_L2D)
    if fixed.isdigit() and 1 <= int(fixed) <= 99:
        return fixed
    return ""


# тело номера РУз без региона: "S772SB" (частный) / "123ABC" (юрлица) / "1234AB" (прицеп)
_BODY_ALTS = r"[A-Z][0-9]{3}[A-Z]{2}|[0-9]{3}[A-Z]{3}|[0-9]{4}[A-Z]{2}"
_BODY_RE = re.compile(rf"^({_BODY_ALTS})$")
_BODY_SUFFIX_RE = re.compile(rf"({_BODY_ALTS})$")


class PlateValidator:
    """Проверяет/разбирает нормализованный номер по regex узбекского формата из конфига."""

    def __init__(self, pattern: str, fix_confusions: bool = False):
        self.pattern = pattern
        self.re = re.compile(pattern)
        self.fix_confusions = fix_confusions

    def normalize(self, text: str) -> str:
        return normalize_plate(text, fix_confusions=self.fix_confusions)

    def is_valid(self, normalized: str) -> bool:
        return bool(self.re.fullmatch(normalized))

    def parse(self, text: str) -> PlateParse:
        """
        Разобрать номер на регион + тело. Интерим-логика: тело распознаётся надёжно,
        регион (2 цифры) — best-effort. region_uncertain=True, если регион не 2 цифры
        (типичный симптом сбоя OCR региона: 01->CI, 80->S и т.п.).
        """
        norm = self.normalize(text)
        valid = self.is_valid(norm)
        # тело ищем как СУФФИКС по шаблону — устойчиво к битому региону
        # (напр. "SV147NA": регион "S" сломан, но тело "V147NA" извлечётся верно)
        m = _BODY_SUFFIX_RE.search(norm)
        if m:
            body = m.group(1)
            region = norm[:m.start()]
            body_ok = True
        else:
            region, body = norm[:2], norm[2:]
            body_ok = bool(_BODY_RE.fullmatch(body))
        region_ok = region.isdigit() and len(region) == 2 and 1 <= int(region) <= 99
        region_fixed = False
        # регион битый, но тело надёжно -> пробуем позиционную коррекцию букв->цифр
        if body_ok and not region_ok:
            fixed = fix_region(region)
            if fixed:
                region = fixed
                norm = region + body
                valid = self.is_valid(norm)
                region_ok = True
                region_fixed = True
        # если тело валидно, но регион — нет, считаем регион ненадёжным (не валим всё событие)
        region_uncertain = body_ok and not region_ok
        return PlateParse(raw=text, normalized=norm, region=region, body=body,
                          valid=valid, region_uncertain=region_uncertain,
                          region_fixed=region_fixed)
