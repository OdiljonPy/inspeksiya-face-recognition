# -*- coding: utf-8 -*-
"""
config.py — загрузка настроек из config/settings.yaml и config/cameras.yaml.

Все относительные пути в settings.yaml разрешаются в абсолютные относительно
корня проекта (родитель папки src), чтобы скрипты работали из любого каталога.
"""
import os
import yaml


def project_root() -> str:
    """Корень проекта = родитель папки src/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(path: str) -> str:
    """Относительный путь -> абсолютный относительно корня проекта."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root(), path))


def load_settings(path: str | None = None) -> dict:
    """Загрузить settings.yaml. Пути в секции paths делает абсолютными."""
    if path is None:
        path = os.path.join(project_root(), "config", "settings.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # разрешаем все пути
    paths = cfg.get("paths", {})
    for k, v in list(paths.items()):
        paths[k] = _abs(v)
    cfg["paths"] = paths
    return cfg


# объект по умолчанию для камер/событий без явного object_id
DEFAULT_OBJECT_ID = "default"
DEFAULT_OBJECT_NAME = "Объект по умолчанию"


def _cameras_path(path):
    return path or os.path.join(project_root(), "config", "cameras.yaml")


def load_cameras(path: str | None = None) -> list[dict]:
    """Загрузить cameras.yaml -> список камер. У каждой гарантируем object_id (дефолт)."""
    p = _cameras_path(path)
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cams = data.get("cameras", []) or []
    for c in cams:
        c.setdefault("object_id", DEFAULT_OBJECT_ID)
    return cams


def load_objects(path: str | None = None) -> list[dict]:
    """Загрузить objects из cameras.yaml. Всегда включает дефолтный объект."""
    p = _cameras_path(path)
    objs = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        objs = data.get("objects", []) or []
    ids = {o["id"] for o in objs}
    if DEFAULT_OBJECT_ID not in ids:
        objs.append({"id": DEFAULT_OBJECT_ID, "name": DEFAULT_OBJECT_NAME, "address": ""})
    return objs


# Удобные пути к артефактам индекса
def index_paths(cfg: dict) -> tuple[str, str]:
    """Вернуть (faiss.index, labels.json) внутри paths.index_dir."""
    idx_dir = cfg["paths"]["index_dir"]
    return (os.path.join(idx_dir, "faiss.index"),
            os.path.join(idx_dir, "labels.json"))
