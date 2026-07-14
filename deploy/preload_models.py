# -*- coding: utf-8 -*-
"""
preload_models.py — предзагрузка всех моделей (устойчиво к блокировке GitHub).

Скачивает:
  1) InsightFace buffalo_l — с HF-зеркала (GitHub-релизы часто недоступны в регионе);
  2) fast-alpr: YOLO-детектор номера + OCR (качаются с CDN при инициализации ALPR);
  3) RapidOCR (PP-OCR) модели (region OCR).

Запускается при сборке Docker-образа (модели вшиваются) или один раз на сервере.
Идемпотентен: уже скачанное не трогает.
"""
import os
import sys
import zipfile
import urllib.request

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

BUFFALO_URL = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l.zip"


def preload_buffalo_l() -> None:
    """buffalo_l в ~/.insightface/models/buffalo_l (с HF-зеркала, не с GitHub)."""
    home = os.environ.get("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))
    models = os.path.join(home, "models")
    target = os.path.join(models, "buffalo_l")
    if os.path.isdir(target) and os.path.exists(os.path.join(target, "w600k_r50.onnx")):
        print(f"[preload] buffalo_l уже на месте: {target}")
        return
    os.makedirs(models, exist_ok=True)
    zip_path = os.path.join(models, "buffalo_l.zip")
    print(f"[preload] качаю buffalo_l с зеркала HF -> {zip_path}")
    urllib.request.urlretrieve(BUFFALO_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        # в архиве .onnx лежат в корне -> кладём в подпапку buffalo_l
        os.makedirs(target, exist_ok=True)
        for name in z.namelist():
            if name.endswith(".onnx"):
                data = z.read(name)
                with open(os.path.join(target, os.path.basename(name)), "wb") as f:
                    f.write(data)
    os.remove(zip_path)
    print(f"[preload] buffalo_l готов: {target}")


def preload_anpr_and_rapidocr() -> None:
    """Инициализация движков триггерит скачивание их моделей (CDN, не GitHub)."""
    from gpu_setup import enable_onnx_cuda
    enable_onnx_cuda()
    from config import load_settings
    cfg = load_settings()

    print("[preload] fast-alpr (YOLO детектор + OCR)...")
    from anpr.engine import AnprEngine
    AnprEngine(cfg)  # скачает detector_model и ocr_model

    print("[preload] RapidOCR (PP-OCR для региона)...")
    try:
        from rapidocr_onnxruntime import RapidOCR
        RapidOCR()
    except Exception as e:
        print(f"[preload] RapidOCR предупреждение: {e}")


def check_person_model() -> None:
    """Модель человека (yolov8n.onnx) приезжает через git (data/models/) —
    экспорт из .pt требует torch, которого на сервере нет."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "data", "models", "yolov8n.onnx")
    if os.path.exists(p):
        print(f"[preload] модель человека на месте: {p}")
    else:
        print("[preload] ВНИМАНИЕ: data/models/yolov8n.onnx нет — детектор человека "
              "(person-first, debug_stream) работать не будет. Модель должна прийти "
              "через git; экспорт делается на dev: ultralytics YOLO('yolov8n.pt')"
              ".export(format='onnx') (yolov8n.pt — с HF-зеркала Ultralytics/YOLOv8).")


def main() -> int:
    try:
        preload_buffalo_l()
    except Exception as e:
        print(f"[preload] ОШИБКА buffalo_l: {e}")
        return 1
    check_person_model()
    try:
        preload_anpr_and_rapidocr()
    except Exception as e:
        print(f"[preload] ОШИБКА ANPR/RapidOCR: {e}")
        return 1
    print("[preload] Все модели готовы.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
