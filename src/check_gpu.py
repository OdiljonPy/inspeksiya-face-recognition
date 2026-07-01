# -*- coding: utf-8 -*-
r"""
check_gpu.py — Этап 0. Диагностика GPU-окружения.

Печатает:
  1) версию torch, доступность CUDA, имя устройства, compute capability, список арх;
  2) версию onnxruntime и список доступных Execution Providers;
  3) РЕАЛЬНЫЙ прогон InsightFace (buffalo_l) и какой провайдер фактически использует
     каждая модель (детектор + распознавание) — т.е. реально ли работает GPU,
     а не просто "виден в списке".

Запуск:
    .\.venv\Scripts\python.exe src\check_gpu.py
"""
import sys
import time
import numpy as np

# --- ВАЖНО: подготовить DLL-пути для onnxruntime-gpu ДО его импорта ---
from gpu_setup import enable_onnx_cuda
_added = enable_onnx_cuda(verbose=False)


def hr(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ============================================================ 1. TORCH
hr("1) PyTorch / CUDA")
try:
    import torch
    print(f"torch.__version__       : {torch.__version__}")
    print(f"torch.version.cuda      : {torch.version.cuda}")
    print(f"cuda.is_available()     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device name             : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"compute capability      : sm_{cap[0]}{cap[1]}")
        print(f"supported arch list     : {torch.cuda.get_arch_list()}")
        # боевой прогон на GPU
        t0 = time.time()
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"GPU matmul 2048x2048    : OK ({(time.time()-t0)*1000:.1f} ms)")
        sm = f"sm_{cap[0]}{cap[1]}"
        if sm in torch.cuda.get_arch_list():
            print(f"=> {sm} ПОДДЕРЖИВАЕТСЯ этой сборкой torch")
        else:
            print(f"!! {sm} НЕ в списке арх — возможны падения/CPU-fallback")
except Exception as e:
    print(f"ОШИБКА torch: {e}")


# ====================================================== 2. ONNXRUNTIME
hr("2) onnxruntime")
ort = None
try:
    import onnxruntime as ort
    print(f"onnxruntime.__version__ : {ort.__version__}")
    print(f"available providers     : {ort.get_available_providers()}")
    print(f"device                  : {ort.get_device()}")
except Exception as e:
    print(f"ОШИБКА onnxruntime: {e}")


# ======================================================= 3. INSIGHTFACE
hr("3) InsightFace (buffalo_l) — реальный прогон")
try:
    from insightface.app import FaceAnalysis

    # Просим CUDA в приоритете, CPU как запасной
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("Инициализация FaceAnalysis(name='buffalo_l') ...")
    print("(при первом запуске модели скачаются в ~/.insightface/models, ~300 MB)")
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 -> GPU

    # Какой провайдер РЕАЛЬНО использует каждая модель
    print("\nФактические провайдеры по моделям:")
    any_cuda = False
    for taskname, model in app.models.items():
        try:
            used = model.session.get_providers()
        except Exception:
            used = ["<нет session>"]
        on_gpu = "CUDAExecutionProvider" in used
        any_cuda = any_cuda or on_gpu
        flag = "GPU " if on_gpu else "CPU "
        print(f"  [{flag}] {taskname:12s} -> {used}")

    # Боевой прогон на синтетическом кадре
    img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    t0 = time.time()
    faces = app.get(img)
    dt = (time.time() - t0) * 1000
    print(f"\napp.get() на тест-кадре : {len(faces)} лиц, {dt:.1f} ms")

    hr("ИТОГ")
    if any_cuda:
        print("OK: InsightFace РЕАЛЬНО использует CUDAExecutionProvider (GPU).")
        print("Этап 0 пройден — можно двигаться к Этапу 1.")
    else:
        print("ВНИМАНИЕ: InsightFace работает на CPU (CUDA-ядра sm_120 не загрузились).")
        print("Лиц немного — это допустимо как запасной вариант (см. ТЗ).")
        print("Детекцию/YOLO при необходимости держим на GPU через torch.")
except Exception as e:
    import traceback
    print(f"ОШИБКА InsightFace: {e}")
    traceback.print_exc()
    sys.exit(1)
