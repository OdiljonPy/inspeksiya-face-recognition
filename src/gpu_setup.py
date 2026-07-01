# -*- coding: utf-8 -*-
"""
gpu_setup.py — кроссплатформенная подготовка CUDA для onnxruntime-gpu.

Проблема: onnxruntime-gpu должен найти CUDA/cuDNN/cuBLAS.
  - Windows (dev, RTX 5060/sm_120): DLL приносит пакет torch (cu128) и/или nvidia-*-cu12.
    Добавляем их папки в пути поиска DLL процесса (os.add_dll_directory).
  - Linux (сервер, Tesla T4/sm_75): если CUDA стоит системно (образ nvidia/cuda в Docker) —
    ничего делать не нужно. Если CUDA приехала pip-колёсами nvidia-*-cu12 — предзагружаем
    .so через ctypes(RTLD_GLOBAL) ДО импорта onnxruntime, чтобы линковщик их увидел.

Вызывать enable_onnx_cuda() нужно ДО первого `import onnxruntime` / insightface / fast_alpr.
torch НЕ обязателен (на сервере его нет).
"""
import os
import sys
import glob
import ctypes

# .so/.dll, от которых зависит CUDA execution provider onnxruntime
_LIB_PATTERNS_LINUX = [
    "libcudart.so*", "libcublas.so*", "libcublasLt.so*", "libcudnn*.so*",
    "libcufft.so*", "libcurand.so*", "libnvrtc.so*", "libnvJitLink.so*",
]


def _site_packages_dirs() -> list[str]:
    """Каталоги site-packages, где могут лежать пакеты nvidia-* / torch."""
    dirs = set()
    for p in sys.path:
        if p and os.path.isdir(p) and p.endswith("site-packages"):
            dirs.add(p)
    try:
        import site
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            if os.path.isdir(p):
                dirs.add(p)
    except Exception:
        pass
    return list(dirs)


def _candidate_lib_dirs(subdir: str) -> list[str]:
    """
    Папки с CUDA-библиотеками: torch/lib (если torch есть) + nvidia/*/<subdir>.
    subdir = 'bin' на Windows, 'lib' на Linux (так раскладывают nvidia-* колёса).
    """
    cands = []
    try:
        import torch  # необязателен
        cands.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    for sp in _site_packages_dirs():
        cands += glob.glob(os.path.join(sp, "nvidia", "*", subdir))
    return [d for d in cands if os.path.isdir(d)]


def _enable_windows(verbose: bool) -> list[str]:
    added = []
    for path in _candidate_lib_dirs("bin") + _candidate_lib_dirs("lib"):
        try:
            os.add_dll_directory(path)
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        added.append(path)
        if verbose:
            print(f"[gpu_setup] + DLL dir: {path}")
    return added


def _enable_linux(verbose: bool) -> list[str]:
    """Предзагрузка CUDA .so в глобальную область, чтобы onnxruntime их разрешил."""
    loaded = []
    for d in _candidate_lib_dirs("lib"):
        for pat in _LIB_PATTERNS_LINUX:
            for so in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(so)
                    if verbose:
                        print(f"[gpu_setup] preload: {so}")
                except OSError:
                    pass
    # Если pip-колёс нет (CUDA системная, напр. в Docker nvidia/cuda) — это норма,
    # onnxruntime найдёт библиотеки через системный ld.so.
    return loaded


def enable_onnx_cuda(verbose: bool = False) -> list[str]:
    """Подготовить CUDA-окружение для onnxruntime-gpu. Вернуть список путей/библиотек."""
    if sys.platform.startswith("win"):
        return _enable_windows(verbose)
    return _enable_linux(verbose)


if __name__ == "__main__":
    items = enable_onnx_cuda(verbose=True)
    print(f"OS={sys.platform}, подготовлено элементов: {len(items)}")
