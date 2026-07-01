# -*- coding: utf-8 -*-
"""
face_engine.py — единая обёртка над InsightFace (buffalo_l).

Используется на всех этапах:
  - Этап 1: только детекция (allowed_modules=['detection']) — быстрее.
  - Этап 2+: детекция + распознавание (эмбеддинг 512-d).

ВАЖНО: gpu_setup.enable_onnx_cuda() вызывается на импорте этого модуля,
ДО первого создания onnxruntime-сессии, иначе CUDA-DLL не найдутся.
"""
from gpu_setup import enable_onnx_cuda
enable_onnx_cuda()  # должно быть до импорта/использования onnxruntime/insightface

import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    """
    Тонкая обёртка над insightface.FaceAnalysis.

    Параметры
    ---------
    model_name : имя модели (buffalo_l).
    det_size   : размер входа детектора (квадрат), напр. (640, 640).
    ctx_id     : 0 -> GPU(0), -1 -> CPU.
    providers  : список Execution Providers onnxruntime.
    allowed_modules : какие подмодели грузить. None = все.
                      Для детекции достаточно ['detection'];
                      для распознавания ['detection', 'recognition'].
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        providers: list[str] | None = None,
        allowed_modules: list[str] | None = None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=allowed_modules,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Зафиксируем, реально ли используется GPU (для логов/диагностики)
        self.on_gpu = False
        for model in self.app.models.values():
            try:
                if "CUDAExecutionProvider" in model.session.get_providers():
                    self.on_gpu = True
                    break
            except Exception:
                pass

    def detect(self, frame_bgr: np.ndarray):
        """Вернуть список Face (bbox, det_score, kps, [embedding если загружен recognition])."""
        return self.app.get(frame_bgr)

    @staticmethod
    def warmup(engine: "FaceEngine", size: int = 640, rounds: int = 2) -> None:
        """Прогрев CUDA-ядер: первый вызов медленный (JIT/autotune)."""
        dummy = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        for _ in range(rounds):
            engine.detect(dummy)
