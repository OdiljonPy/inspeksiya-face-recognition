#!/usr/bin/env bash
# Точка входа образа: выбор сервиса по первому аргументу.
set -e

# CUDA из pip-колёс nvidia-*-cu12 -> onnxruntime-gpu должен их найти.
# gpu_setup их ещё и ctypes-предзагружает, но LD_LIBRARY_PATH — надёжная страховка.
NVLIBS=$(python - <<'PY'
import os, glob, sys
dirs = []
for s in sys.path:
    if s.endswith("site-packages"):
        dirs += glob.glob(os.path.join(s, "nvidia", "*", "lib"))
print(":".join(sorted(set(dirs))))
PY
)
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"

case "${1:-recognition}" in
  recognition)
    echo "[entrypoint] старт распознавания (лица + ANPR по режимам камер)"
    exec python src/main.py
    ;;
  dashboard)
    echo "[entrypoint] старт дашборда на 0.0.0.0:8000"
    exec python -m uvicorn web.app:app --host 0.0.0.0 --port 8090
    ;;
  preload)
    exec python deploy/preload_models.py
    ;;
  gpucheck)
    exec python src/check_gpu.py
    ;;
  *)
    # произвольная команда
    exec "$@"
    ;;
esac
