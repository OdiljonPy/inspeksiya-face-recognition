#!/usr/bin/env bash
# Точка входа образа: выбор сервиса по первому аргументу.
set -e

case "${1:-recognition}" in
  recognition)
    echo "[entrypoint] старт распознавания (лица + ANPR по режимам камер)"
    exec python src/main.py
    ;;
  dashboard)
    echo "[entrypoint] старт дашборда на 0.0.0.0:8000"
    exec python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
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
