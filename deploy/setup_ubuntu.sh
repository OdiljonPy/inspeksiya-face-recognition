#!/usr/bin/env bash
# =============================================================================
# Установка bare-metal на Ubuntu (Tesla T4). Драйвер NVIDIA должен быть уже
# установлен (проверяется nvidia-smi). CUDA Toolkit НЕ нужен — CUDA userspace
# приезжает pip-колёсами onnxruntime-gpu[cuda,cudnn].
#
# Запуск из корня проекта:  bash deploy/setup_ubuntu.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "Проект: $ROOT"

echo "== Проверка драйвера NVIDIA =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ОШИБКА: nvidia-smi не найден. Установи драйвер NVIDIA и повтори."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== Системные пакеты =="
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3.12 python3.12-venv python3.12-dev \
  ffmpeg libgl1 libglib2.0-0 libgomp1 ca-certificates

echo "== venv + зависимости =="
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-linux.txt

echo "== Предзагрузка моделей =="
python deploy/preload_models.py || echo "preload не удался — модели скачаются при первом старте"

echo "== Проверка GPU (onnxruntime CUDAExecutionProvider) =="
python src/check_gpu.py || true

cat <<EOF

=== Готово. Дальше: ===
1) Заполни config/cameras.yaml (RTSP-адреса, mode: face/plate/both).
2) Ручной запуск для проверки:
     . .venv/bin/activate
     python src/main.py                # распознавание
     python -m uvicorn web.app:app --host 0.0.0.0 --port 8000   # дашборд
3) Автозапуск через systemd — см. deploy/systemd/*.service и deploy/DEPLOY.md.
EOF
