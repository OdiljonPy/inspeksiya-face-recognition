# =============================================================================
# Образ распознавания лиц + ANPR для Ubuntu-сервера с NVIDIA Tesla T4 (sm_75).
# CUDA userspace приходит pip-колёсами (onnxruntime-gpu[cuda,cudnn]); драйвер —
# на хосте, пробрасывается через nvidia-container-toolkit (--gpus all).
# Базовый образ CUDA НЕ нужен — берём обычный python:3.12-slim.
# =============================================================================
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/root/.insightface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Системные зависимости: ffmpeg (RTSP), libGL/glib (opencv), libgomp (onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libgl1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости (кэш слоёв)
COPY requirements-linux.txt .
RUN pip install --upgrade pip && pip install -r requirements-linux.txt

# Код и конфиг
COPY src/   ./src/
COPY web/   ./web/
COPY config/ ./config/
COPY deploy/preload_models.py ./deploy/preload_models.py
COPY deploy/entrypoint.sh /entrypoint.sh
# убрать возможные CRLF (файл редактировался на Windows) и сделать исполняемым
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Вшиваем модели в образ (нужен интернет на этапе сборки: HF-зеркало + CDN).
# Если сборка без сети — строка не валит образ, модели скачаются при первом старте.
RUN python deploy/preload_models.py || echo "[build] preload пропущен — модели скачаются в рантайме"

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["recognition"]
