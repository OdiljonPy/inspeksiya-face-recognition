# Деплой на Ubuntu-сервер (NVIDIA Tesla T4)

Tesla T4 — Turing (sm_75), поддерживается стандартными сборками CUDA 12. Никаких
плясок с sm_120 (как на dev-машине RTX 5060) не нужно. torch на сервере не ставится —
всё работает на onnxruntime-gpu. Сервер headless: живого окна (`cv2.imshow`) нет,
только `main.py` (распознавание) и дашборд.

Предпосылка: **драйвер NVIDIA уже установлен** (`nvidia-smi` работает). CUDA Toolkit
ставить НЕ нужно — CUDA userspace приходит pip-колёсами.

Есть два способа: **Docker** (рекомендуется) и **bare-metal + systemd**.

---

## Проверка перед стартом
```bash
nvidia-smi          # должен показать Tesla T4 и версию драйвера
```

---

## Вариант A. Docker + docker compose (рекомендуется)

### 1. Docker + NVIDIA Container Toolkit (драйвер уже есть)
```bash
# Docker Engine
curl -fsSL https://get.docker.com | sh

# NVIDIA Container Toolkit (проброс GPU в контейнеры)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# проверка проброса GPU в контейнер:
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 2. Сборка и запуск
```bash
cd /opt/face-recognition           # где лежит проект
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d
```

### 3. Проверка GPU внутри образа
```bash
docker compose -f deploy/docker-compose.yml run --rm recognition gpucheck
# ожидаем: InsightFace РЕАЛЬНО использует CUDAExecutionProvider
```

### 4. Доступ и логи
- Дашборд: `http://<IP-сервера>:8000`
- Логи: `docker compose -f deploy/docker-compose.yml logs -f recognition`

Данные (галерея, кропы, `events.db`) лежат в `./data` на хосте (том), конфиг — в
`./config`. Правишь `config/cameras.yaml` на хосте и перезапускаешь `recognition`.

---

## Вариант B. Bare-metal + systemd

### 1. Установка
```bash
cd /opt/face-recognition
bash deploy/setup_ubuntu.sh        # venv, зависимости, предзагрузка моделей, проверка GPU
```

### 2. Автозапуск (systemd)
```bash
sudo cp deploy/systemd/face-recognition.service /etc/systemd/system/
sudo cp deploy/systemd/face-dashboard.service   /etc/systemd/system/
# отредактируй в обоих файлах User= и пути (WorkingDirectory, .venv), если не /opt/face-recognition и не ubuntu
sudo systemctl daemon-reload
sudo systemctl enable --now face-recognition face-dashboard
```

### 3. Статус и логи
```bash
systemctl status face-recognition face-dashboard
journalctl -u face-recognition -f
```

---

## Модели и блокировка сети
`deploy/preload_models.py` качает `buffalo_l` с **HF-зеркала** (не с GitHub — он в
регионе часто недоступен), а модели fast-alpr/RapidOCR — с CDN. В Docker модели
**вшиваются на этапе сборки** (нужен интернет при `build`). Для bare-metal их тянет
`setup_ubuntu.sh`. Если сборка/установка шли без сети — модели скачаются при первом
старте (если сеть будет). Кэш моделей: `~/.insightface`, `~/.cache/open-image-models`,
кэш fast-plate-ocr/rapidocr.

## Порт и фаервол
```bash
sudo ufw allow 8000/tcp     # если включён ufw и нужен доступ к дашборду извне
```
Дашборд слушает `0.0.0.0:8000` без авторизации — за пределами локальной сети закрой
его reverse-proxy (nginx + basic auth / VPN).

## Диагностика камеры (почему не распознаёт)
Живой просмотр детекций в браузере (сервер headless → отдаём MJPEG по HTTP).
Рисует боксы лиц и номеров с их размером в пикселях — сразу видно, мелко ли для
распознавания.
```bash
# по id из cameras.yaml (по умолчанию БЕЗ ресайза — нативное разрешение, det_size=1600):
python src/debug_stream.py --camera cam05 --port 8091
# прямой RTSP + подпись ID из галереи; det_size больше = ловит более мелкие лица:
python src/debug_stream.py --source "rtsp://user:pass@ip:554/..." --port 8091 --recognize --det-size 1600
# если нужен ресайз (быстрее): --width 1280
```
Дебаг-стрим НЕ жмёт кадр (в отличие от `main.py`, где ресайз до 960 ради 10 камер) и
поднимает `det_size` до 1600 — чтобы увидеть, ловятся ли лица на полном разрешении.

То же самое доступно прямо в дашборде: вкладка **«Камеры (live)»** — клик по камере
запускает живой просмотр с боксами (лениво, одна камера за раз, авто-стоп при уходе).
Для этого дашборд-процессу нужен GPU: в Docker у сервиса `dashboard` стоит `gpus: all`;
на bare-metal (systemd на хосте с GPU) — автоматически. Live-движок использует det_size 1280.
Открой `http://<IP-сервера>:8091` (открой порт: `sudo ufw allow 8091/tcp`).
В Docker: `docker compose -f deploy/docker-compose.yml run --rm -p 8091:8091 recognition \
  python src/debug_stream.py --camera cam05 --port 8091`.

Как читать: если лицо/номер **не в боксе или бокс оранжевый/крошечный** — объект
слишком мелкий для распознавания (далеко/угол/сильный даунскейл). Смотри HUD:
`src WxH -> ...` (реальное разрешение и до какого ресайзим) и размер бокса в px.
Ориентир: лицу нужно ~≥40–60 px по стороне, номеру ~≥90 px по ширине.

Типичная причина на объектах: камера 6 Мп, а кадр ресайзится до 960 px → далёкое
лицо становится ~10 px. Лечится увеличением `--width` (найди в дебаг-стриме порог,
при котором детект появляется) и затем `recognition.width` в `config/settings.yaml`
(или размещением/зумом камеры). Мейнстрим (subtype=0) вместо субпотока тоже помогает.

## Бэкап
Достаточно сохранить папку `data/` (в ней `events.db`, `gallery/`, `plates/`).

## Важно: два процесса и галерея
`recognition` держит галерею лиц в памяти и периодически сохраняет её на диск.
Удаление человека через дашборд во время работы `recognition` может быть перезаписано.
Удаляй людей из галереи как обслуживание: останови `recognition`
(`docker compose ... stop recognition` или `systemctl stop face-recognition`),
удали в дашборде, запусти снова. Удаление событий и номеров безопасно в любой момент.

## Обновление
- Docker: `git pull && docker compose -f deploy/docker-compose.yml up -d --build`
- systemd: `git pull && . .venv/bin/activate && pip install -r requirements-linux.txt && sudo systemctl restart face-recognition face-dashboard`
