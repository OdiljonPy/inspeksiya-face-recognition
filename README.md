# Распознавание людей по лицам для стройплощадки

Локальная система: детекция + распознавание лиц с 10 RTSP-камер, поиск по базе
известных людей, логирование событий и веб-дашборд. Всё работает offline на
Windows 11 + NVIDIA RTX 5060.

## Стек
- **InsightFace** (`buffalo_l`: SCRFD-детекция + ArcFace-эмбеддинги 512-d) — на GPU.
- **FAISS** (cosine / inner product) — поиск по базе.
- **OpenCV** (FFmpeg backend) — чтение RTSP.
- **SQLite** — события, снимки лиц — на диск.
- **FastAPI** — дашборд на localhost.

---

## Деплой на Linux-сервер (Tesla T4)
Продакшн-развёртывание на Ubuntu с NVIDIA Tesla T4 — см. **[deploy/DEPLOY.md](deploy/DEPLOY.md)**.
T4 (sm_75) поддерживается стандартным CUDA 12 без плясок sm_120; torch на сервере не нужен.
Готовы: `deploy/Dockerfile` + `deploy/docker-compose.yml` (Docker) и
`deploy/setup_ubuntu.sh` + `deploy/systemd/*.service` (bare-metal), плюс
`deploy/preload_models.py` (предзагрузка моделей с зеркал) и `requirements-linux.txt`.

## Установка на Windows 11 (dev-машина)

### Предусловия
- Python **3.14** (этот проект проверен на 3.14.0; подойдёт и 3.13).
- NVIDIA GPU с актуальным драйвером (проверено: 591.86, поддержка CUDA 13.1).
  Отдельный CUDA Toolkit ставить **не нужно** — рантайм приезжает pip-колёсами.

### 1. venv
```powershell
cd D:\GASN\face-recognition
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. PyTorch под sm_120 (Blackwell) — отдельный индекс cu128
```powershell
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 3. ONNX Runtime GPU + CUDA 12 runtime (extras с индекса NVIDIA)
```powershell
pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0" --extra-index-url https://pypi.nvidia.com
```
> Это ключевой шаг под sm_120. Обычный `onnxruntime-gpu` без CUDA-DLL падает с
> `cublasLt64_*.dll is missing` и откатывается на CPU. Версия 1.26.0 собрана под
> CUDA 12 и совпадает с CUDA 12.8 у torch — единый рантайм, ничего не конфликтует.

### 4. Остальные зависимости
```powershell
pip install -r requirements.txt
```

### 5. Модель buffalo_l
Скачивается автоматически при первом запуске в `%USERPROFILE%\.insightface\models\buffalo_l`.
Если GitHub-релизы недоступны (таймаут в регионе) — скачать зеркало вручную:
```powershell
$dest = "$env:USERPROFILE\.insightface\models"
mkdir $dest -Force
Invoke-WebRequest "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l.zip" -OutFile "$dest\buffalo_l.zip"
Expand-Archive "$dest\buffalo_l.zip" "$dest" -Force
mkdir "$dest\buffalo_l" -Force
Move-Item "$dest\*.onnx" "$dest\buffalo_l\" -Force
```

---

## Проверка GPU (Этап 0)
```powershell
python src\check_gpu.py
```
Скрипт печатает версии, доступность CUDA, providers onnxruntime и **реальный**
прогон InsightFace на GPU. Ожидаемый итог: `InsightFace РЕАЛЬНО использует CUDAExecutionProvider`.

### Эталонный вывод на RTX 5060
```
torch 2.11.0+cu128 | CUDA available | sm_120 в arch_list | matmul OK
onnxruntime-gpu 1.26.0 | providers: [Tensorrt, CUDA, CPU] | device: GPU
InsightFace buffalo_l: все 5 моделей -> CUDAExecutionProvider
steady-state детекция 960x960: ~9 мс/кадр (~111 кадров/сек на одном инстансе)
```

---

## Прогресс по этапам
- [x] **Этап 0** — окружение + `check_gpu.py` (GPU работает на sm_120).
- [x] **Этап 1** — чтение одного потока + детекция с боксами.
- [x] **Этап 2** — `enroll.py` + распознавание (Имя/Unknown).
- [x] **Этап 3** — 10 камер из `cameras.yaml`, общий пул моделей, reconnect/backoff.
- [x] **Этап 4** — авто-галерея лиц (уникальный ID на человека, снимок 1 раз, re-ID) + SQLite + анти-дребезг.
- [x] **Этап 5** — FastAPI дашборд (события + миниатюры + фильтр по камере + галерея ID).

## Модуль ANPR (автомобильные номера)
Отдельный модуль `src/anpr/` (fast-alpr: YOLO-детектор номера + OCR), работает на том
же GPU sm_120, что и лица (через `gpu_setup`). Каждой камере в `cameras.yaml` задаётся
`mode: face | plate | both` — модуль запускается только если есть камеры нужного режима.

**Узбекские номера (важно):** штатный OCR надёжно читает ТЕЛО номера
(буква+3цифры+2буквы), но систематически ошибается в маленьком левом боксе РЕГИОНА
(2 цифры): `01→CI`, `80→S`. Это проверено на реальных номерах. Текущий режим —
**интерим**: тело логируется надёжно, регион — best-effort с флагом `region_uncertain`,
дедуп идёт по ТЕЛУ номера. Надёжное решение региона — дообучение OCR на узбекских
номерах (см. [[anpr-uzbek-region-ocr]] в памяти проекта). Формат номера — regex
`anpr.plate_regex` в `config/settings.yaml` (регион 01–99 + латиница).

Регион для интерима читается через `rapidocr-onnxruntime` (PP-OCR модели PaddleOCR на
onnxruntime — `paddlepaddle` под Python 3.14 недоступен).

Таблица событий: `vehicle_events`, кропы номеров — `data/plates/`.

Запуск (камеры из `cameras.yaml` с их режимами):
```powershell
python src\main.py                                   # лица и/или ANPR по mode каждой камеры
python src\anpr\check_anpr.py --dir data\anpr_test   # пакетная проверка ANPR на фото
python src\anpr\stage2_anpr_stream.py --source "rtsp://..."   # ANPR на одном потоке
```
Дашборд: вкладка **«Транспорт»** — последние события, миниатюра номера, фильтр по
камере и поиск по номеру.

## Дашборд (Этап 5)
```powershell
# 1) в одном окне — запускаем распознавание (наполняет events.db и галерею):
python src\main.py
# 2) в другом окне — поднимаем дашборд:
.\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```
Открыть **http://127.0.0.1:8000**. Две вкладки:
- **События** — последние события с миниатюрами лиц, статус НОВЫЙ/замечен, фильтр по камере.
- **Галерея ID** — все уникальные люди (снимок, число событий и ракурсов).

Страница автообновляется каждые 3 с. Снимки берутся из `data/gallery/faces/`.

## Авто-галерея лиц (Этап 4)
Система не требует заранее заводить людей: она **сама** присваивает каждому новому
лицу уникальный ID (`person_0001`, …), сохраняет его снимок один раз в
`data/gallery/faces/` и узнаёт человека при повторной встрече (в т.ч. на другой
камере). На один ID хранится 1 снимок и до 5 эмбеддингов (разные ракурсы).
Параметры — в `config/settings.yaml` секция `gallery`. События пишутся в
`events.db` (SQLite) с анти-дребезгом (`events.dedup_seconds`).

### Стабилизация ID (против дублей при повороте головы)
Чтобы один человек не получал новый ID при повороте/смазе, решение принимается
не по одному кадру:
- **Трекинг по камере (IoU)** — лицо ведётся между кадрами и держит свой ID даже
  на плохих кадрах (`tracker.py`).
- **Гейт качества для нового ID** — новый человек заводится только с фронтального
  (оценка по 5 точкам), чёткого (резкость) и крупного лица. Профиль/смаз ID не плодят.
- **Двухпороговая логика** — `≥match_threshold` тот же ID; «серая зона»
  `[new_id_threshold, match_threshold)` → ближайший существующий ID (не новый);
  `<new_id_threshold` и хорошее качество → кандидат в новые (после подтверждения
  `new_id_confirm_frames` кадров).
- **Накопление ракурсов** — в закреплённый ID дописываются фронтальные эмбеддинги
  (до `max_embeddings_per_id`), дальше профиль матчится к тому же ID.

Все пороги — в `config/settings.yaml` секция `gallery`. Если всё ещё появляются
дубли — поднимите `new_id_threshold`/снизьте `match_threshold`; если реальный новый
человек долго не регистрируется — снизьте `new_id_min_frontality`.
