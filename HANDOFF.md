# HANDOFF — передача проекта в новую сессию Claude Code

Кратко для новой сессии: что это за проект, как устроен, какие решения приняты,
где грабли, и в каком состоянии всё сейчас. Читать целиком перед изменениями.

---

## 1. Что за проект
Локальная (offline) система для стройплощадок: **распознавание людей по лицам + ANPR
(автомобильные номера)** с RTSP-камер. Регион — **Узбекистан**. Есть веб-дашборд.

Пользователь работает по-русски, поэтапно, после каждого этапа ждёт подтверждения.
Комментарии в коде — на русском.

---

## 2. Окружение (КРИТИЧНО — не переустанавливать вслепую)

**Dev-машина (здесь):** Windows 11, GPU **RTX 5060 (Blackwell, sm_120)**, Python **3.14**, venv в `.venv`.
**Прод-сервер:** Ubuntu, GPU **Tesla T4 (Turing, sm_75)**, Python 3.12, проект в
`/root/inspeksiya-face-recognition`, запуск через **systemd** (`face-recognition`,
`face-dashboard` на порту **8089**), `User=root`.

**GPU-решение (уже работает, не ломать):**
- Всё распознавание — на **onnxruntime-gpu** (InsightFace + fast-alpr). torch нужен ТОЛЬКО
  на dev для проверки sm_120; на сервере torch НЕ нужен.
- `src/gpu_setup.py` кроссплатформенный: Windows — `os.add_dll_directory` на CUDA-DLL из
  pip-колёс; Linux — `ctypes`-preload `.so` (в несколько проходов). Вызывать
  `enable_onnx_cuda()` ДО импорта onnxruntime.
- Dev: torch `2.11.0+cu128` (даёт CUDA-DLL) + `onnxruntime-gpu 1.26.0` + `nvidia-*-cu12`.
  Также на dev: `ultralytics` + `huggingface_hub` — ТОЛЬКО для экспорта yolov8n.onnx
  (person-детектор); на сервере они не нужны.
- Сервер (T4): `onnxruntime-gpu==1.20.1` + **явно** `nvidia-cublas/cudnn/cufft/curand/
  cuda-runtime/cuda-nvrtc-cu12` (extras `[cuda,cudnn]` НЕ подтянули колёса — отсюда была
  ошибка `libcublasLt.so.12 missing`). См. `requirements-linux.txt`.
- Проверка GPU: `python src/check_gpu.py` — должен показать `CUDAExecutionProvider` и
  «InsightFace РЕАЛЬНО использует CUDA».

**Сеть:** GitHub-ассеты в регионе часто блокируются. Поэтому модель `buffalo_l` качается
с **HF-зеркала** `public-data/insightface` (см. `deploy/preload_models.py`). Модели
fast-alpr/rapidocr идут с CDN (работают).

---

## 3. Стек
- Лица: **InsightFace buffalo_l** (SCRFD-детекция + ArcFace 512-d).
- Поиск: **FAISS** `IndexFlatIP` на L2-норм. векторах (= cosine).
- Номера: **fast-alpr** (YOLO-v9 детектор + OCR `global-plates-mobile-vit-v2`),
  + **rapidocr-onnxruntime** (PP-OCR) для попыток чтения региона.
- RTSP: OpenCV/FFmpeg (TCP). SQLite. FastAPI + один HTML (`web/templates/index.html`).

---

## 4. Архитектура и файлы

```
src/
  gpu_setup.py          кроссплатформенная подготовка CUDA для onnxruntime
  config.py             load_settings/load_cameras/load_objects (+ DEFAULT_OBJECT_ID)
  face_engine.py        обёртка InsightFace (детекция/распознавание), пул по det_size
  gallery.py            авто-галерея person_XXXX: identify/add_new/maybe_add_embedding,
                        delete_identity, maybe_reload (mtime), quality_ok_for_new, frontality/blur
  tracker.py            CameraTracker (IoU-трекинг, стабилизация ID, гейт face_quality)
  faiss_index.py        build/load/search (legacy именованный режим)
  recognizer.py, enroll.py   LEGACY (именованные люди) — main.py их НЕ использует
  events.py             EventLog (SQLite events + анти-дребезг + метрики + object_id)
  camera_worker.py      поток чтения RTSP (frame-skip, reconnect+backoff, TCP-префлайт)
  inference_worker.py   ЕДИНЫЙ поток инференса: маршрут по mode (face/plate/both),
                        пул движков по det_size, per-camera width/scale
  results.py            FaceResult/FrameResult (датаклассы; object_id, метрики качества)
  main.py               оркестратор: камеры+объекты, движки, логи, статистика
  face_quality.py       фильтр качества лица (Задача 1) — СЕЙЧАС ВЫКЛЮЧЕН в конфиге
  tune_quality.py       прогон по папке снимков -> метрики (для подбора порогов)
  objects_db.py         таблица objects + sync из cameras.yaml
  draw_overlay.py       отрисовка боксов (общая для debug_stream и live)
  debug_stream.py       MJPEG-дебаг одной камеры с боксами (нативное разрешение, det_size)
  check_gpu.py, stage1_single_stream.py, stage2_recognize.py   диагностика/этапы
  anpr/
    engine.py           AnprEngine (fast-alpr, CUDA-провайдеры)
    plate_format.py     PlateValidator (regex РУз, разбор регион/тело, region_uncertain)
    vehicle_log.py      VehicleLog (SQLite vehicle_events + дедуп по ТЕЛУ номера + object_id)
    pipeline.py         process_frame (детект+разбор+кроп+лог)
    check_anpr.py, stage2_anpr_stream.py, stage3_test.py
web/
  app.py                FastAPI: /api/{objects,cameras,events,vehicle_events,gallery,
                        analytics,person}, /live/*, DELETE, статика /faces /plates /full /lowq
  live.py               LiveManager (мульти-камера live с боксами, ленивые движки, авто-стоп)
  templates/index.html  весь дашборд (вкладки: События, Галерея ID, Транспорт,
                        Камеры(live), Аналитика + карточка человека)
config/
  settings.yaml         пороги, пути, anpr, face_quality, gallery, events
  cameras.yaml          objects: [...] + cameras: [{id, object_id, zone, rtsp, mode, det_size, fps}]
scripts/
  migrate_quality_columns.py   миграция: колонки метрик качества в events
  migrate_objects.py           миграция: таблица objects + object_id + backfill + индексы
deploy/
  Dockerfile, docker-compose.yml, entrypoint.sh, setup_ubuntu.sh, preload_models.py,
  systemd/{face-recognition,face-dashboard}.service, DEPLOY.md
requirements.txt (dev, sm_120), requirements-linux.txt (сервер, T4), README.md
```

**Два процесса** (важно): `main.py` (распознавание, пишет) и `web/app.py` (дашборд,
читает/удаляет) — РАЗНЫЕ процессы, общаются через файлы галереи (`data/gallery/`) и
`data/events.db`. Дашборду при live нужен GPU.

---

## 5. Ключевые решения и ГРАБЛИ (частые причины багов)
1. **OpenCV FFmpeg глобальный мьютекс открытия**: зависшее `open()` мёртвой RTSP-камеры
   блокирует ОТКРЫТИЕ других камер. Решение: свой **TCP-префлайт** (`tcp_reachable`) в
   `camera_worker.py` перед `cv2.VideoCapture`.
2. **det_size — реальный потолок разрешения**, а не ресайз до 960. Детектор вписывает кадр
   в `det_size` (деф. 640). Дальние мелкие лица теряются. Per-camera `det_size` в cameras.yaml.
   В `main.py` кадр НЕ ресайзится до 960 (это только в debug_stream/stage-скриптах).
3. **Гонка двух процессов + галерея**: удаление человека из дашборда воскрешалось
   распознаванием. Решение: `gallery.maybe_reload()` (по mtime meta.json) в inference-цикле.
4. **Кэш браузера на миниатюрах**: файл `person_XXXX.jpg` менялся, URL тот же → старое фото.
   Решение: cache-busting `?v=<mtime>` + no-cache заголовки (`web/app.py _versioned`).
5. **LOW_QUALITY снимки** лежат в `data/lowq`, а не в `data/gallery/faces` — `_face_url`
   маршрутизирует по папке (`/lowq` vs `/faces`).
6. **uvicorn `web.app:app`** требует корень проекта в sys.path → в systemd добавлен
   `--app-dir /root/inspeksiya-face-recognition`. И `User=root` (проект в /root).
7. **CUDNN failure на dev** случался из-за зависших python-процессов + отсутствия прогрева.
   Движки прогреваются (`FaceEngine.warmup`), инференс в try/except.
8. **Дубликат `id: cam06`** в cameras.yaml ломал статистику/трекер — переименован в cam07.
9. **onnxruntime + rapidocr** делят пакет `onnxruntime`: после `pip install rapidocr*`
   снести CPU-`onnxruntime` и `--force-reinstall --no-deps onnxruntime-gpu`.

---

## 6. Функционал (что готово)
- Распознавание лиц: авто-галерея `person_XXXX` (open-set). `match_threshold=0.5` (тот же ID),
  `new_id_threshold=0.3` (новый ID если score<0.3, иначе серая зона → ближайший).
  Трекер держит ID между кадрами. Гейты создания нового ID (det/px/frontality/blur/confirm),
  профиль МЯГКИЙ (0.45/30/0.4/30/3) — максимальное покрытие.
  **ВАЖНО (13.07.2026): улучшения ЛОГИКИ ЛИЦ откачены по запросу пользователя**
  («system is not working correctly») — tracker.py/gallery.py = коммит 4122686
  (+ защитная загрузка meta.json: незнакомые поля отбрасываются, не падаем).
  Откачено ТОЛЬКО по лицам: own_score-гейт ракурсов, подтверждение матча по 2 кадрам,
  адаптивный det-порог, best-shot фото. ANPR/API/дашборд — актуальные (см. ниже).
  Флаг uncertain в events/API теперь всегда 0 (колонка осталась). Следствия мягкого
  профиля: возможны мусорные ID на текстурах, фото галереи фиксируется первым кадром.
  Новые доработки лиц вносить ПО ОДНОМУ с проверкой на живой камере пользователя.
- ANPR: **тело** номера читается надёжно; регион восстанавливается ДВУМЯ механизмами:
  1) `fix_region` (plate_format.py) — позиционная коррекция букв→цифр (CI→01, S0→50);
  2) `region_ocr.py` (rapidocr, только для событий с region_uncertain) — кроп левой части
     номера, перебор долей 0.25/0.32/0.40, апскейл ×4, text_score=0.3, два варианта
     препроцессинга (как есть + CLAHE+Otsu — тёмные номера иначе не читаются).
  Регион валиден ТОЛЬКО из списка действующих кодов РУз `VALID_REGIONS`
  (plate_format.py: 01,10,20,25,30,40,50,60,70,75,80,85,90,95) — синхронизирован с
  plate_regex в settings.yaml (менять в обоих местах!). Ложный регион (напр. OCR
  прочитал 61 вместо 01 из-за болта на рамке) отсекается списком и чинится вторым проходом.
  Форматы РУз — РОВНО ДВА нужных типа (требование пользователя 14.07.2026):
  юрлицо «01 123 ABC» (регион+3цифры+3буквы) и физлицо «01 A 123 BC»
  (регион+буква+3цифры+2буквы). Прицепный формат (4цифры+2буквы) УБРАН.
  На тестовых данных (data/anpr_test + _cam_plate.mp4) 7/7 валидных (прицеп
  отфильтрован по формату). В дашборде на вкладке Транспорт фильтр по статусу:
  OK (дефолт) / Невалидные / Все. Кнопка «✎» у номера — исправить текст номера
  (PATCH /api/vehicle_event/{id}/plate: нормализация + перевалидация флагов).
  Дедуп по телу; чтение с БОЛЬШЕЙ conf в
  окне дедупа ОБНОВЛЯЕТ запись (голосование). Полный кадр события — в data/full (*_veh.jpg,
  колонка full_path; показывается в дашборде на вкладке Транспорт, колонка «Фото»).
  Номера уже 60px не логируются (`min_plate_px`). `require_valid_body: true` — события
  пишутся ТОЛЬКО если тело соответствует формату РУз (фары/решётки давали OCR-мусор
  с conf>=0.5 и попадали в базу как «номера»).
  vehicle_events + кропы в `data/plates/`.
- Режимы камер `face|plate|both`, 10 камер, один пул GPU-движков.
- SQLite: `events`, `vehicle_events`, `objects`. Снимки: `data/gallery/faces` (по ID),
  `data/full` (полный кадр события), `data/plates`, `data/lowq`.
- **Объекты (стройплощадки)**: `objects` в cameras.yaml + `object_id` в событиях.
  Фильтр по объекту на всех страницах дашборда; камеры показываются только выбранного объекта.
- **Аналитика**: уникальные люди по объектам за сутки/неделя/месяц; «Неопознанные»
  (Unknown/LOW_QUALITY) отдельно. **Карточка человека**: объекты, всего, график по дням, снимки.
- **Дашборд**: вкладки События / Галерея ID / Транспорт / Камеры (live) / Аналитика.
  Live — MJPEG с боксами по клику, мозаика, выбор det_size, авто-стоп. Удаление людей/номеров.
- **Задача 1 (фильтр качества)** реализована, но **ВЫКЛЮЧЕНА** (`face_quality.enabled: false`).
- **Person-first (архитектура «сначала человек, потом лицо») — ЭТАП 1 из 3 (14.07.2026)**:
  `src/person_engine.py` — детектор человека YOLOv8n на onnxruntime-gpu (~8ms/кадр,
  без torch в рантайме). Модель `data/models/yolov8n.onnx` идёт ЧЕРЕЗ GIT (экспорт
  требует torch/ultralytics — есть только на dev; исходник .pt — HF-зеркало
  Ultralytics/YOLOv8, НЕ github). Этап 1: боксы людей ТОЛЬКО в debug_stream
  (`--no-persons` — выключить), main.py НЕ тронут. План (по одному, после проверки
  пользователем на живой камере): этап 2 — лицо ищем внутри бокса человека, трек по
  телу; этап 3 — сравнение с базой один раз на трек по лучшему (чёткому) кадру лица,
  без чёткого лица — событие «неопознанное присутствие». Диагноз, который к этому
  привёл: bodycam 848x480 субпоток + компрессия -> cosine одного человека между
  кадрами 0.19..0.86 (перекрывается с «чужими» 0..0.2) — пороги не разделяют;
  главный рычаг — основной RTSP-поток/битрейт камеры, person-first сокращает ущерб.
- **Автопроверка транспорта по базе ГАИ (14.07.2026)**: НОВОЕ событие транспорта сразу
  проверяется по gai_url в фоновом потоке (`src/anpr/gai_check.py`, запускается в main.py,
  НЕ блокирует инференс; кэш по номеру 1ч). Статус в `vehicle_events.gai_status`:
  found | not_found (HTTP 404/500 от сервиса ИЛИ pResult!=1 = «машины нет в базе ГАИ»)
  | error (сервис недоступен) | NULL (не проверялся, старые записи). В дашборде: красный
  бейдж «нет в ГАИ» в колонке Статус + отдельный фильтр «ГАИ» (все/есть/нет/не проверен/
  ошибка); фильтр `gai=` есть и в /api/vehicle_events, и в /api/v1/vehicles. Ручной клик
  «ГАИ» в модалке тоже обновляет статус всех событий номера. Выключатель:
  integration.gai_check_on_new.
- **Интеграция ГАИ (14.07.2026)**: кнопка «ГАИ» в таблице Транспорт -> модалка с данными
  владельца ТС. Дашборд проксирует запрос через `GET /api/gai/{plate}` (web/app.py) ->
  POST на `integration.gai_url` из settings.yaml (сервис доступен только из сети СЕРВЕРА,
  с dev не проверить — тестировано мок-сервером). Кэш ответов по номеру
  (`gai_cache_seconds`, деф. 1ч). Ошибки сервиса показываются в модалке текстом.
- **Сверка с налогом (14.07.2026)**: кнопка «⚖ Сверка с налогом» ВНУТРИ модалки ГАИ
  (видна только если у владельца есть ИНН организации и у события задан объект).
  `GET /api/tax-check?owner_inn&object_id` шлёт ДВА запроса на `integration.facturas_url`
  (get-facturas-by-inn): buyer = ИНН заказчика объекта и buyer = ИНН генподрядчика,
  seller = ИНН владельца ТС, период `facturas_months` (деф. 3 мес) назад от сегодня.
  ИНН-ы объектов — в cameras.yaml (objects[].zakazchik_inn / construction_inn /
  object_index). Названия компаний НЕ хранятся в конфиге — берутся из ответа
  налоговой (buyerName/sellerName первой фактуры); если фактур нет — только ИНН.
  В модалке: поля дат «с/по» (деф. 3 мес назад, /api/tax-check принимает
  date_from/date_to), названия организаций (владелец — в шапке), таблица фактур
  с колонкой «Услуга» (catalogName). «Фактур не найдено» = владелец ТС не связан
  с объектом финансово. Если ИНН владельца ТС СОВПАДАЕТ с construction_inn объекта —
  вверху модалки ГАИ яркий зелёный баннер «МАШИНА ПРИНАДЛЕЖИТ ГЕНПОДРЯДЧИКУ»
  (проверка на фронте: /api/objects отдаёт ИНН-ы объектов из cameras.yaml).
- **API интеграции (v1)** для внешней системы (web/app.py; полная дока — docs/API_INTEGRATION.md):
  `GET /api/v1/faces` (события лиц), `GET /api/v1/persons` (уникальные люди, агрегация),
  `GET /api/v1/vehicles` (транспорт). Фильтры: object_id, camera_id, person/plate/valid,
  date_from/date_to (unix ts | YYYY-MM-DD | YYYY-MM-DDTHH:MM:SS; date_to по дате — включительно),
  limit/offset + total. URL снимков абсолютные. Битая дата -> 422.
- **ROI per-camera**: `roi: [x1,y1,x2,y2]` в cameras.yaml — кроп зоны обработки (лица+ANPR),
  экономит GPU и даёт мелким лицам больше пикселей детектора (см. inference_worker._apply_roi).

---

## 7. ТЕКУЩЕЕ состояние / открытая проблема (последнее, над чем работали)
**Проблема пользователя:** камеры на объектах — обзорные, лица **30–45px**. На таких лицах
эмбеддинг ненадёжный → либо ничего не распознаётся (когда фильтр строгий), либо
**сотни мусорных/дубль ID** (когда мягко).

**Что решили на данный момент:**
- `face_quality.enabled: false` (фильтр LOW_QUALITY мешал больше, чем помогал).
- Ужесточили гейты создания нового ID: `new_id_min_det_score 0.70`, `new_id_min_face_px 55`,
  `new_id_min_frontality 0.55`, `new_id_min_blur 60`, `new_id_confirm_frames 5`.
- Рекомендовано пользователю: **сбросить галерею** (в ней ~400 мусорных ID, они портят
  матчинг) и **поставить камеру у прохода/входа** (лицо ≥80–100px) для реального ID;
  обзорные камеры лучше в `mode: plate` (ANPR/присутствие).

**ФУНДАМЕНТАЛЬНОЕ ограничение (объяснено пользователю):** надёжного распознавания лиц на
30–45px нет в принципе — это геометрия, не баг. Рычаг баланса «покрытие ↔ мусор» —
`new_id_min_face_px`.

---

## 8. Полезные команды

**Dev (Windows):** venv в `.venv`, запуск `python src\main.py`, дашборд
`python -m uvicorn web.app:app --host 127.0.0.1 --port 8000`.
Тест-клипы: `data/_cam_*.mp4`. Тест-фото номеров: `data/anpr_test/`.
UTF-8 в консоли: `$env:PYTHONUTF8=1; chcp 65001`.

**Сервер (Ubuntu, T4):**
```bash
cd /root/inspeksiya-face-recognition && git pull
python scripts/migrate_quality_columns.py
python scripts/migrate_objects.py
python scripts/fix_object_ids.py          # разово: старые события avloniy -> obj_avloniy
python scripts/revalidate_plates.py       # перефлаговать номера по текущим правилам
                                          # (--purge-invalid — удалить мусорные события)
sudo systemctl restart face-recognition face-dashboard
journalctl -u face-recognition -f
```
Новые колонки (events.uncertain, vehicle_events.full_path) добавляются идемпотентно
при старте main.py/дашборда — отдельная миграция не нужна.
Для region_ocr на сервере нужен rapidocr-onnxruntime (см. грабли №9: после его установки
переставить onnxruntime-gpu). Если rapidocr нет — модуль сам выключится, всё работает.
Дашборд: http://<IP>:8089

**Сброс галереи (когда накопился мусор):**
```bash
sudo systemctl stop face-recognition
rm -rf data/gallery
sqlite3 data/events.db "DELETE FROM events;"
sudo systemctl start face-recognition face-dashboard
```

**Диагностика камеры (видит ли лица, какого размера):**
```bash
python src/debug_stream.py --camera <id> --port 8091   # смотреть в браузере, HUD с px/det
python src/tune_quality.py --dir data/lowq             # метрики отбракованных лиц
```

**Оценка качества на тестовых клипах (запускать после изменений порогов/логики):**
```bash
python scripts/eval_quality.py            # лица (2 прохода: создание ID + стабильность) и ANPR
```
Эталон (июль 2026): _cam_known -> ровно 1 ID, проход 2 без новых ID; ANPR 8/8 valid,
region_uncertain 0/8. Прод-галерею/БД скрипт не трогает (data/_eval/).

---

## 9. Что можно делать дальше (идеи, не начато)
- Привязка человеческих имён к `person_XXXX` (переименование в дашборде).
- Экспорт аналитики в CSV/Excel.
- Перевод обзорных камер в `mode: plate`, лица только на входных.
- Второй inference-поток / TensorRT для throughput на T4 (тонкость: трекер per-camera —
  нужна привязка камер к потокам).
- A/B модели лиц: antelopev2 (glintr100) вместо buffalo_l — точнее на сложных ракурсах;
  переключается конфигом `recognition.model_name`, нужен прогон на реальных клипах.
- Дообучение OCR под узбекский регион-бокс — если fix_region+region_ocr не хватит
  (нужен размеченный датасет).
- Показ uncertain-событий и полного фото транспорта в дашборде (в API уже есть:
  events.uncertain, vehicle_events.full_url).
