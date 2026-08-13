# Карта пайплайна лиц: файл/функция → роль

Аудит по ТЗ (06.08.2026). Полная история решений — в HANDOFF.md (читать первым).

## Поток кадра (процесс main.py)

| Шаг | Файл / функция | Роль |
|---|---|---|
| Чтение RTSP | `src/camera_worker.py` | поток на камеру: frame-skip до target_fps, reconnect+backoff, TCP-префлайт (грабля №1) |
| Очередь | `src/main.py` (queue) | все камеры → один inference-поток |
| Маршрутизация | `src/inference_worker.py run()` | по mode камеры face/plate/both; ROI-кроп; per-camera det_size/width; maintenance-хук (GC) |
| Детекция лиц | `src/face_engine.py` → SCRFD | bbox + 5 kps + det_score; порог `recognition.min_det_score` |
| Эмбеддинг | тот же вызов `engine.detect` (insightface) → ArcFace | 512-d normed_embedding на каждое детектированное лицо |
| Трекинг + решение | `src/tracker.py CameraTracker.update/_decide` | IoU-связывание; ГЛАВНАЯ логика идентичности (см. docs/identity_pipeline.md) |
| Качество кадра | `src/gallery.py quality_ok_for_new / frontality / pose_score / blur_var`; `src/face_quality.py` (флаг, ВЫКЛ) | гейты создания ID; вес кадра для фото |
| Галерея | `src/gallery.py Gallery` | identify/identify2 (FAISS IndexFlatIP), add_new / add_new_from_track / add_known, best-shot (maybe_update_photo), provisional/GC, персистентность (embeddings.npy + owners.npy + meta.json) |
| События | `src/events.py EventLog.log` | SQLite events, анти-дребезг 30с, метрики качества, delete_person (GC) |
| Снимки | `src/main.py make_face_handler` | полный кадр → data/full, LOW_QUALITY кроп → data/lowq |

## Второй процесс (дашборд, web/app.py)

- отдаёт события/галерею/аналитику, live (MJPEG), v1-API для внешней платформы;
- enrollment известных: `POST /api/v1/known-faces` → `_decode_face` → `Gallery.add_known` (мульти-фото, strict-режим за флагом `known_faces.strict_enroll`);
- синхронизация процессов: файлы галереи + mtime meta.json (`Gallery.maybe_reload` в inference-цикле).

## Offline-инструменты (scripts/)

| Скрипт | Роль |
|---|---|
| `eval_quality.py` | регресс на тестовых клипах (лица 2 прохода + ANPR) |
| `eval_lfw100.py` | бенчмарк enrollment+матчинг на 100 реальных людях (LFW) |
| `eval_smallface.py` | бенчмарк мелких/деградированных лиц (сценарий обзорной камеры) |
| `dedup_gallery.py` | слияние дублей person_XXXX (dry-run HTML → --apply) |
| `tune_quality.py` | метрики качества по папке кадров (калибровка порогов) |
