# API интеграции (v1)

REST API для внешних систем. Отдаёт события лиц и транспорта по объектам
(стройплощадкам) с фильтрами и фильтром по дате.

**Базовый URL:** `http://<IP-сервера>:8089` (порт дашборда `face-dashboard`).
Аутентификации нет — API доступен внутри сети; наружу выставлять через
reverse-proxy с авторизацией.

Все ответы — JSON вида:
```json
{ "total": 123, "limit": 100, "offset": 0, "items": [ ... ] }
```
`total` — всего записей под фильтром (для пагинации), `items` — страница.
Сортировка — новые сверху. URL снимков абсолютные, открываются с любого хоста.

## Формат дат (общий для всех эндпоинтов)

`date_from` / `date_to` принимают:
- unix timestamp: `1783657800`
- дату: `2026-07-12` (для `date_to` — включительно, весь день до 23:59:59)
- дату-время: `2026-07-12T14:30:00` (локальное время сервера)

Битый формат → `422` с описанием ошибки.

---

## API 1 — события лиц

```
GET /api/v1/faces
```

| Параметр | Описание |
|---|---|
| `object_id` | фильтр по объекту (`obj_avloniy`, `obj_102maktab`, ...) |
| `camera_id` | фильтр по камере (`cam03`, ...) |
| `person` | фильтр по ID человека (`person_0001`) |
| `date_from`, `date_to` | период (см. формат дат) |
| `exclude_uncertain` | `1` — убрать события «серой зоны» (ID присвоен с низкой уверенностью) |
| `exclude_unidentified` | `1` — убрать `Unknown` / `LOW_QUALITY` |
| `limit` | 1..1000, по умолчанию 100 |
| `offset` | смещение страницы, по умолчанию 0 |

Пример:
```
GET /api/v1/faces?object_id=obj_avloniy&date_from=2026-07-01&date_to=2026-07-12&exclude_uncertain=1&limit=100
```

Элемент `items`:
```json
{
  "id": 41,
  "ts": 1783657800.0,
  "datetime": "2026-07-10 09:30:00",
  "object_id": "obj_avloniy",
  "object_name": "Avloniy",
  "camera_id": "cam03",
  "zone": "Avloniy - 1",
  "person": "person_0001",
  "score": 0.72,
  "is_new": true,
  "uncertain": false,
  "face_url": "http://<host>/faces/person_0001.jpg?v=1783657800",
  "full_url": "http://<host>/full/1783657800000_cam03.jpg?v=1783657800"
}
```
`face_url` — снимок лица (один на ID), `full_url` — полный кадр события.

---

## API 1а — уникальные люди на объекте

Агрегация событий по человеку (кто был на объекте за период):

```
GET /api/v1/persons?object_id=obj_avloniy&date_from=2026-07-01&date_to=2026-07-12
```

Параметры: `object_id`, `date_from`, `date_to`, `limit`, `offset`.
`Unknown`/`LOW_QUALITY` исключены всегда.

Элемент `items`:
```json
{
  "person": "person_0001",
  "events": 17,
  "first_seen": 1783657800.0, "first_seen_dt": "2026-07-10 09:30:00",
  "last_seen": 1783830000.0,  "last_seen_dt": "2026-07-12 09:20:00",
  "cameras": ["cam03", "cam04"],
  "face_url": "http://<host>/faces/person_0001.jpg?v=..."
}
```

---

## API 2 — события транспорта

```
GET /api/v1/vehicles
```

| Параметр | Описание |
|---|---|
| `object_id` | фильтр по объекту |
| `camera_id` | фильтр по камере |
| `plate` | поиск по номеру (подстрока, регистр не важен) |
| `valid` | `1` — только валидные по формату РУз, `0` — только невалидные |
| `date_from`, `date_to` | период |
| `limit`, `offset` | пагинация |

Пример:
```
GET /api/v1/vehicles?object_id=obj_avloniy&date_from=2026-07-12&plate=772
```

Элемент `items`:
```json
{
  "id": 7,
  "ts": 1783657800.0,
  "datetime": "2026-07-10 09:30:00",
  "object_id": "obj_avloniy",
  "object_name": "Avloniy",
  "camera_id": "cam03",
  "zone": "Avloniy - 1",
  "plate": "01S772SB",
  "plate_raw": "CI S772SB",
  "region": "01",
  "body": "S772SB",
  "valid": true,
  "region_uncertain": false,
  "confidence": 0.84,
  "plate_url": "http://<host>/plates/1783657800000_cam03_01S772SB.jpg?v=...",
  "full_url": "http://<host>/full/1783657800000_cam03_veh.jpg?v=..."
}
```
`plate` — нормализованный номер (регион восстановлен, если удалось),
`plate_raw` — сырой текст OCR, `plate_url` — кроп номера, `full_url` — общее фото машины.
Если регион прочитать не удалось — `region_uncertain: true`, тело (`body`) при этом надёжно.

---

## Примеры интеграции

Python:
```python
import requests

BASE = "http://192.168.x.x:8089"

# все лица на объекте за сегодня
r = requests.get(f"{BASE}/api/v1/faces", params={
    "object_id": "obj_avloniy",
    "date_from": "2026-07-13",
    "exclude_unidentified": 1,
}).json()
for e in r["items"]:
    print(e["datetime"], e["person"], e["face_url"])

# постраничная выгрузка транспорта за период
offset = 0
while True:
    page = requests.get(f"{BASE}/api/v1/vehicles", params={
        "object_id": "obj_avloniy",
        "date_from": "2026-07-01", "date_to": "2026-07-12",
        "limit": 500, "offset": offset,
    }).json()
    for e in page["items"]:
        print(e["datetime"], e["plate"], e["full_url"])
    offset += 500
    if offset >= page["total"]:
        break
```

curl:
```bash
curl "http://<host>:8089/api/v1/persons?object_id=obj_avloniy&date_from=2026-07-01"
curl "http://<host>:8089/api/v1/vehicles?plate=S772&valid=1"
```

Интерактивная документация (Swagger UI): `http://<host>:8089/docs`
