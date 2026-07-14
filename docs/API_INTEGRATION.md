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
| `object_index` | фильтр по индексу объекта во внешней системе (`41109`, `38357`); неизвестный индекс → 404, конфликт с `object_id` → 422 |
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
  "object_index": 41109,
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

Параметры: `object_id`, `object_index`, `date_from`, `date_to`, `limit`, `offset`.
`Unknown`/`LOW_QUALITY` исключены всегда. Если задан `object_id`, в корне ответа
дополнительно `object_id` и `object_index` (индекс объекта во внешней системе).

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
| `object_index` | фильтр по индексу объекта во внешней системе |
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
  "object_index": 41109,
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

---

## Прокси к сервису ГАИ (для дашборда)

```
GET /api/gai/{plate}
```
Проксирует POST на `integration.gai_url` (settings.yaml) с телом `{"plate_number": "<plate>"}`
и возвращает ответ сервиса как есть. Ответы кэшируются по номеру
(`integration.gai_cache_seconds`, по умолчанию 1 час). Ошибки сервиса -> HTTP 502 с описанием.
Используется кнопкой «ГАИ» на вкладке Транспорт; можно дёргать и напрямую.

## Сверка с налогом (для дашборда)

```
GET /api/tax-check?owner_inn=<ИНН владельца>&object_id=<объект>
```
Два POST-запроса на `integration.facturas_url` (`get-facturas-by-inn`): покупатель —
ИНН заказчика и ИНН генподрядчика объекта (из cameras.yaml), продавец — владелец ТС,
период — `integration.facturas_months` (деф. 3) месяцев назад от сегодня.
Ответ: `{owner_inn, object_name, start_date, end_date, checks: [{role, buyer_inn,
facturas: [...] | error}]}`. Используется кнопкой «Сверка с налогом» в модалке ГАИ.

---

Интерактивная документация (Swagger UI): `http://<host>:8089/docs`
