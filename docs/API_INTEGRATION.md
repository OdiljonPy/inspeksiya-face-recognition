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
  "q_det": null, "q_px": null, "q_blur": null, "q_yaw": null,
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
| `gai` | статус проверки по базе ГАИ: `found` / `not_found` (машины нет в базе) / `error` / `unchecked` |
| `owner_type` | тип владельца: `shaxsiy` (физлицо) / `yuridik` (юрлицо) / `kompaniya` (машина генподрядчика объекта) / `unknown` (не определён) |
| `has_contract` | сверка с налогом: `1` — фактуры с заказчиком/генподрядчиком есть, `0` — нет, `unchecked` — не проверялся |
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
  "gai_status": "found",
  "region_uncertain": false,
  "owner_type": "yuridik",
  "owner_inn": "301234567",
  "has_contract": true,
  "confidence": 0.84,
  "plate_url": "http://<host>/plates/1783657800000_cam03_01S772SB.jpg?v=...",
  "full_url": "http://<host>/full/1783657800000_cam03_veh.jpg?v=..."
}
```
`plate` — нормализованный номер (регион восстановлен, если удалось),
`plate_raw` — сырой текст OCR, `plate_url` — кроп номера, `full_url` — общее фото машины.
Если регион прочитать не удалось — `region_uncertain: true`, тело (`body`) при этом надёжно.

**`owner_type`** — тип владельца ТС:
- `shaxsiy` — физлицо; `yuridik` — юрлицо. Базово определяется по ФОРМАТУ номера
  (`01 A 123 BC` — физлицо, `01 123 ABC` — юрлицо), затем уточняется данными ГАИ
  (`pOwnerType`) при фоновой проверке нового события.
- `kompaniya` — машина принадлежит генподрядчику объекта: ИНН владельца из ГАИ
  совпадает с `construction_inn` объекта (cameras.yaml).
- `""` — не определён (номер не по формату РУз).

**`owner_inn`** — ИНН организации-владельца из ГАИ (пусто у физлиц).

**`has_contract`** — сверка с налогом (были ли счета-фактуры владелец ТС →
заказчик/генподрядчик объекта за `integration.facturas_months`):
`true` — фактуры есть, `false` — нет, `null` — не проверялся или неприменимо
(физлицо без ИНН, машина генподрядчика, сервис недоступен).

---

## API 2а — счётчики транспорта по типу владельца

```
GET /api/v1/vehicles/stats
```

Сколько УНИКАЛЬНЫХ машин (по номеру): юрлиц, физлиц, машин генподрядчика —
плюс разрез по сверке с налогом. Тип машины берётся из её последнего события
под фильтром.

Параметры: `object_id`, `object_index`, `camera_id`, `valid`, `date_from`, `date_to`.

```
GET /api/v1/vehicles/stats?object_id=obj_avloniy&date_from=2026-07-01
```
```json
{
  "vehicles_total": 42,
  "events_total": 310,
  "object_id": "obj_avloniy",
  "object_index": 41109,
  "by_owner_type": { "yuridik": 25, "shaxsiy": 12, "kompaniya": 3, "unknown": 2 },
  "by_contract": { "with": 18, "without": 7, "unchecked": 17 }
}
```
`by_owner_type` — число машин юрлиц / физлиц / генподрядчика / неопределённых.
`by_contract` — машины с фактурами / без / непроверенные (машины генподрядчика
не сверяются — попадают в `unchecked`).

---

## API 2б — владелец ТС по номеру (запрос в ГАИ)

```
GET /api/v1/vehicles/owner/{plate}
```

Параметры: `object_id` ИЛИ `object_index` (опционально — чтобы определить тип
`kompaniya` по ИНН генподрядчика этого объекта).

```
GET /api/v1/vehicles/owner/01123ABC?object_index=41109
```
```json
{
  "plate": "01123ABC",
  "found": true,
  "owner_type": "yuridik",
  "owner_type_source": "gai",
  "owner_inn": "301234567",
  "owner_name": "OOO QURILISH",
  "object_id": "obj_avloniy",
  "gai": { "pResult": 1, "pOwnerType": 1, "pOrganizationInn": 301234567, "...": "полный ответ ГАИ" }
}
```
`owner_type_source`: `gai` — тип из базы ГАИ; `plate_format` — машины нет в базе
(или сервис недоступен — тогда дополнительно поле `error`), тип определён по
формату номера. Ответы ГАИ кэшируются (`integration.gai_cache_seconds`).
Побочно обновляет `gai_status` / `owner_type` / `owner_inn` у всех событий номера.

---

## API 2в — сверка с налогом

```
GET /api/v1/tax-check?owner_inn=<ИНН>&object_id=<объект>[&plate=<номер>]
```

Параметры: `owner_inn` (обязателен), `object_id` ИЛИ `object_index` (обязателен
один из них), `date_from`/`date_to` (`YYYY-MM-DD` | `DD.MM.YYYY`; по умолчанию —
`facturas_months` месяцев назад от сегодня), `plate` (опционально — записать
результат в `has_contract` событий этого номера на объекте).

Ответ — как у `/api/tax-check` (см. ниже): `{owner_inn, owner_name, object_name,
has_contract, start_date, end_date, checks: [{role, buyer_inn, facturas | error}]}`.
`has_contract`: `true` — фактуры есть хотя бы с одним из ИНН объекта, `false` — нет,
`null` — все запросы к сервису упали.

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

## Удаление (DELETE)

```
DELETE /api/v1/vehicles/{event_id}        # одно событие транспорта (+ его кроп)
DELETE /api/v1/vehicles/plate/{plate}     # ВСЕ события номера (+ кропы и полные кадры)
DELETE /api/v1/faces/{event_id}           # одно событие лица (фото галереи НЕ трогается)
DELETE /api/v1/persons/{label}            # человек из галереи + ВСЕ его события
```
Полные кадры, на которые ссылаются другие события (несколько лиц/номеров в одном
кадре), не удаляются, пока жива хоть одна ссылка. Несуществующий id/номер → 404.

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
GET /api/tax-check?owner_inn=<ИНН владельца>&object_id=<объект>[&plate=<номер>]
```
Два POST-запроса на `integration.facturas_url` (`get-facturas-by-inn`): покупатель —
ИНН заказчика и ИНН генподрядчика объекта (из cameras.yaml), продавец — владелец ТС,
период — `integration.facturas_months` (деф. 3) месяцев назад от сегодня.
Ответ: `{owner_inn, object_name, has_contract, start_date, end_date, checks:
[{role, buyer_inn, facturas: [...] | error}]}`. Если задан `plate`, результат
записывается в `has_contract` всех событий этого номера на этом объекте.
Используется кнопкой «Сверка с налогом» в модалке ГАИ.

---

Интерактивная документация (Swagger UI): `http://<host>:8089/docs`
