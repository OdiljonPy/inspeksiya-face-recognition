# Integration API (v1) — English Reference

REST API of the face recognition + ANPR system for external platforms.
Russian version (primary, more detailed): [API_INTEGRATION.md](API_INTEGRATION.md).

- Base URL: `http://<host>:8089` (production dashboard service).
- All responses are JSON; image URLs are absolute and cache-busted (`?v=`).
- Dates accepted as `unix ts | YYYY-MM-DD | YYYY-MM-DDTHH:MM:SS`;
  `date_to` given as a date is inclusive. Invalid date → `422`.
- Every listing supports `limit` + `offset` and returns `total`.
- Objects (construction sites) are identified by our `object_id` or by the
  external system's `object_index` (both filters accepted everywhere;
  unknown `object_index` → `404`, conflicting pair → `422`).

## Faces

### GET /api/v1/faces — face events
Filters: `object_id | object_index, camera_id, person, date_from, date_to,
exclude_uncertain=1, exclude_unidentified=1, limit, offset`.
Item: `id, ts, datetime, object_id/name/index, zakazchik_inn, construction_inn,
camera_id, zone, person, person_name` (full name for known workers), `score,
is_new, uncertain, q_*` (quality metrics), `face_url`, `full_url`.

**Full frames:** the full camera frame (`full_url`) is stored ONLY for the
event where a person appears for the FIRST time (`is_new=true`). Other face
events have an empty `full_url` (disk economy). Vehicle events are unaffected.

### GET /api/v1/persons — unique people over a period
Filters: `object_id | object_index, date_from, date_to, limit, offset`.
Item: `person, person_name, events, first_seen(_dt), last_seen(_dt), cameras,
face_url, angle_urls`.

### GET /api/v1/attendance — daily attendance
Per calendar day: how many unique people were on the site + the list of those
people with gallery photos.

Params: `object_id | object_index`, `date_from`, `date_to`
(default: last 7 days), `include_people` (default 1; `0` = counters only).
`Unknown/LOW_QUALITY` are not people — they are counted separately as
`unknown_events`.

```json
{
  "object_id": "obj_avloniy", "object_index": "41109", "total_days": 2,
  "items": [{
    "date": "2026-08-05", "people": 14, "unknown_events": 3,
    "persons": [{
      "person": "known_0002", "person_name": "ALIYEV VALI", "events": 9,
      "first_seen_dt": "2026-08-05 08:03:32",
      "last_seen_dt": "2026-08-05 17:06:52",
      "face_url": "http://<host>/faces/known_0002.jpg?v=...",
      "angle_urls": ["http://<host>/faces/known_0002_r2.jpg?v=..."]
    }]
  }]
}
```

### Angle photos (`angle_urls`)
When the system stores an additional embedding (angle) for a person, it now
also stores that angle's photo: `faces/<label>_r<N>.jpg` (JPEG 100).
`angle_urls` (ordered list of absolute URLs) is present in `/api/v1/persons`,
`/api/v1/attendance`, `GET /api/v1/known-faces` and in the `POST
/api/v1/known-faces` response. The main `face_url` is always the best shot;
angle photos are deleted together with the person.

## Known workers (enrollment by photo)

### POST /api/v1/known-faces
Create a worker from photo(s), or add angles to an existing one.

```json
{ "full_name": "ALIYEV VALI", "object_index": "41109",
  "images_base64": ["<jpeg/png base64>", "..."] }
```
- `image_base64` (single) and/or `images_base64` (2–5 recommended — improves
  recognition; the best-quality photo becomes the gallery photo, the rest are
  stored as angle templates with their photos).
- Exactly ONE face per photo, otherwise `422`.
- `label="known_XXXX"` instead of `full_name` → add angle(s) to existing worker.
- Response: `label, full_name, object_index, n_emb, photos_accepted,
  enrolled(_dt), face_url, angle_urls`.

**Strict mode** (`known_faces.strict_enroll: true` in settings.yaml, default
OFF): each photo must pass quality checks — face size ≥ `min_face_px` (112),
frontal pose, sharpness — otherwise `422` with a per-photo reason; creating a
worker whose face already exists in the gallery is rejected with `409`
(duplicate label + hint).

### GET /api/v1/known-faces
List of workers + appearance stats (`events`, `last_seen`), filter by
`object_index`, period `date_from/date_to` limits the stats counting.
Items include `face_url` and `angle_urls`.

### DELETE /api/v1/known-faces/{label}
Removes the worker (gallery + all their face events + photos).

## Vehicles (ANPR)

### GET /api/v1/vehicles — vehicle events
Filters: `object_id | object_index, camera_id, plate` (substring),
`valid, gai` (`found|not_found|error|unchecked`), `owner_type`
(`shaxsiy|yuridik|kompaniya`), `has_contract` (`0|1|2`), dates, paging.
`details=1` adds full GAI response + tax-check details per plate.
`has_contract` codes: `0` = no invoices / not in GAI / no INN, `1` = invoices
exist, `2` = vehicle belongs to the site's general contractor.

### GET /api/v1/vehicles/stats — unique vehicle counters
By owner type and contract status; same filters.

### GET /api/v1/vehicles/owner/{plate}
Owner info from GAI (cached 1h). With `object_id`/`object_index` also resolves
type `kompaniya` (owner INN == contractor INN). Degrades to plate-format-based
type if GAI is unreachable.

### GET /api/v1/tax-check
Invoices between the vehicle owner (`owner_inn`) and the site's customer /
general contractor over a period (`date_from/date_to`, default 3 months).
`plate=` also writes the result into the plate's events (`has_contract`).

## Deletion

`DELETE /api/v1/faces/{event_id}`, `/api/v1/persons/{label}`,
`/api/v1/vehicles/{event_id}`, `/api/v1/vehicles/plate/{plate}`.
Shared full frames are removed only when no events reference them; a person's
gallery photo is kept when deleting a single face event.

## Notes for integrators

- Send 2–5 photos per worker on enrollment — measured on 100 real people:
  recognition at threshold rises from 94.8% (1 photo) to 98.4% (3 photos).
- Photo requirements when strict mode is enabled: single face, ≥112 px face
  size, frontal, sharp. Portrait/badge photos work best.
- All image URLs must be fetched from the same host; they are stable but
  cache-busted via `?v=<mtime>` (the photo may be auto-upgraded to a sharper
  frame over time — best-shot).
