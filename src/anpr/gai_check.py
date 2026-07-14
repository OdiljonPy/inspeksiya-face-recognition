# -*- coding: utf-8 -*-
"""
gai_check.py — фоновая проверка нового транспорта по базе ГАИ.

При появлении НОВОГО события транспорта pipeline ставит (rowid, номер) в очередь;
этот поток шлёт POST на integration.gai_url и пишет статус в vehicle_events.gai_status:
  found      — ГАИ вернуло данные (HTTP 200, pResult == 1)
  not_found  — машины нет в базе ГАИ (HTTP 404/500 или pResult != 1)
  error      — сервис недоступен/таймаут (статус неизвестен)

Отдельный поток, чтобы 12-секундный таймаут внешнего сервиса НЕ блокировал
GPU-инференс. Кэш по номеру (TTL) — не дёргать сервис по одной машине повторно.
"""
import json
import time
import queue
import threading
import urllib.request
import urllib.error


def check_plate(url: str, plate: str, timeout: float) -> str:
    """Один запрос в ГАИ. Возвращает статус: found | not_found | error."""
    req = urllib.request.Request(
        url, data=json.dumps({"plate_number": plate}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return "found" if data.get("pResult") == 1 else "not_found"
    except urllib.error.HTTPError as e:
        # по договорённости: 404/500 от сервиса = машины нет в базе ГАИ
        return "not_found" if e.code in (404, 500) else "error"
    except Exception:
        return "error"                             # таймаут/сеть — статус неизвестен


class GaiChecker(threading.Thread):
    def __init__(self, url: str, timeout: float, vlog, cache_ttl: float = 3600.0):
        super().__init__(daemon=True, name="gai-check")
        self.url = url
        self.timeout = float(timeout)
        self.vlog = vlog                          # VehicleLog (потокобезопасен: lock внутри)
        self.cache_ttl = float(cache_ttl)
        self.q: "queue.Queue[tuple[int, str]]" = queue.Queue(maxsize=200)
        self._cache: dict[str, tuple[float, str]] = {}   # plate -> (ts, status)
        self.checked = 0                          # для статистики

    def enqueue(self, rowid: int, plate: str):
        """Поставить событие в очередь проверки (не блокирует: полная очередь -> пропуск)."""
        try:
            self.q.put_nowait((rowid, plate))
        except queue.Full:
            pass

    def _check(self, plate: str) -> str:
        return check_plate(self.url, plate, self.timeout)

    def run(self):
        while True:
            rowid, plate = self.q.get()
            now = time.time()
            hit = self._cache.get(plate)
            if hit and now - hit[0] < self.cache_ttl and hit[1] != "error":
                status = hit[1]
            else:
                status = self._check(plate)
                self._cache[plate] = (now, status)
                if len(self._cache) > 2000:        # не расти бесконечно
                    cutoff = now - self.cache_ttl
                    self._cache = {k: v for k, v in self._cache.items() if v[0] >= cutoff}
            try:
                self.vlog.set_gai_status(rowid, status)
                self.checked += 1
            except Exception:
                pass
