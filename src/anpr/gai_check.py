# -*- coding: utf-8 -*-
"""
gai_check.py — фоновая проверка нового транспорта по базе ГАИ + сверка с налогом.

При появлении НОВОГО события транспорта pipeline ставит (rowid, номер, объект) в
очередь; этот поток шлёт POST на integration.gai_url и пишет в vehicle_events:
  gai_status: found      — ГАИ вернуло данные (HTTP 200, pResult == 1)
              not_found  — машины нет в базе ГАИ (HTTP 404/500 или pResult != 1)
              error      — сервис недоступен/таймаут (статус неизвестен)
  owner_type: тип владельца, уточнённый данными ГАИ (pOwnerType):
              shaxsiy (физлицо) | yuridik (юрлицо) | kompaniya (ИНН владельца
              совпадает с construction_inn генподрядчика объекта)
  owner_inn:  ИНН организации-владельца (pOrganizationInn, только юрлица)
  has_contract: сверка с налогом (integration.facturas_url): были ли счета-фактуры
              владелец ТС (продавец) -> заказчик/генподрядчик объекта (покупатели)
              за facturas_months. 1=есть, 0=нет, NULL=не проверялся/неприменимо
              (физлицо без ИНН, машина генподрядчика, сервис недоступен).

Отдельный поток, чтобы 12-секундный таймаут внешнего сервиса НЕ блокировал
GPU-инференс. Кэши по номеру и по (ИНН, объект) — не дёргать сервисы повторно.
"""
import json
import time
import queue
import calendar
import threading
import urllib.request
import urllib.error
from datetime import datetime

from anpr.plate_format import OWNER_SHAXSIY, OWNER_YURIDIK, OWNER_KOMPANIYA


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_plate(url: str, plate: str, timeout: float) -> tuple[str, dict | None]:
    """
    Один запрос в ГАИ. Возвращает (статус, данные ответа | None).
    Статус: found | not_found | error.
    """
    try:
        data = _post_json(url, {"plate_number": plate}, timeout)
        return ("found" if data.get("pResult") == 1 else "not_found"), data
    except urllib.error.HTTPError as e:
        # по договорённости: 404/500 от сервиса = машины нет в базе ГАИ
        return ("not_found" if e.code in (404, 500) else "error"), None
    except Exception:
        return "error", None                       # таймаут/сеть — статус неизвестен


def check_plate(url: str, plate: str, timeout: float) -> str:
    """Совместимость: только статус (используется backfill_gai_status.py)."""
    return fetch_plate(url, plate, timeout)[0]


def owner_from_gai(data: dict, construction_inn: str = "") -> tuple[str, str]:
    """
    (owner_type, owner_inn) из ответа ГАИ. pOwnerType: 1=юрлицо, 2=физлицо.
    Если ИНН владельца совпадает с construction_inn объекта -> kompaniya
    (машина принадлежит генподрядчику).
    """
    inn = str(data.get("pOrganizationInn") or "").strip()
    if inn and construction_inn and inn == str(construction_inn).strip():
        return OWNER_KOMPANIYA, inn
    ot = data.get("pOwnerType")
    if ot == 1:
        return OWNER_YURIDIK, inn
    if ot == 2:
        return OWNER_SHAXSIY, inn
    return "", inn


def contract_period(months: int) -> tuple[str, str]:
    """Период сверки (DD.MM.YYYY): months месяцев назад -> сегодня."""
    end = datetime.now()
    m = end.month - months
    y = end.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    start = end.replace(year=y, month=m, day=min(end.day, calendar.monthrange(y, m)[1]))
    return start.strftime("%d.%m.%Y"), end.strftime("%d.%m.%Y")


def check_contract(url: str, owner_inn: str, buyer_inns: list[str],
                   start_date: str, end_date: str, timeout: float):
    """
    Сверка с налогом: фактуры владелец ТС (продавец) -> каждый buyer_inn (покупатель).
    Возвращает (has_contract, facturas): 1 + список фактур, 0 + [] (ни одной),
    None + [] (все запросы упали — не знаем).
    """
    any_ok = False
    facturas = []
    for buyer in buyer_inns:
        if not buyer or not str(buyer).strip().isdigit():
            continue
        payload = {"buyer_inn": int(buyer), "seller_inn": int(owner_inn),
                   "start_date": start_date, "end_date": end_date}
        try:
            data = _post_json(url, payload, timeout)
        except Exception:
            continue                               # сервис упал по этому покупателю
        any_ok = True
        facturas.extend(data.get("facturas") or [])
    if not any_ok:
        return None, []
    return (1 if facturas else 0), facturas


class GaiChecker(threading.Thread):
    def __init__(self, url: str, timeout: float, vlog, cache_ttl: float = 3600.0,
                 objects: dict | None = None, facturas_url: str = "",
                 facturas_months: int = 3):
        """
        objects: object_id -> {"construction_inn": ..., "zakazchik_inn": ...}
                 (из cameras.yaml) — для kompaniya и сверки с налогом.
        facturas_url: пусто -> сверка с налогом выключена (только ГАИ).
        """
        super().__init__(daemon=True, name="gai-check")
        self.url = url
        self.timeout = float(timeout)
        self.vlog = vlog                          # VehicleLog (потокобезопасен: lock внутри)
        self.cache_ttl = float(cache_ttl)
        self.objects = objects or {}
        self.facturas_url = facturas_url
        self.facturas_months = int(facturas_months)
        self.q: "queue.Queue[tuple[int, str, str]]" = queue.Queue(maxsize=200)
        # plate -> (ts, status, data|None)
        self._cache: dict[str, tuple[float, str, dict | None]] = {}
        # (owner_inn, object_id) -> (ts, has_contract)
        self._contract_cache: dict[tuple[str, str], tuple[float, object]] = {}
        self.checked = 0                          # для статистики

    def enqueue(self, rowid: int, plate: str, object_id: str = ""):
        """Поставить событие в очередь проверки (не блокирует: полная очередь -> пропуск)."""
        try:
            self.q.put_nowait((rowid, plate, object_id))
        except queue.Full:
            pass

    def _fetch(self, plate: str) -> tuple[str, dict | None]:
        now = time.time()
        hit = self._cache.get(plate)
        if hit and now - hit[0] < self.cache_ttl and hit[1] != "error":
            return hit[1], hit[2]
        status, data = fetch_plate(self.url, plate, self.timeout)
        self._cache[plate] = (now, status, data)
        if len(self._cache) > 2000:                # не расти бесконечно
            cutoff = now - self.cache_ttl
            self._cache = {k: v for k, v in self._cache.items() if v[0] >= cutoff}
        return status, data

    def _contract(self, owner_inn: str, object_id: str):
        """(has_contract, facturas) по кэшу (ИНН, объект) или запросами к налоговой."""
        obj = self.objects.get(object_id) or {}
        buyers = [str(obj.get("zakazchik_inn") or ""), str(obj.get("construction_inn") or "")]
        if not any(b.strip().isdigit() for b in buyers):
            return None, []                        # у объекта нет ИНН — сверять не с кем
        key = (owner_inn, object_id)
        now = time.time()
        hit = self._contract_cache.get(key)
        if hit and now - hit[0] < self.cache_ttl and hit[1][0] is not None:
            return hit[1]
        start_s, end_s = contract_period(self.facturas_months)
        res = check_contract(self.facturas_url, owner_inn, buyers,
                             start_s, end_s, self.timeout)
        self._contract_cache[key] = (now, res)
        if len(self._contract_cache) > 2000:
            cutoff = now - self.cache_ttl
            self._contract_cache = {k: v for k, v in self._contract_cache.items()
                                    if v[0] >= cutoff}
        return res

    def _process(self, plate: str, object_id: str):
        """
        Полная проверка номера на объекте: ГАИ (статус + владелец + полный JSON в
        plate_info) и сверка с налогом (has_contract + фактуры). Обновляются ВСЕ
        события этого номера (важно для sweep-а по старым данным).
        """
        status, data = self._fetch(plate)
        try:
            self.vlog.set_gai_status_plate(plate, status)
            self.vlog.upsert_gai_info(plate, status, data)
            self.checked += 1
        except Exception:
            pass
        if status != "found" or not data:
            return
        # тип владельца по данным ГАИ (авторитетнее формата номера)
        constr_inn = str((self.objects.get(object_id) or {}).get("construction_inn") or "")
        owner_type, owner_inn = owner_from_gai(data, constr_inn)
        try:
            if owner_type or owner_inn:
                self.vlog.set_owner_plate(plate, object_id, owner_type, owner_inn)
        except Exception:
            pass
        # сверка с налогом: только юрлица с ИНН; машине генподрядчика договор не нужен
        if not self.facturas_url or not owner_inn.isdigit():
            return
        try:
            if owner_type == OWNER_KOMPANIYA:
                # фиксируем «неприменимо», чтобы sweep не перепроверял на каждом старте
                self.vlog.upsert_soliq_info(plate, object_id, {
                    "has_contract": None, "reason": "kompaniya", "facturas": []})
                return
            has_contract, facturas = self._contract(owner_inn, object_id)
            if has_contract is not None:
                self.vlog.set_contract_plate(plate, object_id, has_contract)
                self.vlog.upsert_soliq_info(plate, object_id, {
                    "has_contract": has_contract, "owner_inn": owner_inn,
                    "facturas": facturas})
        except Exception:
            pass

    def run(self):
        while True:
            _rowid, plate, object_id = self.q.get()
            self._process(plate, object_id)

    def sweep_old(self, delay: float = 0.5):
        """
        Разовый проход по СТАРЫМ событиям: ставит в очередь все (номер, объект),
        которым не хватает проверки ГАИ/soliq (см. VehicleLog.pending_checks).
        Запускать отдельным потоком после start(); дедуп сетевых запросов —
        кэшами по номеру и (ИНН, объект). delay бережёт внешние сервисы.
        """
        try:
            pending = self.vlog.pending_checks()
        except Exception:
            return
        if not pending:
            return
        print(f"[gai-sweep] непроверенных (номер, объект): {len(pending)}")
        for plate, object_id in pending:
            # object_id как есть (может быть NULL у древних событий — IS ? его матчит)
            self.q.put((0, plate, object_id))       # блокируется при полной очереди
            time.sleep(delay)
        print(f"[gai-sweep] очередь заполнена, проверка идёт в фоне")
