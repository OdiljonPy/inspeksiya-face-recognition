# -*- coding: utf-8 -*-
"""
Тест enrollment+матчинга на 100 РЕАЛЬНЫХ лицах (LFW multifaces, HF-зеркало).

Схема:
  - 100 человек заводятся в свежую галерею (изолированную от прода):
      G1 — по 1 фото на человека (как раньше);
      G3 — по 3 фото на человека (новый мульти-фото enrollment);
  - проверка: по 2 НОВЫХ фото каждого заведённого (genuine) + по 1 фото
    100 НЕзаведённых людей (impostors);
  - метрики: rank-1, попадание с порогом match_threshold=0.5, false accepts,
    распределения score, латентность identify().
Прод data/gallery и events.db НЕ трогаются. Датасет качается с HF-зеркала
(vilsonrodrigues/lfw, ~180MB в кэш HF) — github в регионе блокируется, HF работает.

Запуск: python scripts\eval_lfw100.py   (на dev при нехватке памяти:
OPENBLAS_NUM_THREADS=1)
Эталон (06.08.2026, buffalo_l, det 640, порог 0.5):
  1 фото:  rank-1 100%, узнан с порогом 94.8%, FAR 0/100
  3 фото:  rank-1 100%, узнан с порогом 98.4%, FAR 0/100 (score своих p10 0.56->0.63)
"""
import os
import re
import sys
import time
import shutil
import zipfile
import collections

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import cv2
import numpy as np

from config import load_settings
from face_engine import FaceEngine
from gallery import Gallery

N_PEOPLE = 100
N_IMPOSTORS = 100


def _dataset_zips():
    """Скачать (в кэш HF) и вернуть пути к zip'ам ingestion/retrieval."""
    from huggingface_hub import hf_hub_download
    ing = hf_hub_download("vilsonrodrigues/lfw", "lfw_multifaces-ingestion.zip",
                          repo_type="dataset")
    ret = hf_hub_download("vilsonrodrigues/lfw", "lfw_multifaces-retrieval.zip",
                          repo_type="dataset")
    return ing, ret


def people_map(zpath):
    z = zipfile.ZipFile(zpath)
    m = collections.defaultdict(list)
    for n in z.namelist():
        if not n.lower().endswith(".jpg"):
            continue
        base = os.path.basename(n)
        pid = re.sub(r"_\d+\.jpg$", "", base)
        m[pid].append(n)
    for v in m.values():
        v.sort()
    return z, m


def embed(engine, z, name):
    """Фото из zip -> (normed_embedding, bbox, img) ровно одного лица или None."""
    img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = [f for f in engine.detect(img) if float(f.det_score) >= 0.5]
    if len(faces) != 1:
        return None
    return faces[0].normed_embedding, faces[0].bbox, img


def main():
    cfg = load_settings()
    ING, RET = _dataset_zips()
    zi, ing = people_map(ING)
    zr, ret = people_map(RET)

    # кандидаты: >=3 фото на заведение и >=2 на проверку
    good = sorted(pid for pid in ing
                  if len(ing[pid]) >= 3 and len(ret.get(pid, [])) >= 2)
    enrolled_ids = good  # берём кандидатов, пока не заведём ровно N_PEOPLE
    # чужие: есть в retrieval, НЕ среди кандидатов на заведение
    impostor_ids = [p for p in sorted(ret) if p not in set(good)]
    print(f"кандидатов с 3+2 фото: {len(good)}; цель: завести {N_PEOPLE}, "
          f"чужих {N_IMPOSTORS}")

    engine = FaceEngine(det_size=(640, 640), ctx_id=cfg["gpu"]["ctx_id"],
                        allowed_modules=["detection", "recognition"])
    FaceEngine.warmup(engine, size=640)

    # две свежие галереи
    dirs = {k: os.path.join("data", "_eval", f"gallery__lfw100_{k}") for k in ("g1", "g3")}
    gals = {}
    for k, d in dirs.items():
        shutil.rmtree(d, ignore_errors=True)
        gals[k] = Gallery({**cfg, "gallery": {**cfg["gallery"], "dir": d}})

    # --- заведение ---
    t0 = time.time()
    ok_enroll, skip = [], 0
    for pid in enrolled_ids:
        if len(ok_enroll) >= N_PEOPLE:
            break
        embs = []
        for name in ing[pid]:
            r = embed(engine, zi, name)
            if r is not None:
                embs.append(r)
            if len(embs) == 3:
                break
        if len(embs) < 3:
            skip += 1
            continue
        e0, bbox0, img0 = embs[0]
        gals["g1"].add_known(e0, img0, bbox0, pid)
        ident = gals["g3"].add_known(e0, img0, bbox0, pid)
        for e, b, im in embs[1:3]:
            gals["g3"].add_known_embedding(ident.label, e)
        ok_enroll.append(pid)
    print(f"заведено {len(ok_enroll)} чел. (пропущено {skip}: <3 годных фото), "
          f"{time.time()-t0:.0f}s; эмбеддингов: g1={gals['g1'].embeddings.shape[0]} "
          f"g3={gals['g3'].embeddings.shape[0]}")
    label2pid = {k: {i.label: i.name for i in gals[k].identities} for k in gals}

    # --- genuine-пробы: 2 новых фото каждого заведённого ---
    thr = float(cfg["gallery"]["match_threshold"])
    stats = {k: {"rank1": 0, "hit@thr": 0, "n": 0, "scores": []} for k in gals}
    lat = []
    for pid in ok_enroll:
        used = 0
        for name in ret[pid]:
            r = embed(engine, zr, name)
            if r is None:
                continue
            emb = r[0]
            for k, g in gals.items():
                t1 = time.time()
                ident, score = g.identify(emb)
                lat.append(time.time() - t1)
                st = stats[k]
                correct = ident is not None and label2pid[k][ident.label] == pid
                st["n"] += 1
                st["rank1"] += int(correct)
                st["hit@thr"] += int(correct and score >= thr)
                st["scores"].append(score if correct else -1.0)
            used += 1
            if used == 2:
                break

    # --- impostors: 1 фото незаведённых ---
    imp = {k: {"fa": 0, "n": 0, "scores": []} for k in gals}
    for pid in impostor_ids:
        if imp["g1"]["n"] >= N_IMPOSTORS:
            break
        r = None
        for name in ret[pid]:
            r = embed(engine, zr, name)
            if r is not None:
                break
        if r is None:
            continue
        for k, g in gals.items():
            ident, score = g.identify(r[0])
            imp[k]["n"] += 1
            imp[k]["fa"] += int(score >= thr)
            imp[k]["scores"].append(score)

    def pct(v, p):
        return float(np.percentile(v, p)) if v else 0.0

    print(f"\n===== РЕЗУЛЬТАТ (порог {thr}) =====")
    for k, title in (("g1", "enrollment по 1 фото (как раньше)"),
                     ("g3", "enrollment по 3 фото (мульти-фото)")):
        st, im = stats[k], imp[k]
        gs = [s for s in st["scores"] if s >= 0]
        print(f"\n[{title}]")
        print(f"  genuine-проб: {st['n']}   rank-1: {st['rank1']} "
              f"({100*st['rank1']/max(st['n'],1):.1f}%)   "
              f"узнан с порогом: {st['hit@thr']} ({100*st['hit@thr']/max(st['n'],1):.1f}%)")
        print(f"  score своих (верный матч): p10={pct(gs,10):.2f} "
              f"med={pct(gs,50):.2f} p90={pct(gs,90):.2f}")
        print(f"  impostors: {im['n']}   false accept >= {thr}: {im['fa']} "
              f"({100*im['fa']/max(im['n'],1):.1f}%)   "
              f"score чужих: med={pct(im['scores'],50):.2f} max={max(im['scores'] or [0]):.2f}")
    print(f"\nidentify() латентность: med={1000*pct(lat,50):.2f}ms "
          f"p99={1000*pct(lat,99):.2f}ms (эмбеддингов в базе: до "
          f"{max(g.embeddings.shape[0] for g in gals.values())})")

    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    print("галереи-песочницы удалены")


if __name__ == "__main__":
    main()
