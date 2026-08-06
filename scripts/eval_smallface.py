# -*- coding: utf-8 -*-
"""
Тест на МЕЛКИХ реальных лицах (ядро проблемы проекта: обзорные камеры, 30-45px).

Сценарий = прод: работники заведены ХОРОШИМИ фото (API known-faces, 3 ракурса),
камера видит их мелко. Меряем на LFW (реальные лица):
  1) узнаваемость заведённых при лице ~110px (эталон) / ~45px / ~32px:
     - одиночный кадр (текущий матчинг)
     - агрегат 5 кадров (шаг 4 track_enroll)
  2) самосогласованность 5 мелких кадров ОДНОГО человека (гейт шага 2:
     проверка дефолта min_consistency=0.35 на реальных данных)
  3) score чужих на тех же масштабах (FAR при 0.5, запас до new_id 0.3)

Запуск: python scripts\eval_smallface.py  (dev: OPENBLAS_NUM_THREADS=1)
Эталон (06.08.2026, buffalo_l, det 640, 60 чел., порог 0.5):
  лицо ~37px + JPEG q35 + смаз (реальность обзорной камеры):
    одиночный кадр 71.5% | АГРЕГАТ 5 КАДРОВ (шаг 4) 100% | FAR 0/60
    согласованность своих: p10 0.41 (гейт 0.35 своих не режет)
  лицо ~26px + камера (за пределом): одиночный 16.4% | агрегат 76.4% | FAR 0/60
    согласованность p10 0.32 — гейт 0.35 тут режет и своих (создание ID на
    таком качестве и не планируется)
  Вывод: match_by_aggregate (шаг 4) — главный рычаг для обзорных камер;
  чистый даунскейл без компрессии почти не вредит (98% даже при 26px) —
  вредит именно компрессия+смаз, поэтому битрейт RTSP-потока критичен.
"""
import os
import re
import sys
import zipfile
import collections

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import cv2
import numpy as np

from config import load_settings
from face_engine import FaceEngine
from gallery import aggregate_embeddings

N_PEOPLE = 60          # заведённые (нужно >=3 ing + >=5 ret фото)
N_IMPOSTORS = 60
SCALES = {"~110px (ориг.)": (1.0, False), "~45px": (0.40, False),
          "~32px": (0.28, False),
          "~45px + камера (JPEG q35 + смаз)": (0.40, True),
          "~32px + камера (JPEG q35 + смаз)": (0.28, True)}


def people_map(zpath):
    z = zipfile.ZipFile(zpath)
    m = collections.defaultdict(list)
    for n in z.namelist():
        if n.lower().endswith(".jpg"):
            m[re.sub(r"_\d+\.jpg$", "", os.path.basename(n))].append(n)
    for v in m.values():
        v.sort()
    return z, m


def one_face(engine, img, min_det=0.4):
    faces = [f for f in engine.detect(img) if float(f.det_score) >= min_det]
    if len(faces) != 1:
        return None
    return faces[0]


def embed_scaled(engine, z, name, scale, degrade=False):
    img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, 0
    if scale < 1.0:
        img = cv2.resize(img, (max(32, int(img.shape[1] * scale)),
                               max(32, int(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    if degrade:  # реальная камера: лёгкий смаз движения + жёсткая компрессия
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 35])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    f = one_face(engine, img)
    if f is None:
        return None, 0
    px = int(min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]))
    return f.normed_embedding, px


def main():
    from huggingface_hub import hf_hub_download
    ING = hf_hub_download("vilsonrodrigues/lfw", "lfw_multifaces-ingestion.zip",
                          repo_type="dataset")
    RET = hf_hub_download("vilsonrodrigues/lfw", "lfw_multifaces-retrieval.zip",
                          repo_type="dataset")
    cfg = load_settings()
    zi, ing = people_map(ING)
    zr, ret = people_map(RET)
    good = sorted(p for p in ing if len(ing[p]) >= 3 and len(ret.get(p, [])) >= 5)
    imps = [p for p in sorted(ret) if p not in set(good)]
    engine = FaceEngine(det_size=(640, 640), ctx_id=cfg["gpu"]["ctx_id"],
                        allowed_modules=["detection", "recognition"])
    FaceEngine.warmup(engine, size=640)

    # --- заведение: 3 ХОРОШИХ фото (полный размер), шаблон = все 3 эмбеддинга ---
    gal = {}                     # pid -> (3,512)
    for pid in good:
        embs = []
        for name in ing[pid]:
            e, _ = embed_scaled(engine, zi, name, 1.0)
            if e is not None:
                embs.append(e)
            if len(embs) == 3:
                break
        if len(embs) == 3:
            gal[pid] = np.stack(embs)
        if len(gal) >= N_PEOPLE:
            break
    pids = list(gal)
    G = np.concatenate([gal[p] for p in pids])          # (3N,512)
    owner = np.repeat(np.arange(len(pids)), 3)
    print(f"заведено {len(pids)} чел. по 3 хороших фото (база {G.shape[0]} эмб.)")

    def best_match(e):
        s = G @ e
        i = int(np.argmax(s))
        return int(owner[i]), float(s[i])

    thr = float(cfg["gallery"]["match_threshold"])

    # --- пробы на каждом масштабе ---
    for title, (sc, dg) in SCALES.items():
        hit1 = hitA = n = 0
        px_list, cons_list, genuine_sc, agg_sc = [], [], [], []
        for k, pid in enumerate(pids):
            embs, pxs = [], []
            for name in ret[pid]:
                e, px = embed_scaled(engine, zr, name, sc, dg)
                if e is not None:
                    embs.append(e); pxs.append(px)
                if len(embs) == 5:
                    break
            if len(embs) < 5:
                continue
            px_list += pxs
            m = np.stack(embs)
            cons = float((m @ m.T).sum() - 5) / 20     # средний попарный cosine
            cons_list.append(cons)
            # одиночный кадр (каждый из 5 — отдельная проба)
            for e in embs:
                o, s = best_match(e)
                n += 1
                hit1 += int(o == k and s >= thr)
                if o == k:
                    genuine_sc.append(s)
            # агрегат 5 кадров (шаг 4)
            agg = aggregate_embeddings(list(m))
            o, s = best_match(agg)
            hitA += int(o == k and s >= thr)
            agg_sc.append(s if o == k else -1)
        # чужие на этом масштабе
        imp_sc = []
        cnt = 0
        for pid in imps:
            if cnt >= N_IMPOSTORS:
                break
            for name in ret[pid][:2]:
                e, _ = embed_scaled(engine, zr, name, sc, dg)
                if e is not None:
                    imp_sc.append(best_match(e)[1])
                    cnt += 1
                    break
        pctl = lambda v, p: float(np.percentile(v, p)) if v else 0.0
        nA = len(agg_sc)
        print(f"\n--- {title}: лицо med={pctl(px_list,50):.0f}px "
              f"(люди с 5 годными кадрами: {nA}/{len(pids)}) ---")
        print(f"  одиночный кадр:  узнан с порогом {thr}: {hit1}/{n} "
              f"({100*hit1/max(n,1):.1f}%)  score своих med={pctl(genuine_sc,50):.2f} "
              f"p10={pctl(genuine_sc,10):.2f}")
        print(f"  агрегат 5 кадров (шаг 4): {hitA}/{nA} ({100*hitA/max(nA,1):.1f}%)  "
              f"score med={pctl([s for s in agg_sc if s>=0],50):.2f}")
        print(f"  самосогласованность своих 5 кадров: med={pctl(cons_list,50):.2f} "
              f"p10={pctl(cons_list,10):.2f} min={min(cons_list or [0]):.2f} "
              f"(гейт шага 2 = 0.35)")
        print(f"  чужие: n={len(imp_sc)} max={max(imp_sc or [0]):.2f} "
              f"FA@{thr}: {sum(s >= thr for s in imp_sc)}  "
              f">=0.3 (new_id): {sum(s >= 0.3 for s in imp_sc)}")


if __name__ == "__main__":
    main()
