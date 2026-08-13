# -*- coding: utf-8 -*-
r"""
dedup_gallery.py — offline-дедупликация накопленных дублей в галерее.

Один реальный человек мог получить несколько person_XXXX (мелкие лица, шумные
эмбеддинги). Скрипт находит кластеры почти одинаковых людей и сливает их:

  1. центроид каждого ID (среднее его эмбеддингов, L2-норм.);
  2. агломеративная кластеризация (union-find) по cosine >= --merge-threshold;
  3. канонический ID кластера = самый старый (min first_seen); эмбеддинги
     остальных переносятся к нему (до max_embeddings_per_id), события в SQLite
     перепривязываются, дубль удаляется вместе с фото.

known_XXXX автоматически НЕ сливаются: кластер с known попадает в отчёт как
«вручную»; --include-known разрешает слить person_XXXX В known (known всегда
канонический). known+known не сливаются никогда.

Режимы:
  python scripts\dedup_gallery.py                  # dry-run: только HTML-отчёт
  python scripts\dedup_gallery.py --apply          # применить слияния
  python scripts\dedup_gallery.py --merge-threshold 0.55 --report out.html

ВАЖНО (--apply): остановить face-recognition (иначе процесс распознавания
перезапишет галерею из памяти): sudo systemctl stop face-recognition
Можно гонять ночным cron'ом (с --apply и остановкой/стартом сервиса вокруг).
"""
import os
import sys
import html
import time
import argparse
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from config import load_settings
from gallery import Gallery


def centroids(g: Gallery):
    """label -> (Identity, центроид)."""
    out = {}
    for ident in g.identities:
        rows = g.embeddings[g.owners == ident.idx]
        if rows.shape[0] == 0:
            continue
        c = rows.mean(axis=0)
        n = float(np.linalg.norm(c))
        if n > 1e-6:
            out[ident.label] = (ident, (c / n).astype(np.float32))
    return out


def find_clusters(cent: dict, thr: float):
    """Union-find по парам центроидов с cosine >= thr. Возвращает кластеры >= 2."""
    labels = sorted(cent)
    if not labels:
        return [], {}
    M = np.stack([cent[l][1] for l in labels])
    S = M @ M.T
    parent = list(range(len(labels)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pair_sim = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            # known+known никогда не сливаем — даже не связываем
            if cent[labels[i]][0].known and cent[labels[j]][0].known:
                continue
            if float(S[i, j]) >= thr:
                pair_sim[(labels[i], labels[j])] = float(S[i, j])
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(root(i), []).append(l)
    clusters = [sorted(v) for v in groups.values() if len(v) >= 2]
    clusters.sort()
    return clusters, pair_sim


def canonical_of(cluster, cent):
    """known приоритетнее (у него ФИО), иначе самый старый ID."""
    knowns = [l for l in cluster if cent[l][0].known]
    if knowns:
        return knowns[0]
    return min(cluster, key=lambda l: (cent[l][0].first_seen, l))


def write_report(path, clusters, cent, pair_sim, thr, root_dir):
    rows = []
    for cl in clusters:
        canon = canonical_of(cl, cent)
        has_known = any(cent[l][0].known for l in cl)
        cards = []
        for l in cl:
            ident = cent[l][0]
            img = ident.crop_path if os.path.isabs(ident.crop_path) else \
                os.path.join(root_dir, ident.crop_path)
            sims = [s for (a, b), s in pair_sim.items() if l in (a, b)]
            tag = " (КАНОНИЧЕСКИЙ)" if l == canon else ""
            name = f" — {html.escape(ident.name)}" if ident.name else ""
            cards.append(
                f"<figure><img src='file:///{html.escape(img.replace(os.sep, '/'))}' "
                f"onerror=\"this.style.opacity=.2\">"
                f"<figcaption><b>{html.escape(l)}</b>{name}{tag}<br>"
                f"эмб: {ident.n_emb}, max sim: {max(sims):.2f}</figcaption></figure>")
        note = "<p class='warn'>⚠ содержит known — автоматически НЕ сливается " \
               "(--include-known: person сольются В known)</p>" if has_known else ""
        rows.append(f"<div class='cluster'><h3>{' + '.join(cl)} → {canon}</h3>"
                    f"{note}<div class='faces'>{''.join(cards)}</div></div>")
    doc = f"""<!doctype html><meta charset="utf-8"><title>dedup report</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;padding:20px}}
.cluster{{border:1px solid #444;border-radius:8px;padding:12px;margin:14px 0}}
.faces{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0;text-align:center}}
img{{width:140px;height:140px;object-fit:cover;border-radius:6px}}
.warn{{color:#fa0}}h3{{margin:0 0 8px}}</style>
<h1>Дедупликация галереи — {time.strftime('%Y-%m-%d %H:%M')}</h1>
<p>Порог слияния cosine: {thr}. Кластеров к слиянию: {len(rows)}.</p>
{''.join(rows) or '<p>Дублей не найдено 🎉</p>'}"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def apply_merges(g: Gallery, clusters, cent, db_path, include_known):
    merged = events_moved = 0
    conn = sqlite3.connect(db_path) if os.path.exists(db_path) else None
    for cl in clusters:
        if any(cent[l][0].known for l in cl) and not include_known:
            print(f"  [skip] {cl}: содержит known (нужен --include-known)")
            continue
        canon_l = canonical_of(cl, cent)
        canon = g.get_by_label(canon_l)
        for l in cl:
            if l == canon_l:
                continue
            dup = g.get_by_label(l)
            if dup is None or canon is None:
                continue
            # эмбеддинги дубля -> каноническому (до лимита)
            rows = g.embeddings[g.owners == dup.idx]
            for r in rows:
                if canon.n_emb >= g.max_emb:
                    break
                g._append_embedding(canon.idx, r)
            # события -> канонический ID
            if conn is not None:
                cur = conn.execute("UPDATE events SET person=? WHERE person=?",
                                   (canon_l, l))
                events_moved += cur.rowcount
            g.delete_identity(l)          # удаляет эмбеддинги, фото, переиндексирует
            canon = g.get_by_label(canon_l)  # idx мог сдвинуться после удаления
            merged += 1
            print(f"  {l} -> {canon_l}")
    if conn is not None:
        conn.commit()
        conn.close()
    g.save()
    return merged, events_moved


def main():
    ap = argparse.ArgumentParser(description="Слияние дублей person_XXXX в галерее")
    ap.add_argument("--merge-threshold", type=float, default=0.6,
                    help="cosine центроидов для слияния (деф. 0.6)")
    ap.add_argument("--report", default=os.path.join("data", "dedup_report.html"))
    ap.add_argument("--apply", action="store_true",
                    help="применить (без него — только отчёт). ОСТАНОВИТЬ face-recognition!")
    ap.add_argument("--include-known", action="store_true",
                    help="разрешить сливать person_XXXX В known_XXXX")
    args = ap.parse_args()

    cfg = load_settings()
    g = Gallery(cfg)
    cent = centroids(g)
    clusters, pair_sim = find_clusters(cent, args.merge_threshold)
    print(f"ID в галерее: {g.count()}, кластеров-дублей: {len(clusters)}")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_report(args.report, clusters, cent, pair_sim, args.merge_threshold, root_dir)
    print(f"отчёт: {args.report}")

    if not clusters:
        return 0
    if not args.apply:
        print("dry-run: ничего не изменено. Проверь отчёт и запусти с --apply "
              "(остановив face-recognition).")
        return 0
    merged, ev = apply_merges(g, clusters, cent, cfg["paths"]["db"], args.include_known)
    print(f"слито ID: {merged}, перепривязано событий: {ev}, осталось ID: {g.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
