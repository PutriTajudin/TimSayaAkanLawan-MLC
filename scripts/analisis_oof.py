"""Analisis prediksi OOF notebook 04: cari di mana modelnya salah.

Kunci penggabungan = kolom `video` ("train/<kategori>/<subject>/<sudut>_<perilaku>.MOV").
Subjek, sudut, dan perilaku diambil dari nama berkas itu, sama seperti src/index.py.
File anotasi memakai kolom `video` yang sama, jadi dipakai untuk memeriksa cakupan.

  python scripts/analisis_oof.py
  python scripts/analisis_oof.py --threshold 0.88 --oof <path lain>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import ROOT, get_path, load_config
from src.metrics import skor

OOF_BAWAAN = "outputs/kaggle/mlc-04-train-baseline/outputs/oof_predictions/oof_04.csv"


def bagian(video):
    """('kategori', 'subject_n', 'sudut', 'perilaku') dari kolom video."""
    p = Path(str(video))
    kategori, subjek, stem = p.parts[-3], p.parts[-2], p.stem
    sudut, _, perilaku = stem.partition("_")
    return kategori, subjek, sudut.lower(), perilaku


def tabel_salah(df, kolom, n=None):
    """Tingkat salah per nilai `kolom`, diurutkan dari paling sering salah."""
    g = df.groupby(kolom)
    t = pd.DataFrame({
        "n": g.size(),
        "n1": g["label"].sum(),
        "salah": g["keliru"].sum(),
        "fp": g["fp"].sum(),
        "fn": g["fn"].sum(),
    })
    t["rate"] = t["salah"] / t["n"]
    t = t.sort_values(["rate", "salah"], ascending=False)
    return t.head(n) if n else t


def cetak(t, nama):
    print(f"  {nama:<24} {'n':>5} {'n(1)':>5} {'salah':>5} {'rate':>6} {'FP':>4} {'FN':>4}")
    for k, r in t.iterrows():
        print(f"  {str(k):<24} {r.n:>5.0f} {r.n1:>5.0f} {r.salah:>5.0f} "
              f"{r.rate:>6.3f} {r.fp:>4.0f} {r.fn:>4.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.88)
    ap.add_argument("--oof", default=OOF_BAWAAN)
    args = ap.parse_args()

    cfg = load_config()
    metrik = cfg["METRIC"]
    thr = args.threshold

    p_oof = Path(args.oof)
    if not p_oof.is_absolute():
        p_oof = ROOT / p_oof
    if not p_oof.exists():
        raise SystemExit(f"OOF tidak ada: {p_oof}")
    oof = pd.read_csv(p_oof)
    for k in ("video", "fold", "label", "prob"):
        if k not in oof.columns:
            raise SystemExit(f"kolom '{k}' tidak ada di {p_oof.name}: {list(oof.columns)}")

    pecah = oof["video"].map(bagian)
    oof["kategori"] = [b[0] for b in pecah]
    oof["subject"] = [b[1] for b in pecah]
    oof["sudut"] = [b[2] for b in pecah]
    oof["perilaku"] = [b[3] for b in pecah]
    oof["pred"] = (oof["prob"] >= thr).astype(int)
    oof["keliru"] = (oof["pred"] != oof["label"]).astype(int)
    oof["fp"] = ((oof["pred"] == 1) & (oof["label"] == 0)).astype(int)
    oof["fn"] = ((oof["pred"] == 0) & (oof["label"] == 1)).astype(int)

    # Pemeriksaan kunci: fold di OOF harus sama dengan annotations/folds.csv
    folds = pd.read_csv(get_path(cfg, "folds")).set_index("subject")["fold"]
    beda = int((oof["subject"].map(folds) != oof["fold"]).sum())
    # Cakupan anotasi (hanya video cheat yang dianotasi)
    p_anot = get_path(cfg, "annotations") / "annotations_all.csv"
    anot = set(pd.read_csv(p_anot, usecols=["video"])["video"]) if p_anot.exists() else set()
    vid_cheat = set(oof.loc[oof["kategori"] == "cheat", "video"])

    print(f"OOF {p_oof.name}: {len(oof)} jendela | {oof['video'].nunique()} video | "
          f"{oof['subject'].nunique()} subjek | thr={thr} | metrik={metrik}")
    print(f"kunci=video | fold cocok dengan folds.csv: {'ya' if beda == 0 else f'TIDAK ({beda} baris)'}"
          f" | video cheat ada di annotations_all.csv: {len(vid_cheat & anot)}/{len(vid_cheat)}")

    print(f"\n1) {metrik} @ thr {thr}")
    print(f"  total: {skor(oof['label'], oof['prob'], metrik, thr):.4f}  "
          f"(rasio label 1 = {oof['label'].mean():.3f})")
    per_fold = [f"f{f}={skor(g['label'], g['prob'], metrik, thr):.4f}"
                for f, g in oof.groupby("fold")]
    print("  per fold: " + "  ".join(per_fold))

    print("\n2) 10 subjek dengan tingkat salah tertinggi")
    cetak(tabel_salah(oof, "subject", 10), "subject")

    print("\n3) tingkat salah per perilaku")
    cetak(tabel_salah(oof, "perilaku"), "perilaku")
    print("\n   per sudut kamera")
    cetak(tabel_salah(oof, "sudut"), "sudut")

    fp, fn = int(oof["fp"].sum()), int(oof["fn"].sum())
    print(f"\n4) FP={fp} ({fp / len(oof):.3%})  FN={fn} ({fn / len(oof):.3%})  "
          f"total salah={fp + fn} ({(fp + fn) / len(oof):.3%})")
    n1 = int(oof["label"].sum())
    print(f"   recall label 1 = {1 - fn / max(n1, 1):.3f} ({n1 - fn}/{n1})  |  "
          f"presisi label 1 = {(n1 - fn) / max(n1 - fn + fp, 1):.3f}")


if __name__ == "__main__":
    main()
