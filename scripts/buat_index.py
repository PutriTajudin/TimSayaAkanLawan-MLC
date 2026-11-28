"""Membuat outputs/index_potongan.csv: satu baris = satu potongan 30 frame, dengan label dari anotasi manual.
Juga menggabungkan annotations_<nama>.csv menjadi annotations/annotations_all.csv.

Logikanya ada di src/index.py (ikut dikirim ke Kaggle); file ini hanya pembungkus CLI.

Kolom frame_ragu = berapa frame dari potongan itu masuk bagian yang ditandai ragu (tombol r di alat anotasi).
Tanda ragu TIDAK mengubah label. Untuk uji "tanpa bagian ragu", buang baris yang frame_ragu > 0.

  python scripts/buat_index.py              # wajib: semua video cheat sudah dianotasi
  python scripts/buat_index.py --sementara  # hanya untuk uji kode: video cheat yang belum dianotasi
                                            # diberi label video (semua potongan = 1). JANGAN dipakai untuk training akhir.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ON_KAGGLE, ROOT, get_path, load_config
from src.index import baca_anotasi, bangun_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementara", action="store_true")
    ap.add_argument("--loncat", type=int, default=10, help="geseran antar potongan (frame)")
    ap.add_argument("--out", default="outputs/index_potongan.csv")
    args = ap.parse_args()

    cfg = load_config()
    anotasi_dir = get_path(cfg, "annotations")
    if not ON_KAGGLE:  # di Kaggle folder annotations hanya-baca
        _, _, df_anotasi = baca_anotasi(anotasi_dir)
        if df_anotasi is not None:
            df_anotasi.to_csv(anotasi_dir / "annotations_all.csv", index=False, lineterminator="\n")

    df = bangun_index(cfg, loncat=args.loncat, sementara=args.sementara)

    out = (Path("/kaggle/working") if ON_KAGGLE else ROOT) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pakai = df[df["dipakai"]]
    print("Total potongan:", len(df), "| dipakai:", len(pakai), "| dibuang (tepat setengah):", len(df) - len(pakai))
    print(f"Bagian ragu: {df.attrs['n_rentang_ragu']} rentang di {df.attrs['n_video_ragu']} video | "
          f"potongan dipakai yang menyentuh bagian ragu: {int((pakai['frame_ragu'] > 0).sum())}")
    print("Label (potongan dipakai):\n", pakai.groupby("label").size().to_string())
    print("Potongan per fold:\n", pakai.groupby("fold").size().to_string())
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()
