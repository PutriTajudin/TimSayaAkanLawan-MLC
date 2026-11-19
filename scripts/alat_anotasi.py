"""Alat bantu label manual: putar video cheat, lalu tandai kapan aksi menyontek mulai dan selesai.
Alat ini HANYA mencatat tombol yang kamu tekan. Waktunya ditentukan dari hasil menontonmu sendiri.

Contoh (jalankan dari folder utama repo):
  python scripts/alat_anotasi.py --nama putri --subjek 1-13
  python scripts/alat_anotasi.py --nama elfa  --subjek 14-26

Hasil: annotations/annotations_<nama>.csv. Boleh dihentikan kapan saja (Esc).
Video yang sudah disimpan dilewati saat dijalankan lagi.

Tombol r = tandai bagian yang RAGU (tekan di awal, lalu di akhir). Tanda ragu hanya catatan,
tidak mengubah label. Tersimpan sebagai baris dengan jenis = ragu.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.data import get_path, load_config

CFG = load_config()
DATA_DIR = get_path(CFG, "raw")
ANOTASI_DIR = get_path(CFG, "annotations")
TRAIN_DIR = DATA_DIR / "train"

KOLOM = ["video", "detik_mulai", "detik_selesai", "frame_mulai", "frame_selesai", "jenis"]
JENDELA = "Anotasi Data Slayer"


def parse_rentang(teks):
    if "-" in teks:
        a, b = teks.split("-")
        return int(a), int(b)
    n = int(teks)
    return n, n


def daftar_video_cheat(lo, hi):
    folder = TRAIN_DIR / "cheat"
    hasil = []
    for s in sorted(os.listdir(folder), key=lambda x: int(x.split("_")[1])):
        if lo <= int(s.split("_")[1]) <= hi:
            for nama in sorted(os.listdir(folder / s)):
                hasil.append((folder / s / nama).relative_to(DATA_DIR).as_posix())
    return hasil


def pastikan_kolom(file_csv):
    """File format lama (tanpa kolom jenis) diperbarui: semua baris lama dianggap cheat."""
    if not file_csv.exists() or file_csv.stat().st_size == 0:
        return
    with open(file_csv, newline="", encoding="utf-8-sig") as f:
        baris = [b for b in csv.reader(f) if b]
    if not baris or "jenis" in baris[0]:
        return
    baris[0].append("jenis")
    for b in baris[1:]:
        b.append("cheat")
    with open(file_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(baris)
    print(f"File {file_csv.name} diperbarui ke format baru (kolom jenis ditambahkan, baris lama = cheat).")


def video_sudah_dianotasi(file_csv):
    if not file_csv.exists() or file_csv.stat().st_size == 0:
        return set()
    with open(file_csv, newline="", encoding="utf-8-sig") as f:
        return {baris["video"] for baris in csv.DictReader(f)}


def simpan_video(file_csv, rel, interval, ragu, kosong, fps):
    baru = not file_csv.exists() or file_csv.stat().st_size == 0
    file_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(file_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        if baru:
            w.writerow(KOLOM)
        if kosong or not interval:
            w.writerow([rel, 0, 0, 0, 0, "cheat"])      # tanda: video ini tidak ada aksi cheat
        for a, b in interval:
            w.writerow([rel, round(a / fps, 2), round(b / fps, 2), a, b, "cheat"])
        for a, b in ragu:
            w.writerow([rel, round(a / fps, 2), round(b / fps, 2), a, b, "ragu"])


def muat_frame(path, lebar):
    """Baca semua frame video secara berurutan (nomor frame pasti tepat), kecilkan ke lebar tertentu."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        frames.append(cv2.resize(f, (lebar, int(round(h * lebar / w)))))
    cap.release()
    return frames, fps


def tulis(img, teks, y, skala=0.5):
    cv2.putText(img, teks, (8, y), cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, teks, (8, y), cv2.FONT_HERSHEY_SIMPLEX, skala, (255, 255, 255), 1, cv2.LINE_AA)


def gambar_layar(frame, judul, pos, n, fps, interval, ragu, tertunda, tertunda_ragu, kosong, status, main):
    img = frame.copy()
    h, w = img.shape[:2]
    y0 = h - 22                      # bar bawah: separuh atas = ragu (biru), separuh bawah = cheat (merah)
    ymid = y0 + 11
    cv2.rectangle(img, (0, y0), (w, h), (50, 50, 50), -1)
    for a, b in ragu:
        cv2.rectangle(img, (int(a / n * w), y0), (int(b / n * w), ymid), (255, 160, 0), -1)
    if tertunda_ragu is not None:
        kiri, kanan = sorted((tertunda_ragu, pos))
        cv2.rectangle(img, (int(kiri / n * w), y0), (int(kanan / n * w), ymid), (255, 230, 120), -1)
    for a, b in interval:
        cv2.rectangle(img, (int(a / n * w), ymid), (int(b / n * w), h), (0, 0, 255), -1)
    if tertunda is not None:
        kiri, kanan = sorted((tertunda, pos))
        cv2.rectangle(img, (int(kiri / n * w), ymid), (int(kanan / n * w), h), (0, 200, 255), -1)
    x = int(pos / n * w)
    cv2.line(img, (x, y0 - 6), (x, h), (255, 255, 255), 2)

    tulis(img, judul, 20)
    tulis(img, f"Frame {pos}/{n - 1} | {pos / fps:.2f} dtk | {'MAIN' if main else 'JEDA'}", 42)
    if kosong:
        ringkas = "Dicatat: TIDAK ADA aksi cheat"
    elif interval:
        ringkas = "Cheat (frame): " + ", ".join(f"{a}-{b}" for a, b in interval)
    else:
        ringkas = "Belum ada catatan"
    if tertunda is not None:
        ringkas += f" | mulai tertunda: {tertunda}"
    tulis(img, ringkas, 64)
    baris_ragu = "Ragu (frame): " + (", ".join(f"{a}-{b}" for a, b in ragu) if ragu else "-")
    if tertunda_ragu is not None:
        baris_ragu += f" | awal ragu tertunda: {tertunda_ragu}"
    tulis(img, baris_ragu, 86)
    if status:
        tulis(img, status, 108)
    tulis(img, "spasi=main/jeda  a/d=1 frame  j/l=1 detik  m=mulai  n=selesai", y0 - 52, 0.45)
    tulis(img, "r=tandai ragu (awal lalu akhir)  u=batal  0=tanpa cheat  f=semua cheat", y0 - 32, 0.45)
    tulis(img, "Enter=simpan  Esc=keluar | bar: merah=cheat, biru=ragu", y0 - 12, 0.45)
    return img


def anotasi_video(judul, frames, fps):
    n = len(frames)
    langkah = int(round(fps))
    pos, main = 0, False
    interval, ragu = [], []
    riwayat = []                       # urutan interval yang sudah selesai: "cheat" atau "ragu"
    tertunda = tertunda_ragu = None
    kosong, status = False, ""
    while True:
        cv2.imshow(JENDELA, gambar_layar(frames[pos], judul, pos, n, fps, interval, ragu,
                                         tertunda, tertunda_ragu, kosong, status, main))
        if cv2.getWindowProperty(JENDELA, cv2.WND_PROP_VISIBLE) < 1:
            return ("keluar",)
        k = cv2.waitKey(int(1000 / fps) if main else 30) & 0xFF
        if k == 255:
            if main:
                if pos < n - 1:
                    pos += 1
                else:
                    main = False
            continue
        if k == 27:
            return ("keluar",)
        ch = chr(k).lower()
        status = ""
        if ch in (" ", "k"):
            main = not main
        elif ch == "a":
            pos, main = max(0, pos - 1), False
        elif ch == "d":
            pos, main = min(n - 1, pos + 1), False
        elif ch == "j":
            pos = max(0, pos - langkah)
        elif ch == "l":
            pos = min(n - 1, pos + langkah)
        elif ch == "m":
            tertunda, kosong = pos, False
        elif ch == "n":
            akhir = n if pos == n - 1 else pos      # di frame terakhir: sampai akhir video
            if tertunda is None:
                status = "Tekan m dulu untuk menandai awal."
            elif akhir <= tertunda:
                status = "Titik selesai harus setelah titik mulai."
            else:
                interval.append((tertunda, akhir))
                riwayat.append("cheat")
                tertunda, kosong = None, False
        elif ch == "r":
            akhir = n if pos == n - 1 else pos
            if tertunda_ragu is None:
                tertunda_ragu = pos
                status = f"Ragu: awal di frame {pos}. Tekan r lagi di akhir bagian ragu."
            elif akhir <= tertunda_ragu:
                status = "Titik akhir ragu harus setelah titik awal."
            else:
                ragu.append((tertunda_ragu, akhir))
                riwayat.append("ragu")
                tertunda_ragu = None
        elif ch == "u":
            if tertunda_ragu is not None:
                tertunda_ragu = None
            elif tertunda is not None:
                tertunda = None
            elif riwayat:
                (interval if riwayat.pop() == "cheat" else ragu).pop()
        elif ch == "0":
            interval, ragu, riwayat = [], [], []
            tertunda = tertunda_ragu = None
            kosong = True
        elif ch == "f":
            interval, ragu, riwayat = [(0, n)], [], ["cheat"]
            tertunda = tertunda_ragu = None
            kosong = False
        elif k in (13, 10) or ch == "s":
            if tertunda is not None:
                status = "Ada tanda mulai yang belum diselesaikan (tekan n)."
            elif tertunda_ragu is not None:
                status = "Ada tanda ragu yang belum diselesaikan (tekan r di akhir)."
            elif not interval and not kosong:
                status = "Belum ada catatan. Tandai interval, atau tekan 0 / f."
            else:
                return ("simpan", interval, ragu, kosong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nama", required=True, help="namamu: elfa atau putri")
    ap.add_argument("--subjek", default="1-26", help="rentang subjek, contoh: 1-13")
    ap.add_argument("--lebar", type=int, default=800, help="lebar tampilan (piksel)")
    args = ap.parse_args()

    file_csv = ANOTASI_DIR / f"annotations_{args.nama}.csv"
    pastikan_kolom(file_csv)
    lo, hi = parse_rentang(args.subjek)
    semua = daftar_video_cheat(lo, hi)
    sudah = video_sudah_dianotasi(file_csv)
    sisa = [v for v in semua if v not in sudah]
    print(f"Video cheat subjek {lo}-{hi}: {len(semua)} | sudah: {len(semua) - len(sisa)} | sisa: {len(sisa)}")

    cv2.namedWindow(JENDELA, cv2.WINDOW_AUTOSIZE)
    for i, rel in enumerate(sisa, start=1):
        print(f"[{i}/{len(sisa)}] memuat {rel} ...")
        frames, fps = muat_frame(DATA_DIR / rel, args.lebar)
        if not frames:
            print("  GAGAL membaca video, dilewati.")
            continue
        hasil = anotasi_video(f"{rel.split('/', 2)[2]}  ({i}/{len(sisa)})", frames, fps)
        del frames
        if hasil[0] == "keluar":
            print("Keluar. Video yang sudah disimpan tetap aman.")
            break
        simpan_video(file_csv, rel, hasil[1], hasil[2], hasil[3], fps)
        print("  tersimpan.")
    else:
        print("Semua video di rentang ini sudah dianotasi.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
