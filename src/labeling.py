"""Aturan label potongan 30 frame dari catatan anotasi manual.
Dipakai oleh scripts/buat_index.py dan notebook 02_make_sequences."""


def gabung_interval(interval):
    """Gabung interval (mulai, selesai) yang tumpang tindih. Interval kosong dibuang."""
    bersih = sorted((a, b) for a, b in interval if b > a)
    hasil = []
    for a, b in bersih:
        if hasil and a <= hasil[-1][1]:
            hasil[-1] = (hasil[-1][0], max(hasil[-1][1], b))
        else:
            hasil.append((a, b))
    return hasil


def frame_dalam_interval(mulai, interval, panjang=30):
    """Berapa frame dari potongan [mulai, mulai + panjang) yang masuk ke interval-interval ini."""
    akhir = mulai + panjang
    total = 0
    for a, b in gabung_interval(interval):
        total += max(0, min(akhir, b) - max(mulai, a))
    return total


def label_potongan(mulai, interval_cheat, panjang=30):
    """Beri label potongan frame [mulai, mulai + panjang).
    Return (label, jumlah_frame_cheat):
      - frame cheat lebih dari setengah  -> 1
      - frame cheat kurang dari setengah -> 0
      - tepat setengah                   -> None (potongan tidak dipakai)
    Batas setengah adalah terjemahan tim atas kata 'perilaku dominan' di FAQ 41,
    bukan aturan resmi panitia."""
    n_cheat = frame_dalam_interval(mulai, interval_cheat, panjang)
    setengah = panjang / 2
    if n_cheat > setengah:
        return 1, n_cheat
    if n_cheat < setengah:
        return 0, n_cheat
    return None, n_cheat
