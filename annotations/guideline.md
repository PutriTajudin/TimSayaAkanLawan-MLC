# Guideline Anotasi

Anotasi dibuat **manual** oleh anggota tim dengan menonton video. Tidak boleh dibantu AI atau VLM (FAQ 42).
Label bawaan panitia (folder `normal` dan `cheat`) tidak diubah. Catatan ini hanya tambahan (FAQ 30, 44).

## Label

| Label | Definisi |
|-------|----------|
| 0 | Normal: perilaku yang terlihat dalam potongan 30 frame (1 detik) bukan menyontek. Berlaku juga untuk potongan menulis biasa di dalam video `cheat` (FAQ 33, 41). |
| 1 | Cheat: perilaku menyontek yang dominan terlihat dalam potongan 30 frame. |

Label ditentukan dari **perilaku yang terlihat di potongan itu**, bukan dari label videonya.

## Aturan

- Yang dicatat hanya video di folder `cheat`: detik mulai dan detik selesai aksi menyontek. Video `normal` semuanya label 0.
- Satu video boleh punya lebih dari satu interval (contoh: `Repeated_Peeking`). Tiap interval satu baris.
- Tiap video dicatat sendiri. Jangan menyalin angka dari video lain, termasuk pasangan `front` dan `side`.
- Interval dicatat dalam frame: `frame_mulai` termasuk, `frame_selesai` tidak termasuk. Ketelitian setengah detik (15 frame) sudah cukup.
- Pembagian kerja: Putri subjek 1-13, Elfa subjek 14-26.
- Jangan memakai informasi dari data test atau skor leaderboard untuk menentukan label train (FAQ 40).
- Proses anotasi ini dicantumkan di notebook final dan bisa diperiksa panitia (FAQ 44).

Cara memakai alat (dari folder utama repo):

```
python scripts/alat_anotasi.py --nama putri --subjek 1-13
python scripts/alat_anotasi.py --nama elfa  --subjek 14-26
```

Tombol: spasi = main/jeda | a/d = 1 frame | j/l = 1 detik | m = tandai mulai | n = tandai selesai |
u = batalkan | 0 = video tanpa aksi cheat | f = seluruh video cheat | Enter = simpan | Esc = keluar.

Hasil tersimpan di `annotations/annotations_<nama>.csv`. `annotations_all.csv` dibuat otomatis oleh
`python scripts/buat_index.py`.

## Aturan label potongan (dari catatan interval)

Potongan 30 frame diberi label dengan menghitung berapa frame yang masuk interval cheat:

| Frame cheat dalam potongan | Label |
|----------------------------|-------|
| lebih dari 15 | 1 |
| kurang dari 15 | 0 |
| tepat 15 | tidak dipakai |

Contoh: potongan frame 230-259 dengan menyontek mulai di frame 240 berisi 20 frame cheat, jadi label 1.
Potongan frame 220-249 berisi 10 frame cheat, jadi label 0.

Batas 15 frame adalah terjemahan tim atas kata "perilaku dominan" di FAQ 41, bukan aturan resmi panitia.

## Kasus ambigu

Kalau ragu, diskusikan berdua, lalu tulis keputusan bersama di sini supaya konsisten.

- (belum ada keputusan bersama)
