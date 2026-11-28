"""Fungsi ekstraksi fitur yang dipakai di 03_feature_extraction dan 06_inference_submission.

Aturan lomba yang dipatuhi di file ini (nomor FAQ mengikuti dokumen "Jawaban FAQ"):
- Test: SEMUA 30 frame diproses langsung oleh encoder, tanpa dikurangi atau diinterpolasi (66, 70).
- Test: tiap sequence diproses sendiri-sendiri. Tidak ada statistik yang dihitung dari data test (38, 68).
- Train: frame diperkecil ke 640x360 lalu bolak-balik JPEG sebelum di-resize ke input model (73).
  Kualitas JPEG dipilih acak dari rentang tetap yang kita tentukan sendiri. TIDAK ditaksir dari file test.
- Encoder: bobot pretrained ImageNet/LVD (DINOv2) dipakai beku sebagai penghasil fitur (3, 15, 22).
  Tidak ada VLM (26.3) dan tidak ada prediksi zero-shot (14.1).
"""
from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Ukuran frame test yang disetujui panitia sebagai acuan praproses train (FAQ 73)
LEBAR_ASAL, TINGGI_ASAL = 640, 360
# Ukuran input encoder (kelipatan 14 untuk DINOv2): 18 x 32 patch
TINGGI_INPUT, LEBAR_INPUT = 252, 448
RATA2_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Rentang kualitas JPEG untuk frame train. Dipilih sendiri, bukan hasil melihat test.
KUALITAS_JPEG = (70, 95)


# --------------------------------------------------------------------------- praproses frame
def jpeg_bolak_balik(frame_bgr: np.ndarray, kualitas: int) -> np.ndarray:
    """Encode lalu decode JPEG supaya artefak kompresinya mirip frame test yang berformat JPEG."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(kualitas)])
    if not ok:
        raise RuntimeError("Gagal meng-encode JPEG")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def siapkan_frame_train(frame_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Frame video train (1280x720) -> 640x360 -> JPEG bolak-balik (kualitas acak)."""
    kecil = cv2.resize(frame_bgr, (LEBAR_ASAL, TINGGI_ASAL), interpolation=cv2.INTER_AREA)
    q = int(rng.integers(KUALITAS_JPEG[0], KUALITAS_JPEG[1] + 1))
    return jpeg_bolak_balik(kecil, q)


def baca_frame_test(path: str | Path) -> np.ndarray:
    """Baca satu frame test (JPEG) apa adanya sebagai BGR uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Frame tidak terbaca: {path}")
    return img


def ke_input_model(frames_bgr: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Daftar frame BGR uint8 -> array float32 (N, 3, TINGGI_INPUT, LEBAR_INPUT) ternormalisasi ImageNet.

    Normalisasi memakai konstanta ImageNet tetap, BUKAN statistik data test.
    """
    out = np.empty((len(frames_bgr), 3, TINGGI_INPUT, LEBAR_INPUT), dtype=np.float32)
    for i, f in enumerate(frames_bgr):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (LEBAR_INPUT, TINGGI_INPUT), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255.0 - RATA2_IMAGENET) / STD_IMAGENET
        out[i] = x.transpose(2, 0, 1)
    return out


# --------------------------------------------------------------------------- baca video / sequence
def seed_video(seed: int, nama: str) -> int:
    """Seed tetap per video supaya kualitas JPEG acaknya bisa diulang persis."""
    return (int(seed) + zlib.crc32(nama.encode("utf-8"))) % (2**32)


def baca_video_train(path: str | Path, seed: int = 42) -> list[np.ndarray]:
    """Baca SEMUA frame sebuah video train, sudah dipraproses (640x360 + JPEG bolak-balik)."""
    path = Path(path)
    rng = np.random.default_rng(seed_video(seed, path.name))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak terbuka: {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(siapkan_frame_train(f, rng))
    cap.release()
    if not frames:
        raise RuntimeError(f"Video kosong: {path}")
    return frames


def baca_sequence_test(folder: str | Path, panjang: int = 30) -> list[np.ndarray]:
    """Baca 30 frame sebuah sequence test berurutan (01.jpg..30.jpg). Semua frame wajib ada."""
    folder = Path(folder)
    files = sorted(folder.glob("*.jpg"))
    if len(files) != panjang:
        raise RuntimeError(f"{folder.name}: ada {len(files)} frame, seharusnya {panjang}")
    return [baca_frame_test(p) for p in files]


# --------------------------------------------------------------------------- encoder
def buat_encoder(nama: str = "vit_small_patch14_dinov2.lvd142m", device: str = "cuda", pretrained: bool = True):
    """Encoder beku (tanpa training). Mengembalikan model timm dalam mode eval."""
    import timm
    import torch

    m = timm.create_model(
        nama, pretrained=pretrained, num_classes=0, img_size=(TINGGI_INPUT, LEBAR_INPUT)
    )
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    m._device = device  # dipakai embed_frames
    m._pakai_fp16 = device.startswith("cuda") and torch.cuda.is_available()
    return m


def dim_fitur(encoder) -> int:
    """Dimensi fitur satu frame = CLS + rata-rata patch."""
    return 2 * int(encoder.num_features)


def embed_frames(encoder, frames_bgr: list[np.ndarray], batch: int = 64) -> np.ndarray:
    """Frame BGR -> fitur (N, D) float16. D = [token CLS ; rata-rata token patch].

    Tiap frame diproses sendiri-sendiri oleh encoder (tidak saling memengaruhi), jadi aman untuk aturan
    "prediksi independen" di test.
    """
    import torch

    hasil = []
    with torch.inference_mode():
        for i in range(0, len(frames_bgr), batch):
            x = torch.from_numpy(ke_input_model(frames_bgr[i : i + batch])).to(encoder._device)
            if encoder._pakai_fp16:
                x = x.half()
                with torch.autocast("cuda", dtype=torch.float16):
                    tok = encoder.forward_features(x)
            else:
                tok = encoder.forward_features(x)
            n_prefix = getattr(encoder, "num_prefix_tokens", 1)
            cls = tok[:, 0]
            patch = tok[:, n_prefix:].mean(dim=1)
            hasil.append(torch.cat([cls, patch], dim=1).float().cpu().numpy().astype(np.float16))
    return np.concatenate(hasil, axis=0)


# --------------------------------------------------------------------------- ekstraksi massal
def ekstrak_video_train(encoder, path: str | Path, seed: int = 42, batch: int = 64) -> np.ndarray:
    """Satu video train -> fitur semua frame (n_frame, D)."""
    return embed_frames(encoder, baca_video_train(path, seed), batch)


def ekstrak_banyak_video(encoder, daftar_video, simpan_ke, seed: int = 42, batch: int = 64, nama_fn=None, cetak=print):
    """Ekstrak banyak video; hasil tiap video disimpan sebagai .npy. Yang sudah ada dilewati (bisa dilanjutkan).

    Video berikutnya dibaca di thread terpisah selagi GPU memproses video sekarang.
    """
    simpan_ke = Path(simpan_ke)
    simpan_ke.mkdir(parents=True, exist_ok=True)
    nama_fn = nama_fn or (lambda p: Path(p).stem)
    tugas = [p for p in daftar_video if not (simpan_ke / f"{nama_fn(p)}.npy").exists()]
    cetak(f"Video: {len(daftar_video)} | sudah ada: {len(daftar_video) - len(tugas)} | dikerjakan: {len(tugas)}")
    if not tugas:
        return
    with ThreadPoolExecutor(max_workers=1) as ex:
        depan = ex.submit(baca_video_train, tugas[0], seed)
        for i, p in enumerate(tugas):
            frames = depan.result()
            if i + 1 < len(tugas):
                depan = ex.submit(baca_video_train, tugas[i + 1], seed)
            feat = embed_frames(encoder, frames, batch)
            np.save(simpan_ke / f"{nama_fn(p)}.npy", feat)
            if (i + 1) % 10 == 0 or i + 1 == len(tugas):
                cetak(f"  {i + 1}/{len(tugas)} video selesai | {nama_fn(p)} -> {feat.shape}")


def ekstrak_test(encoder, test_dir: str | Path, batch: int = 60, cetak=print):
    """Semua sequence test -> (ids, fitur (n_seq, 30, D)). Tiap sequence diproses sendiri-sendiri."""
    test_dir = Path(test_dir)
    folders = sorted(p for p in test_dir.iterdir() if p.is_dir())
    ids, feats = [], []
    for i, f in enumerate(folders):
        feats.append(embed_frames(encoder, baca_sequence_test(f), batch=batch))
        ids.append(f.name)
        if (i + 1) % 200 == 0 or i + 1 == len(folders):
            cetak(f"  test {i + 1}/{len(folders)}")
    return ids, np.stack(feats, axis=0)


# --------------------------------------------------------------------------- fitur spasial (notebook 08)
# Fitur per frame yang MEMPERTAHANKAN LETAK: token patch DINOv2 (18 x 32 untuk input 252 x 448)
# dikumpulkan ke kisi kasar, bukan dirata-rata jadi satu angka. Detail kecil (tangan, jam,
# kertas) jadi tidak hilang. Praproses frame, seed per video, dan JPEG bolak-balik PERSIS sama
# dengan embed_frames di atas: bedanya hanya cara token patch diringkas.
TINGGI_PATCH, LEBAR_PATCH = TINGGI_INPUT // 14, LEBAR_INPUT // 14      # 18 x 32


def bentuk_spasial(encoder, grid: tuple[int, int] = (3, 8)) -> tuple[int, int]:
    """(jumlah token per frame, dimensi) = (1 + gh*gw, num_features). Token 0 = CLS."""
    gh, gw = grid
    return 1 + gh * gw, int(encoder.num_features)


def embed_frames_spatial(model, frames, device, grid: tuple[int, int] = (3, 8),
                         batch: int = 64) -> np.ndarray:
    """Frame BGR -> fitur (N, 1 + gh*gw, dim) float16, token 0 = CLS.

    Jalurnya sama dengan embed_frames (forward_features, fp16 di cuda), hanya token patch yang
    diperlakukan berbeda: [B, 18, 32, dim] -> adaptive_avg_pool2d ke `grid` -> diurutkan
    baris demi baris (sel (0,0), (0,1), ... (gh-1, gw-1)) lalu ditempel di belakang CLS.

    Karena 18 dan 32 habis dibagi grid (3,8) maupun (3,4), tiap sel merata-ratakan jumlah patch
    yang sama, jadi rata-rata semua sel = rata-rata patch global di embed_frames.
    Tiap frame diproses sendiri-sendiri oleh encoder (tidak saling memengaruhi).
    """
    import torch
    import torch.nn.functional as TF

    gh, gw = int(grid[0]), int(grid[1])
    if TINGGI_PATCH % gh or LEBAR_PATCH % gw:
        raise ValueError(f"grid {grid} tidak membagi kisi patch {TINGGI_PATCH}x{LEBAR_PATCH} "
                         "dengan rata; pakai grid yang membagi habis.")
    fp16 = bool(getattr(model, "_pakai_fp16", False)) and str(device).startswith("cuda")

    hasil = []
    with torch.inference_mode():
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(ke_input_model(frames[i:i + batch])).to(device)
            if fp16:
                x = x.half()
                with torch.autocast("cuda", dtype=torch.float16):
                    tok = model.forward_features(x)
            else:
                tok = model.forward_features(x)
            n_prefix = getattr(model, "num_prefix_tokens", 1)
            cls = tok[:, 0]                                     # [B, dim]
            patch = tok[:, n_prefix:]                           # [B, 18*32, dim]
            b, n, d = patch.shape
            if n != TINGGI_PATCH * LEBAR_PATCH:
                raise ValueError(f"dapat {n} token patch, diharap "
                                 f"{TINGGI_PATCH * LEBAR_PATCH} ({TINGGI_PATCH}x{LEBAR_PATCH})")
            kisi = patch.reshape(b, TINGGI_PATCH, LEBAR_PATCH, d).permute(0, 3, 1, 2)
            kisi = TF.adaptive_avg_pool2d(kisi.float(), (gh, gw))        # [B, dim, gh, gw]
            kisi = kisi.permute(0, 2, 3, 1).reshape(b, gh * gw, d)       # baris demi baris
            gabung = torch.cat([cls.float().unsqueeze(1), kisi], dim=1)  # [B, 1+gh*gw, dim]
            hasil.append(gabung.cpu().numpy().astype(np.float16))
    return np.concatenate(hasil, axis=0)


def ekstrak_video_train_spatial(encoder, path, device, grid=(3, 8), seed: int = 42,
                                batch: int = 64) -> np.ndarray:
    """Satu video train -> fitur spasial semua frame (n_frame, 1 + gh*gw, dim)."""
    return embed_frames_spatial(encoder, baca_video_train(path, seed), device, grid, batch)


def ekstrak_banyak_video_spatial(encoder, daftar_video, simpan_ke, device, grid=(3, 8),
                                 seed: int = 42, batch: int = 64, nama_fn=None, cetak=print):
    """Seperti ekstrak_banyak_video tapi fitur spasial.

    Hasil tiap video LANGSUNG disimpan sebagai .npy dan video yang berkasnya sudah ada
    dilewati, jadi ekstraksi bisa dilanjutkan kalau sesi terputus. Video berikutnya dibaca
    di thread terpisah selagi GPU memproses video sekarang.
    """
    simpan_ke = Path(simpan_ke)
    simpan_ke.mkdir(parents=True, exist_ok=True)
    nama_fn = nama_fn or (lambda p: Path(p).stem)
    tugas = [p for p in daftar_video if not (simpan_ke / f"{nama_fn(p)}.npy").exists()]
    cetak(f"Video: {len(daftar_video)} | sudah ada: {len(daftar_video) - len(tugas)} | "
          f"dikerjakan: {len(tugas)}")
    if not tugas:
        return
    with ThreadPoolExecutor(max_workers=1) as ex:
        depan = ex.submit(baca_video_train, tugas[0], seed)
        for i, p in enumerate(tugas):
            frames = depan.result()
            if i + 1 < len(tugas):
                depan = ex.submit(baca_video_train, tugas[i + 1], seed)
            feat = embed_frames_spatial(encoder, frames, device, grid, batch)
            sementara = simpan_ke / f"{nama_fn(p)}.npy.part"   # bukan *.npy: tidak ikut ter-glob
            with open(sementara, "wb") as fh:              # tulis ke berkas sementara lalu ganti
                np.save(fh, feat)                          # nama, supaya .npy tidak pernah separuh
            sementara.replace(simpan_ke / f"{nama_fn(p)}.npy")
            if (i + 1) % 10 == 0 or i + 1 == len(tugas):
                cetak(f"  {i + 1}/{len(tugas)} video selesai | {nama_fn(p)} -> {feat.shape}")


def ekstrak_test_spatial(encoder, test_dir, simpan_ke, device, grid=(3, 8), batch: int = 30,
                         cetak=print):
    """Semua sequence test -> (ids, array [n_seq, 30, 1+gh*gw, dim] di berkas .npy).

    Ditulis langsung ke berkas lewat memmap dan kemajuannya dicatat di <nama>.progress,
    jadi kalau sesi terputus ekstraksi lanjut dari sequence terakhir, bukan dari awal.
    Tiap sequence diproses sendiri-sendiri; tidak ada statistik lintas-sequence.
    """
    test_dir, simpan_ke = Path(test_dir), Path(simpan_ke)
    folders = sorted(p for p in test_dir.iterdir() if p.is_dir())
    ids = [p.name for p in folders]
    n_token, dim = bentuk_spasial(encoder, grid)
    bentuk = (len(folders), 30, n_token, dim)

    p_maju = simpan_ke.with_suffix(".progress")
    mulai = 0
    if simpan_ke.exists() and p_maju.exists():
        lama = np.load(simpan_ke, mmap_mode="r")
        if tuple(lama.shape) == bentuk:
            mulai = int(p_maju.read_text().strip() or 0)
            cetak(f"lanjut dari sequence {mulai}/{len(folders)}")
        else:
            cetak(f"bentuk lama {lama.shape} != {bentuk}, ekstraksi test dimulai dari awal")
        del lama

    from numpy.lib.format import open_memmap

    simpan_ke.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if mulai else "w+"
    arr = open_memmap(simpan_ke, mode=mode, dtype=np.float16, shape=bentuk) if mode == "w+" \
        else open_memmap(simpan_ke, mode="r+")
    for i in range(mulai, len(folders)):
        arr[i] = embed_frames_spatial(encoder, baca_sequence_test(folders[i]), device, grid, batch)
        if (i + 1) % 100 == 0 or i + 1 == len(folders):
            arr.flush()
            p_maju.write_text(str(i + 1))
            cetak(f"  test {i + 1}/{len(folders)}")
    arr.flush()
    p_maju.write_text(str(len(folders)))
    return ids, arr
