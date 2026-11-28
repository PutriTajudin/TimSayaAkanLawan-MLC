"""Definisi model baseline dan temporal.

Di sini juga pemuat embedding hasil notebook 03:
  <features>/emb_train/<kategori>__<subject>__<namavideo>.npy  -> [n_frame, D] float16
  <features>/emb_test.npy                                      -> [n_test, SEQ_LEN, D] float16
  <features>/emb_test_ids.txt                                   -> satu id per baris
"""
from pathlib import Path

import numpy as np
import pandas as pd


def nama_embedding(video):
    """Nama file .npy untuk kolom `video` di index ("train/<kategori>/<subject>/<nama>").
    Return daftar kandidat: dengan dan tanpa ekstensi video."""
    bagian = Path(str(video)).parts
    if len(bagian) < 3:
        raise ValueError(f"kolom video tidak sesuai dugaan: {video!r}")
    kategori, subject, nama = bagian[-3], bagian[-2], bagian[-1]
    stem = Path(nama).stem
    return [f"{kategori}__{subject}__{nama}.npy", f"{kategori}__{subject}__{stem}.npy"]


class PemuatEmbedding:
    """Baca embedding per video dari <features>/emb_train, dengan cache satu video.

    Index diurutkan per video sebelum dipakai, jadi satu slot cache sudah cukup dan
    pemakaian memori tetap kecil.
    """

    def __init__(self, dir_fitur, sub="emb_train"):
        self.dir_emb = Path(dir_fitur) / sub
        self._video = None
        self._emb = None
        self.hilang = []

    def path(self, video):
        for nama in nama_embedding(video):
            p = self.dir_emb / nama
            if p.exists():
                return p
        return None

    def ambil(self, video):
        """Embedding [n_frame, D] float32, atau None kalau file tidak ada."""
        if video == self._video:
            return self._emb
        p = self.path(video)
        if p is None:
            if video not in self.hilang:
                self.hilang.append(video)
            return None
        emb = np.load(p).astype(np.float32, copy=False)
        if emb.ndim != 2:
            raise ValueError(f"{p.name}: bentuk {emb.shape}, diharap [n_frame, D]")
        self._video, self._emb = video, emb
        return emb


def jendela(emb, mulai, seq_len=30):
    """Potongan [mulai, mulai + seq_len) dari embedding satu video.
    Return None kalau jendela melewati panjang embedding."""
    akhir = mulai + seq_len
    if mulai < 0 or akhir > len(emb):
        return None
    return emb[mulai:akhir]


def fitur_ringkas(x):
    """Ringkas waktu jadi [rata-rata, maks, simpangan baku] -> 3 x D.
    x boleh [seq_len, D] (hasil 1 x 3D) atau [n, seq_len, D] (hasil n x 3D)."""
    x = np.asarray(x, dtype=np.float32)
    satu = x.ndim == 2
    if satu:
        x = x[None]
    f = np.concatenate([x.mean(axis=1), x.max(axis=1), x.std(axis=1)], axis=1).astype(np.float32)
    return f[0] if satu else f


def fitur_dari_index(idx, dir_fitur, seq_len=30, sub="emb_train"):
    """Ambil jendela seq_len frame untuk tiap baris index, lalu ringkas jadi fitur.

    Return: (X [n_sisa, 3D] float32, index_sisa DataFrame, laporan dict).
    Baris yang jendelanya melewati panjang embedding (atau embeddingnya tidak ada) dibuang;
    jumlahnya ada di laporan: {"dibuang_lewat", "dibuang_tanpa_embedding", "video_hilang"}.
    """
    pemuat = PemuatEmbedding(dir_fitur, sub=sub)
    idx = idx.sort_values(["video", "mulai_frame"], kind="stable")
    fitur, simpan = [], []
    lewat = 0
    tanpa = 0
    for posisi, video, mulai in zip(idx.index, idx["video"], idx["mulai_frame"]):
        emb = pemuat.ambil(video)
        if emb is None:
            tanpa += 1
            continue
        w = jendela(emb, int(mulai), seq_len)
        if w is None:
            lewat += 1
            continue
        fitur.append(fitur_ringkas(w))
        simpan.append(posisi)
    if not fitur:
        raise SystemExit("Tidak ada satu pun jendela yang bisa dipakai. Cek folder emb_train.")
    X = np.stack(fitur).astype(np.float32)
    laporan = {"dibuang_lewat": lewat, "dibuang_tanpa_embedding": tanpa,
               "video_hilang": pemuat.hilang}
    return X, idx.loc[simpan].copy(), laporan


def muat_test(dir_fitur, seq_len=30):
    """Return (X [n, 3D] float32, ids list) dari emb_test.npy + emb_test_ids.txt.
    Tiap sequence test diringkas sendiri-sendiri, tanpa statistik antar-sequence."""
    dir_fitur = Path(dir_fitur)
    emb = np.load(dir_fitur / "emb_test.npy").astype(np.float32, copy=False)
    ids = [s.strip() for s in (dir_fitur / "emb_test_ids.txt").read_text().splitlines() if s.strip()]
    if emb.ndim != 3:
        raise ValueError(f"emb_test.npy bentuk {emb.shape}, diharap [n, seq_len, D]")
    if emb.shape[1] != seq_len:
        raise ValueError(f"emb_test.npy punya {emb.shape[1]} frame per sequence, diharap {seq_len}")
    if len(ids) != len(emb):
        raise ValueError(f"emb_test_ids.txt {len(ids)} id vs emb_test.npy {len(emb)} sequence")
    return fitur_ringkas(emb), ids


def tabel_prediksi(ids, prob):
    """DataFrame id/prob, satu baris per sequence test."""
    return pd.DataFrame({"id": list(ids), "prob": np.asarray(prob, dtype=float)})


# ---------------------------------------------------------------------------
# Bagian temporal (notebook 05). Butuh PyTorch; bagian di atas tidak.
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset as _Dataset

    _Module = nn.Module
except ImportError:            # venv laptop tanpa torch: bagian non-torch di atas tetap jalan
    torch = nn = None
    _Module = _Dataset = object


def _butuh_torch():
    if torch is None:
        raise ImportError("Bagian ini butuh PyTorch (ada di Kaggle, tidak di venv laptop).")


def muat_semua_embedding(dir_fitur, videos, sub="emb_train"):
    """Baca embedding seluruh video ke memori sebagai float16 (hemat, 2 byte per nilai).
    Return (dict video -> [n_frame, D], daftar video tanpa embedding)."""
    pemuat = PemuatEmbedding(dir_fitur, sub=sub)
    emb, hilang = {}, []
    for v in dict.fromkeys(videos):          # unik, urutan tetap
        p = pemuat.path(v)
        if p is None:
            hilang.append(v)
            continue
        a = np.load(p)
        if a.ndim != 2:
            raise ValueError(f"{p.name}: bentuk {a.shape}, diharap [n_frame, D]")
        emb[v] = np.ascontiguousarray(a, dtype=np.float16)
    return emb, hilang


def saring_jendela(idx, emb, seq_len=30):
    """Buang baris index yang embeddingnya tidak ada atau jendelanya melewati panjang embedding.
    Return (index_sisa, laporan {dibuang_tanpa_embedding, dibuang_lewat})."""
    panjang = np.array([len(emb[v]) if v in emb else -1 for v in idx["video"]], dtype=np.int64)
    mulai = idx["mulai_frame"].to_numpy(np.int64)
    tanpa = panjang < 0
    ok = (~tanpa) & (mulai >= 0) & (mulai + seq_len <= panjang)
    lap = {"dibuang_tanpa_embedding": int(tanpa.sum()),
           "dibuang_lewat": int((~tanpa & ~ok).sum())}
    return idx.loc[ok].reset_index(drop=True), lap


def susun_jendela(idx, emb, seq_len=30):
    """Susun jendela dari baris index jadi satu array [n, seq_len, D] float16 (tanpa torch).
    Semua baris harus sudah lolos saring_jendela."""
    if len(idx) == 0:
        raise ValueError("index kosong")
    dim = emb[idx["video"].iloc[0]].shape[1]
    keluar = np.empty((len(idx), seq_len, dim), dtype=np.float16)
    for i, (v, m) in enumerate(zip(idx["video"], idx["mulai_frame"])):
        keluar[i] = emb[v][int(m):int(m) + seq_len]
    return keluar


def muat_test_mentah(dir_fitur, seq_len=30):
    """(emb [n, seq_len, D] float16, ids) dari emb_test.npy + emb_test_ids.txt, tanpa diringkas.
    Tiap sequence test berdiri sendiri; tidak ada statistik lintas-sequence."""
    dir_fitur = Path(dir_fitur)
    emb = np.load(dir_fitur / "emb_test.npy")
    ids = [s.strip() for s in (dir_fitur / "emb_test_ids.txt").read_text().splitlines() if s.strip()]
    if emb.ndim != 3:
        raise ValueError(f"emb_test.npy bentuk {emb.shape}, diharap [n, seq_len, D]")
    if emb.shape[1] != seq_len:
        raise ValueError(f"emb_test.npy punya {emb.shape[1]} frame per sequence, diharap {seq_len}")
    if len(ids) != len(emb):
        raise ValueError(f"emb_test_ids.txt {len(ids)} id vs emb_test.npy {len(emb)} sequence")
    return np.ascontiguousarray(emb, dtype=np.float16), ids


def peta_perilaku(idx, kolom="perilaku"):
    """{nama perilaku -> nomor kelas}, urut abjad. Dibangun dari index train saja."""
    return {n: i for i, n in enumerate(sorted(idx[kolom].astype(str).unique()))}


class DataJendela(_Dataset):
    """Satu item = satu jendela seq_len frame: (x [seq_len, D] float32, y float, kelas int).

    Potongan diambil langsung dari embedding video di memori, jadi tidak perlu salinan
    sebesar n_jendela x seq_len x D.
    """

    def __init__(self, idx, emb, seq_len=30, peta=None, kolom_kelas="perilaku"):
        _butuh_torch()
        self.emb = emb
        self.seq_len = seq_len
        self.video = idx["video"].to_numpy(object)
        self.mulai = idx["mulai_frame"].to_numpy(np.int64)
        self.y = idx["label"].to_numpy(np.float32)
        if peta is None:
            self.kelas = np.zeros(len(idx), dtype=np.int64)
        else:
            self.kelas = idx[kolom_kelas].astype(str).map(peta).fillna(0).to_numpy(np.int64)

    def __len__(self):
        return len(self.mulai)

    def __getitem__(self, i):
        m = int(self.mulai[i])
        x = self.emb[self.video[i]][m:m + self.seq_len]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), self.y[i], self.kelas[i]


def augmentasi(x, noise=0.05, drop_fitur=0.1, mask_frame=0.2):
    """Augmentasi ringan, HANYA untuk batch training (jangan dipakai saat validasi/test).
      - noise Gaussian kecil, skalanya mengikuti simpangan baku jendela itu sendiri
      - dropout fitur: sebagian kanal dimatikan untuk seluruh jendela
      - mask frame acak: sampai `mask_frame` bagian frame dinolkan
    x: [B, T, D] float. Return tensor baru dengan bentuk sama."""
    _butuh_torch()
    B, T, D = x.shape
    if noise > 0:
        x = x + torch.randn_like(x) * (noise * x.std(dim=(1, 2), keepdim=True))
    if drop_fitur > 0:
        simpan = (torch.rand(B, 1, D, device=x.device) >= drop_fitur).to(x.dtype)
        x = x * simpan / (1.0 - drop_fitur)
    if mask_frame > 0:
        n_mask = torch.randint(0, int(T * mask_frame) + 1, (B, 1), device=x.device)
        peringkat = torch.rand(B, T, device=x.device).argsort(dim=1)
        x = x * (peringkat >= n_mask).to(x.dtype).unsqueeze(-1)
    return x


class PoolingAtensi(_Module):
    """Pooling berbobot: bobot tiap frame dipelajari dari isi frame itu."""

    def __init__(self, d_model, sembunyi=128):
        _butuh_torch()
        super().__init__()
        self.skor = nn.Sequential(nn.Linear(d_model, sembunyi), nn.Tanh(), nn.Linear(sembunyi, 1))

    def forward(self, h):
        bobot = torch.softmax(self.skor(h), dim=1)          # [B, T, 1]
        return (bobot * h).sum(dim=1), bobot.squeeze(-1)


class ModelTemporal(_Module):
    """Transformer kecil untuk satu sequence 30 frame embedding beku.

    LayerNorm -> Linear(768, 256) -> GELU -> Dropout -> + posisi belajar
      -> TransformerEncoder (2 layer, d_model 256, 4 head, ff 512, dropout 0.2, norm_first)
      -> [rata-rata ; pooling atensi] -> kepala biner + kepala sub-perilaku.
    Di bawah 2 juta parameter.
    """

    def __init__(self, dim_fitur=768, d_model=256, n_head=4, ff=512, n_layer=2,
                 dropout_masuk=0.3, dropout=0.2, seq_len=30, n_perilaku=0):
        _butuh_torch()
        super().__init__()
        self.norm_masuk = nn.LayerNorm(dim_fitur)
        self.proyeksi = nn.Linear(dim_fitur, d_model)
        self.aktivasi = nn.GELU()
        self.drop_masuk = nn.Dropout(dropout_masuk)
        self.posisi = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.posisi, std=0.02)
        lapis = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor=False: dengan norm_first torch tidak memakai jalur nested tensor
        # dan kalau dibiarkan True hanya memunculkan peringatan.
        self.encoder = nn.TransformerEncoder(lapis, num_layers=n_layer, enable_nested_tensor=False)
        self.norm_keluar = nn.LayerNorm(d_model)      # norm_first butuh norm di ujung
        self.pool = PoolingAtensi(d_model)
        self.kepala = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))
        self.kepala_perilaku = nn.Linear(2 * d_model, n_perilaku) if n_perilaku > 1 else None

    def forward(self, x):
        """x: [B, seq_len, dim_fitur] -> (logit [B], logit_perilaku [B, n_perilaku] atau None)."""
        h = self.drop_masuk(self.aktivasi(self.proyeksi(self.norm_masuk(x))))
        h = self.norm_keluar(self.encoder(h + self.posisi))
        rata = h.mean(dim=1)
        aten, _ = self.pool(h)
        g = torch.cat([rata, aten], dim=1)
        logit_perilaku = None if self.kepala_perilaku is None else self.kepala_perilaku(g)
        return self.kepala(g).squeeze(-1), logit_perilaku


def jumlah_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prediksi_batch(model, x, device, batch=512, amp=True):
    """Probabilitas cheat untuk x [n, seq_len, D] (numpy atau tensor).
    Selalu model.eval() + torch.no_grad(), tanpa augmentasi. Return numpy float64 [n]."""
    _butuh_torch()
    model.eval()
    keluar = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            b = x[i:i + batch]
            b = b.float() if torch.is_tensor(b) else torch.from_numpy(np.asarray(b, dtype=np.float32))
            b = b.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=bool(amp) and device.type == "cuda"):
                logit, _ = model(b)
            keluar.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(keluar).astype(np.float64)
