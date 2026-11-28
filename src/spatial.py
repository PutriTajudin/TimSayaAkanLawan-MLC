"""Model temporal untuk fitur SPASIAL hasil notebook 08 (notebook 09).

Masukan per video: [T, n_sel, dim_token] float16, token 0 = CLS dan token 1.. = sel kisi
(3x8, baris demi baris). Jendela 30 frame TIDAK diwujudkan semuanya sekaligus (bisa ~22 GB):
array per video dibuka memmap dan irisannya diambil saat batch disusun.

Yang dipakai ulang dari src/models.py tanpa diubah: peta_perilaku, saring_jendela,
PemuatEmbedding/nama_embedding (pencarian berkas), ModelMultitask (seluruh bagian temporal
dan ketiga kepalanya), dan prediksi_batch (inferensi test).
"""
import numpy as np

from .models import DataJendela, PemuatEmbedding, _butuh_torch, _Module, nn, torch
from .multitask import ModelMultitask


# --------------------------------------------------------------------------- data
def muat_embedding_spasial(dir_fitur, videos, sub="emb_sp_train", mmap=True):
    """{video -> array [T, n_sel, dim]} + daftar video tanpa embedding.

    mmap=True (bawaan): array dibuka read-only lewat memmap, jadi yang masuk memori hanya
    irisan yang benar-benar dipakai. Penting karena seluruh fitur train bisa beberapa GB.
    """
    pemuat = PemuatEmbedding(dir_fitur, sub=sub)
    emb, hilang = {}, []
    for v in dict.fromkeys(videos):
        p = pemuat.path(v)
        if p is None:
            hilang.append(v)
            continue
        a = np.load(p, mmap_mode="r") if mmap else np.load(p)
        if a.ndim != 3:
            raise ValueError(f"{p.name}: bentuk {a.shape}, diharap [T, n_sel, dim]")
        emb[v] = a
    return emb, hilang


def bentuk_sel(emb):
    """(n_sel, dim) dari salah satu video; sekaligus memastikan semuanya seragam."""
    bentuk = {tuple(a.shape[1:]) for a in emb.values()}
    if len(bentuk) != 1:
        raise ValueError(f"bentuk sel tidak seragam antar video: {bentuk}")
    n_sel, dim = bentuk.pop()
    return int(n_sel), int(dim)


def susun_jendela_sp(idx, emb, seq_len=30):
    """Susun jendela jadi satu array [n, seq_len, n_sel, dim] float16.

    Hanya untuk kumpulan KECIL (uji / potongan). Untuk validasi dan test pakai
    prediksi_jendela / prediksi_batch supaya tidak ada array raksasa.
    """
    if len(idx) == 0:
        raise ValueError("index kosong")
    n_sel, dim = bentuk_sel({v: emb[v] for v in idx["video"].iloc[:1]})
    keluar = np.empty((len(idx), seq_len, n_sel, dim), dtype=np.float16)
    for i, (v, m) in enumerate(zip(idx["video"], idx["mulai_frame"])):
        keluar[i] = emb[v][int(m):int(m) + seq_len]
    return keluar


def muat_test_spasial(dir_fitur, seq_len=30, mmap=True):
    """(array [n, seq_len, n_sel, dim], ids) dari emb_sp_test.npy + emb_sp_test_ids.txt.

    Tiap sequence test berdiri sendiri; tidak ada statistik lintas-sequence.
    """
    from pathlib import Path

    dir_fitur = Path(dir_fitur)
    p = dir_fitur / "emb_sp_test.npy"
    emb = np.load(p, mmap_mode="r") if mmap else np.load(p)
    ids = [s.strip() for s in (dir_fitur / "emb_sp_test_ids.txt").read_text().splitlines() if s.strip()]
    if emb.ndim != 4:
        raise ValueError(f"emb_sp_test.npy bentuk {emb.shape}, diharap [n, seq_len, n_sel, dim]")
    if emb.shape[1] != seq_len:
        raise ValueError(f"emb_sp_test.npy punya {emb.shape[1]} frame/sequence, diharap {seq_len}")
    if len(ids) != len(emb):
        raise ValueError(f"emb_sp_test_ids.txt {len(ids)} id vs {len(emb)} sequence")
    return emb, ids


class DataJendelaSpasial(DataJendela):
    """Satu item = (x [seq_len, n_sel, dim] float32, y, kelas_perilaku, kelas_sudut).

    Irisan diambil langsung dari array per video (memmap), jadi tidak pernah ada salinan
    sebesar n_jendela x seq_len x n_sel x dim.
    """

    def __init__(self, idx, emb, seq_len=30, peta=None, peta_sudut=None,
                 kolom_kelas="perilaku", kolom_sudut="sudut"):
        super().__init__(idx, emb, seq_len=seq_len, peta=peta, kolom_kelas=kolom_kelas)
        if peta_sudut is None:
            self.kelas_sudut = np.zeros(len(idx), dtype=np.int64)
        else:
            self.kelas_sudut = (idx[kolom_sudut].astype(str).map(peta_sudut)
                                .fillna(0).to_numpy(np.int64))

    def __getitem__(self, i):
        m = int(self.mulai[i])
        x = self.emb[self.video[i]][m:m + self.seq_len]
        return (torch.from_numpy(np.asarray(x, dtype=np.float32)), self.y[i],
                self.kelas[i], self.kelas_sudut[i])


def augmentasi_sp(x, noise=0.05, mask_frame=0.2):
    """Augmentasi batch training untuk fitur spasial [B, T, S, D] (jangan di validasi/test).

    Dropout sel ada di dalam ModelSpasial (supaya skor perhatian ikut dimask), jadi di sini
    hanya noise Gaussian kecil dan mask frame acak.
    """
    _butuh_torch()
    b, t = x.shape[0], x.shape[1]
    if noise > 0:
        x = x + torch.randn_like(x) * (noise * x.std(dim=(1, 2, 3), keepdim=True))
    if mask_frame > 0:
        n_mask = torch.randint(0, int(t * mask_frame) + 1, (b, 1), device=x.device)
        peringkat = torch.rand(b, t, device=x.device).argsort(dim=1)
        x = x * (peringkat >= n_mask).to(x.dtype)[:, :, None, None]
    return x


# --------------------------------------------------------------------------- model
class ModelSpasial(_Module):
    """Fitur spasial per frame -> satu probabilitas cheat per jendela 30 frame.

    Depan (baru):  LayerNorm(dim_token) -> Linear(dim_token, d_sel) bersama untuk tiap token
                   -> + embedding posisi sel (n_sel x d_sel)
                   -> dropout sel acak saat latih (skor perhatian ikut dimask)
                   -> pooling perhatian per frame dengan `n_query` query belajar
                   -> vektor frame [n_query * d_sel]
    Belakang:      ModelMultitask dari src/models.py apa adanya -> Linear ke d_model, posisi
                   waktu belajar, TransformerEncoder 2 layer, rata-rata + atensi, lalu kepala
                   cheat / perilaku / sudut.

    forward -> (logit_cheat [B], {"perilaku": ..., "sudut": ...}), bentuk yang sama dengan
    ModelMultitask, jadi prediksi_batch di src/models.py bisa dipakai apa adanya.
    """

    def __init__(self, dim_token=384, n_sel=25, d_sel=128, n_query=4, d_model=256, seq_len=30,
                 n_perilaku=0, n_sudut=0, dropout_sel=0.25, dropout_masuk=0.3, dropout=0.2,
                 n_head=4, ff=512, n_layer=2):
        _butuh_torch()
        super().__init__()
        if not 0.0 <= dropout_sel < 1.0:
            raise ValueError(f"dropout_sel harus di [0, 1), dapat {dropout_sel}")
        self.n_sel, self.d_sel, self.n_query = int(n_sel), int(d_sel), int(n_query)
        self.dropout_sel = float(dropout_sel)
        self.norm_token = nn.LayerNorm(dim_token)
        self.proy_token = nn.Linear(dim_token, d_sel)
        self.pos_sel = nn.Parameter(torch.zeros(1, 1, n_sel, d_sel))
        nn.init.trunc_normal_(self.pos_sel, std=0.02)
        self.query = nn.Parameter(torch.zeros(n_query, d_sel))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.drop_frame = nn.Dropout(dropout)
        self.temporal = ModelMultitask(
            dim_fitur=n_query * d_sel, d_model=d_model, n_head=n_head, ff=ff, n_layer=n_layer,
            dropout_masuk=dropout_masuk, dropout=dropout, seq_len=seq_len,
            n_perilaku=n_perilaku, n_sudut=n_sudut)

    def vektor_frame(self, x):
        """x [B, T, S, dim_token] -> [B, T, n_query * d_sel]."""
        h = self.proy_token(self.norm_token(x)) + self.pos_sel          # [B, T, S, d_sel]
        skor = torch.einsum("qd,btsd->btqs", self.query, h) * (self.d_sel ** -0.5)
        if self.training and self.dropout_sel > 0:
            simpan = torch.rand(h.shape[0], h.shape[1], 1, h.shape[2], device=h.device) >= self.dropout_sel
            simpan[..., 0] = True          # CLS selalu disimpan -> tidak mungkin semua sel mati
            skor = skor.masked_fill(~simpan, float("-inf"))
        bobot = torch.softmax(skor, dim=-1)                             # [B, T, n_query, S]
        frame = torch.einsum("btqs,btsd->btqd", bobot, h)
        return self.drop_frame(frame.flatten(2))

    def forward(self, x):
        return self.temporal(self.vektor_frame(x))


# --------------------------------------------------------------------------- inferensi
def prediksi_jendela(model, idx, emb, device, seq_len=30, batch=64, amp=True):
    """Probabilitas cheat untuk tiap baris `idx`, batch disusun dari memmap per video.

    Selalu model.eval() + torch.no_grad(), tanpa augmentasi, urutan keluaran = urutan idx.
    Dipakai untuk validasi supaya tidak perlu mewujudkan semua jendela sekaligus.
    """
    _butuh_torch()
    model.eval()
    video = idx["video"].to_numpy(object)
    mulai = idx["mulai_frame"].to_numpy(np.int64)
    keluar = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            potong = [np.asarray(emb[video[j]][int(mulai[j]):int(mulai[j]) + seq_len],
                                 dtype=np.float32)
                      for j in range(i, min(i + batch, len(idx)))]
            b = torch.from_numpy(np.stack(potong)).to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=bool(amp) and device.type == "cuda"):
                logit, _ = model(b)
            keluar.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(keluar).astype(np.float64)
