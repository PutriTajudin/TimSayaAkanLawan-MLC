"""Model temporal multi-task (notebook 07): kepala cheat + dua kepala bantu.

Dipisah dari src/models.py supaya notebook 05 dan berkas lama tidak ikut berubah.
Label bantu (perilaku dan sudut kamera) berasal dari nama berkas video di index train,
bukan dari data test. Keluaran yang dipakai untuk submission tetap HANYA kepala cheat.
"""
import numpy as np

from .models import DataJendela, ModelTemporal, nn, torch


class ModelMultitask(ModelTemporal):
    """ModelTemporal ditambah kepala sudut kamera.

    forward -> (logit_cheat [B], {"perilaku": [B, Kp] atau None, "sudut": [B, Ks] atau None}).
    Bentuk (logit, aux) ini sengaja dipertahankan supaya prediksi_batch di src/models.py
    (yang memakai `logit, _ = model(x)`) bisa dipakai apa adanya untuk inferensi test.
    """

    def __init__(self, *a, n_sudut=0, **k):
        super().__init__(*a, **k)
        d_model = self.posisi.shape[-1]
        self.kepala_sudut = nn.Linear(2 * d_model, n_sudut) if n_sudut > 1 else None

    def _badan(self, x):
        """Badan yang sama dengan ModelTemporal.forward -> gabungan [rata-rata ; atensi]."""
        h = self.drop_masuk(self.aktivasi(self.proyeksi(self.norm_masuk(x))))
        h = self.norm_keluar(self.encoder(h + self.posisi))
        return torch.cat([h.mean(dim=1), self.pool(h)[0]], dim=1)

    def forward(self, x):
        g = self._badan(x)
        aux = {
            "perilaku": None if self.kepala_perilaku is None else self.kepala_perilaku(g),
            "sudut": None if self.kepala_sudut is None else self.kepala_sudut(g),
        }
        return self.kepala(g).squeeze(-1), aux


class DataJendelaMultitask(DataJendela):
    """Satu item = (x [seq_len, D] float32, y float, kelas_perilaku int, kelas_sudut int)."""

    def __init__(self, idx, emb, seq_len=30, peta=None, peta_sudut=None,
                 kolom_kelas="perilaku", kolom_sudut="sudut"):
        super().__init__(idx, emb, seq_len=seq_len, peta=peta, kolom_kelas=kolom_kelas)
        if peta_sudut is None:
            self.kelas_sudut = np.zeros(len(idx), dtype=np.int64)
        else:
            self.kelas_sudut = (idx[kolom_sudut].astype(str).map(peta_sudut)
                                .fillna(0).to_numpy(np.int64))

    def __getitem__(self, i):
        x, y, kelas = super().__getitem__(i)
        return x, y, kelas, self.kelas_sudut[i]
