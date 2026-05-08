# Keputusan Tooling Anotasi

**Status**: TBD — diisi tim peneliti setelah evaluasi pilot.
**Anggota**: Daniel, Dhitan, Aldiaz.

## Opsi A — Google Sheets / Excel (DEFAULT)

**Cara kerja**:
1. Upload `data/gold/sample.csv` ke Google Sheets bersama, satu shared workbook.
2. Buat 3 sheet (atau 3 file terpisah): `sample_DD`, `sample_DI`, `sample_AK`.
3. Setiap anotator isi kolom anotasi langsung di sheet-nya (gunakan data validation untuk dropdown sentimen + 0/1 untuk toxicity).
4. Setelah selesai, export tiap sheet ke `data/gold/raw_annotations/<initial>.csv`.

**Keunggulan**:
- Zero learning curve, semua anotator familiar.
- Realtime collaboration, mudah audit.
- Data validation built-in (dropdown via "Data > Validation").
- Sortir/filter cepat.

**Kekurangan**:
- Tidak ada UI khusus anotasi — anotator bisa kelelahan dengan navigasi cell.
- Tidak otomatis hitung agreement; harus eksport CSV dulu.
- Risiko typo (mis. "negatve" vs "negative") — mitigasi via data validation.

**Setup data validation di Sheets**:
- Kolom `sentiment`: list = `negative,neutral,positive`.
- Kolom `tox_*`: list = `0,1`.
- Kolom `is_dota_jargon`, `is_ambiguous`: list = `0,1`.

## Opsi B — Doccano (lokal, open-source)

**Cara kerja**:
1. Install: `pip install doccano && doccano init`.
2. Import `sample.csv` sebagai sequence labeling task (atau document classification + multi-label task).
3. Anotator login terpisah dengan akun masing-masing.
4. Export anotasi → JSONL → script konversi ke `raw_annotations/<initial>.csv`.

**Keunggulan**:
- UI khusus anotasi, fokus distraction-free.
- Multi-label toxicity native didukung.
- Keyboard shortcuts cepat.
- Server akun multi-user.

**Kekurangan**:
- Setup awal: install + run server lokal di salah satu mesin tim (atau Heroku/Railway).
- Anotator harus akses ke server (LAN atau VPN) — bisa tricky kalau anotator remote.
- Skema dual-axis (sentiment 3-class + 6 toxicity binary) butuh dua project Doccano terpisah, atau one project dengan custom label set.

## Opsi C — Label Studio (lokal/cloud)

Mirip Doccano dengan UI lebih kaya. `pip install label-studio && label-studio start`. Lebih powerful untuk multi-axis labeling tapi setup lebih berat.

---

## Rekomendasi (sebelum pilot)

**Default**: Opsi A (Google Sheets) untuk pilot 200 pesan. Alasan: setup instan, ketiga anotator bisa langsung mulai. Jika setelah pilot kappa < 0.6 dan dirasa karena UI Sheets terlalu menyilaukan / banyak typo → migrasi ke Opsi B (Doccano) untuk produksi 8.000 pesan.

## Keputusan Final

**Tooling produksi**: <TBD — diisi setelah pilot>
**Alasan**: <TBD>
**Tanggal keputusan**: <TBD>

## Path file anotasi

Output dari tooling apapun MUST didistilasi ke skema CSV ini di `data/gold/raw_annotations/`:

```
data/gold/raw_annotations/
├── sample_DD.csv   # anotasi Daniel (kolom: match_id, time, player_slot, sentiment, tox_*, is_*, notes)
├── sample_DI.csv   # anotasi Dhitan
└── sample_AK.csv   # anotasi Aldiaz
```

Notebook `03_annotation_review.ipynb` membaca tiga file ini untuk konsolidasi.
