## Context

Skripsi ini bertujuan menjawab tiga pertanyaan utama (lihat proposal.md): perbandingan model deep learning, tren temporal sentimen & toksisitas 2016-2026, dan korelasi faktor kontekstual. Data sumber sudah tersedia di [`dota2_dataset_bersih/`](../../../dota2_dataset_bersih/), berisi `chat.csv` per-tahun (2016-2025) plus per-kuartal (202601-202604) dan `Constants/` (referensi heroes, items, chatwheel codes). Setiap folder tahunan juga memiliki `main_metadata.csv` (hasil pertandingan, durasi, league/tier, patch, region) yang akan menjadi sumber fitur kontekstual.

Tim peneliti adalah tiga mahasiswa S1 Teknik Informatika BINUS (Daniel, Dhitan, Aldiaz) — bekerja di lingkungan lokal Windows (CUDA-capable) dengan akses Colab/Kaggle untuk training. Output akhir: laporan skripsi BAB IV (hasil & pembahasan) dan ekspektasinya artikel ilmiah turunan. Stakeholder: pembimbing skripsi, penguji, dan komunitas peneliti NLP/esports.

Kendala:
- **Volume data**: estimasi puluhan juta baris chat lintas dekade — pipeline harus stream-friendly dan caching di parquet.
- **Kualitas label**: tidak ada gold-standard publik untuk chat MOBA — harus diproduksi sendiri (anotasi manual stratified).
- **Compute**: 4 model deep learning × 2 tugas (sentimen, toksisitas) × inferensi puluhan juta baris membutuhkan batching dan checkpointing yang disiplin.
- **Reproducibility**: skripsi membutuhkan jejak audit yang kuat (seed, versi model, hash data) agar penguji bisa memverifikasi.

## Goals / Non-Goals

**Goals:**
- Pipeline end-to-end yang reproducible dari raw `chat.csv` → laporan BAB IV.
- Perbandingan apple-to-apple antar 4 model deep learning di gold-standard set yang sama, dengan metrik akurasi/presisi/recall/F1 macro & per-class.
- Analisis temporal granularitas tahun & bulan untuk dekade 2016-2026, dengan event-overlay (patch, COVID-19, format DPC).
- Analisis korelasi faktor kontekstual dengan uji statistik yang sesuai dan koreksi multiple-testing.
- Output siap-kutip: tabel CSV, plot PNG/SVG, dan ringkasan markdown yang bisa langsung diambil ke laporan skripsi.

**Non-Goals:**
- TIDAK mengembangkan arsitektur model NLP baru — hanya menerapkan model yang sudah ada (proposal eksplisit, BAB I §1.5).
- TIDAK menganalisis voice chat, emote, atau perintah non-verbal (proposal §1.5).
- TIDAK menganalisis pertandingan public matchmaking (proposal §1.5) — hanya turnamen profesional yang sudah ada di `dota2_dataset_bersih/`.
- TIDAK mengembangkan model untuk bahasa selain Inggris — teks non-Inggris difilter keluar.
- TIDAK menerapkan SVM/Naive Bayes/LogReg/VADER/TextBlob — sesuai keputusan user, fokus penuh pada 4 model deep learning.
- TIDAK menyediakan deployment moderasi real-time — output adalah analisis offline untuk skripsi.

## Decisions

### 1. Empat model deep learning: BERT-base + RoBERTa-base + DistilBERT + Detoxify (toxic-bert)

**Pilihan:** BERT-base-uncased (baseline klasik), RoBERTa-base (improved pretraining), DistilBERT (efficiency reference, 60% faster, 40% smaller), Detoxify / unitary/toxic-bert (specialist multi-label toksisitas dengan label: toxic, severe_toxic, obscene, threat, insult, identity_hate).

**Alternatif yang dipertimbangkan:**
- BERTweet (pretrained di Twitter) — dipertimbangkan untuk informalitas/jargon, ditolak supaya line-up tetap 4 model dan ada specialist toksisitas.
- DeBERTa-v3 — performa SOTA tapi training cost lebih berat, ditolak demi feasibility skripsi.
- HateBERT — alternatif specialist toksisitas; Detoxify dipilih karena multi-label (lebih kaya untuk RQ tentang kategori toksisitas).

**Rasional:** Lineup mencakup (a) baseline standar (BERT), (b) improved general-purpose (RoBERTa), (c) efficient reference (DistilBERT) supaya bisa membahas trade-off speed vs accuracy di skripsi, dan (d) specialist multi-label toxicity (Detoxify) untuk menjawab RQ kategori toksisitas. Kombinasi ini memungkinkan pembahasan: apakah specialist (Detoxify) mengalahkan generalis fine-tuned di domain spesifik, dan apakah model ringan (DistilBERT) cukup untuk produksi moderasi.

### 2. Strategi label: transfer learning + gold-standard manual evaluation

**Pilihan:** Fine-tune setiap model di dataset publik berlabel — Sentiment140 atau SST-2 untuk sentimen 3-class, Jigsaw Toxic Comment Classification + HateEval untuk toksisitas multi-label — lalu inferensi ke seluruh chat Dota 2. Evaluasi dilakukan di **gold-standard set** hasil anotasi manual stratified (5-10k pesan, lihat keputusan #4).

**Alternatif yang dipertimbangkan:**
- Pure inference tanpa fine-tune (zero-shot pretrained checkpoint) — ditolak karena domain gap chat MOBA terhadap web/Twitter terlalu besar.
- Fine-tune di chat Dota 2 berlabel manual — ditolak karena membutuhkan anotasi 50-100k baris untuk training transformer dari awal, di luar bandwidth tim 3 mahasiswa.

**Rasional:** Transfer learning kompromi terbaik antara akurasi dan effort. Gold-standard manual tetap diperlukan untuk evaluasi (jawaban RQ1) dan untuk audit kualitas anotasi (Cohen's kappa).

### 3. Pipeline data: parquet-first dengan layered staging

**Pilihan:** Tiga tingkat penyimpanan:
- `data/raw/` — read-only symlink/reference ke `dota2_dataset_bersih/` (tidak duplikasi).
- `data/processed/` — chat per-tahun yang sudah di-clean & di-merge dengan metadata, format **parquet** (kolom-kompresi, schema stabil).
- `data/gold/` — hasil anotasi manual + guideline.

Cleaning steps: drop chatwheel (`type == "chatwheel"`), normalisasi unicode, lowercasing untuk model uncased, language detection (`fasttext-langid` ringan vs `langdetect` — fasttext dipilih karena lebih akurat untuk teks pendek), filter ke `lang == "en"`, hash de-duplication exact-match per match_id.

**Rasional:** Parquet jauh lebih cepat dibaca berulang untuk eksperimen iteratif dibanding CSV. Layering memisahkan immutable raw dari turunan yang bisa di-rebuild.

### 4. Anotasi manual: dual-axis, stratified, dual-rater dengan adjudication

**Pilihan:**
- **Dimensi**: dual-axis — (a) sentiment 3-class (positive / neutral / negative), (b) toxicity multi-label (toxic, obscene, threat, insult, identity_hate; sesuai skema Detoxify supaya comparable).
- **Sample**: 8.000 pesan stratified per (tahun × hasil pertandingan menang/kalah × tier turnamen) supaya tiap stratum >= 50 pesan.
- **Anotator**: 3 mahasiswa peneliti, masing-masing menganotasi 100% sample (overlap penuh untuk inter-annotator agreement) atau dua-dari-tiga overlap dengan adjudikasi pihak ketiga jika bandwidth terbatas.
- **Tooling**: spreadsheet berbasis CSV dengan validasi schema, atau opsional Doccano/Label Studio lokal jika tim familiar.
- **Agreement**: Cohen's kappa per-pair untuk sentimen, Krippendorff's alpha untuk toxicity multi-label. Threshold rilis: kappa >= 0.6.

**Alternatif yang dipertimbangkan:**
- Single-rater 5k — ditolak karena tidak bisa melaporkan reliability.
- Pakai pseudo-label dari Detoxify sebagai gold — ditolak karena evaluasi jadi sirkular.

**Rasional:** 8k sample cukup untuk metrik stabil (CI < ±2% di F1) dan masih realistis untuk 3 anotator dalam 2-3 minggu. Dual-rater + adjudikasi standar industri.

### 5. Evaluasi: matched test set + cross-tugas reporting

**Pilihan:**
- Split gold: 70% train-validation (jika ada fine-tuning Dota), 30% test. Test set tidak pernah disentuh sampai laporan akhir.
- Metrik **sentimen**: akurasi, F1-macro, F1 per-class, confusion matrix, ROC-AUC OVR.
- Metrik **toksisitas multi-label**: F1-micro & F1-macro per-label, Hamming loss, Subset accuracy, Average Precision.
- Setiap model dilaporkan di **kedua tugas** ketika applicable (BERT/RoBERTa/DistilBERT fine-tuned untuk masing-masing; Detoxify hanya toksisitas, sentimen dilaporkan via Detoxify-toxic-bert pada tugas sentimen sebagai zero-shot reference).
- Bootstrap 95% CI di setiap metrik (1000 resample) untuk significance testing.

**Rasional:** Multi-label toksisitas membutuhkan metrik yang berbeda dari klasifikasi 3-class sentimen. CI mencegah klaim signifikansi yang lemah.

### 6. Analisis temporal: time-series resampling + event overlay

**Pilihan:**
- Resample per-bulan (granularitas utama) dan per-tahun (ringkasan).
- Metrik agregat per-bin: % positif/negatif, rata-rata `sentiment_score`, % toksik (any-label), rata-rata `toxicity_score`.
- Event overlay (vertikal-line di plot):
  - Pergantian patch major (ambil dari kolom `patch` di metadata; kelompokkan patch besar 7.00, 7.20, 7.30, 7.33, 7.34+).
  - Era pandemi COVID-19 daring (2020-03 sd 2021-12 sebagai band).
  - The International tahunan (sebagai marker).
  - Format DPC era (2017-2022) vs post-DPC (2023+).
- Uji tren: Mann-Kendall trend test, structural break test (Chow test) di breakpoints kandidat.

**Rasional:** Memungkinkan pembahasan kualitatif (BAB IV) dengan dukungan kuantitatif yang formal.

### 7. Analisis korelasi: uji yang sesuai per-tipe variabel + koreksi FDR

**Pilihan:**
- **Hasil pertandingan (menang/kalah, biner)** vs sentimen (categorical) → chi-square; vs toksisitas (continuous score) → Mann-Whitney U.
- **Durasi pertandingan (continuous)** vs sentimen score / toxicity score → Spearman rank correlation (robust ke non-normal).
- **Tier turnamen (ordinal: TI > Major > DPC Tour > Lainnya)** → Kruskal-Wallis lintas grup; pairwise dengan Dunn's test.
- **Fase turnamen (group stage / playoffs / grand final, kategorikal)** → Kruskal-Wallis.
- **Periode waktu (kategorikal: 2016-2018 / 2019-2021 / 2022-2026, atau yang relevan)** → Kruskal-Wallis.
- Koreksi multiple-testing: **Benjamini-Hochberg FDR** dengan q = 0.05 di seluruh suite uji.

**Rasional:** Uji parametrik (t-test/Pearson) tidak cocok karena distribusi metrik per-pertandingan biasanya skewed. FDR > Bonferroni karena suite uji moderat (~20-30 uji).

### 8. Reproducibility: manifest + frozen seeds + Hugging Face artifact pinning

**Pilihan:**
- File `configs/experiment.yaml` berisi: `seed`, `transformers_version`, `torch_version`, `model_checkpoints` (revision SHA), `dataset_versions` (URL + SHA256 file), `data_processed_hash` (SHA256 parquet output).
- Notebook eksekusi bernomor 01-07 supaya urutan deterministik (lihat tasks.md).
- Notebook menampilkan banner versi paket + seed di awal.
- Hasil eksperimen disimpan ke `reports/` dengan timestamp + commit hash.

**Rasional:** Penguji skripsi mungkin mau menjalankan ulang sebagian eksperimen; manifest membuat ini feasible.

## Risks / Trade-offs

- **[Bias pseudo-label dari Detoxify ketika dipakai sebagai pesaing di gold-test]** → Mitigasi: gold-standard set diannotasi manual sebelum melihat output Detoxify; Detoxify dievaluasi seperti model lain di test set yang sama.
- **[Anotasi manual 8k pesan oleh 3 anotator membutuhkan 2-3 minggu]** → Mitigasi: mulai anotasi paralel dengan tahap pipeline data; siapkan guideline anotasi konkret + 200 pesan pilot untuk kalibrasi sebelum produksi.
- **[Compute fine-tuning 4 model di Sentiment140 (~1.6 juta) bisa menghabiskan kuota Colab gratis]** → Mitigasi: subset Sentiment140 ke 200k stratified, atau gunakan SST-2 (67k) untuk efisiensi; gunakan mixed-precision (fp16) dan batch size adaptif.
- **[Volume inferensi dari puluhan juta baris chat]** → Mitigasi: batched GPU inference (batch 64-128), simpan probabilitas per-pesan ke parquet, checkpoint per-folder tahun. Jika overheating, sample turun ke 1-2 juta pesan stratified untuk analisis temporal (cukup untuk power statistik).
- **[Distribusi label sangat tidak seimbang — toksisitas minoritas]** → Mitigasi: laporkan F1-macro & per-class (bukan akurasi), pertimbangkan class-weighted loss saat fine-tune, sampling bootstrap CI.
- **[Bahasa Inggris bercampur jargon Dota (gg, mid, jungle, fed) yang tidak ada di dataset publik]** → Mitigasi: tambah leksikon jargon Dota ke pre-tokenization map; analisis error per-jargon di laporan.
- **[Language detection meleset di chat sangat pendek (`gg`, `wp`)]** → Mitigasi: short-circuit allowlist untuk pesan <= 5 token yang seluruhnya ASCII alfanumerik (anggap Inggris); evaluasi di sample manual.
- **[Patch labels di metadata berformat `54`, `52` — bukan versi human-readable]** → Mitigasi: bangun mapping `patch_id → patch_name → release_date` di `Constants/` (cross-check sumber Liquipedia/Dotabuff).

## Migration Plan

Tidak ada data migration sense-tradisional — perubahan ini menambah pipeline dan artefak baru tanpa mengubah data sumber. Roll-back = hapus `notebooks/0[1-7]_*.ipynb`, `src/`, `data/processed/`, `data/gold/`, `models/`, `reports/`, `configs/experiment.yaml`. File raw `dota2_dataset_bersih/` dan import scripts/notebooks tidak terdampak.

## Open Questions

- Apakah anotasi penuh 100% overlap (3 anotator × 8k pesan) feasible, atau pakai mode 2-of-3 dengan adjudikasi? Keputusan: mulai dengan pilot 200 pesan full-overlap, evaluasi waktu, lalu finalisasi mode produksi.
- Apakah subset Sentiment140 200k atau full SST-2 yang dipakai untuk fine-tune sentimen? Keputusan: A/B kecil di 10% sample, pilih yang akurasi validation lebih baik.
- Untuk era post-DPC (2024-2026 dengan turnamen non-DPC seperti Esports World Cup, Riyadh Masters), apakah dimasukkan? Keputusan: ya jika ada di `dota2_dataset_bersih/`, dengan tier label "non-DPC-major".
- Apakah anonimisasi `account_id`/`player_slot` dilakukan sejak processed atau hanya di reports publik? Keputusan: pertahankan ID di processed (dibutuhkan untuk join), redact hanya di reports/skripsi.
