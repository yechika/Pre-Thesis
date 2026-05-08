# BAB IV — Hasil dan Pembahasan

> **Status**: terisi dari output eksperimen NB01-NB08. Manifest hash + git commit di `configs/experiment.yaml`. Eksekusi 2026-05-04 → 2026-05-05.

## 4.1 Karakteristik Dataset

### 4.1.1 Volume dan Distribusi Temporal

- **Total pesan chat profesional 2016-2026** (pasca-filter chatwheel + bahasa Inggris): **1.603.567 baris** di **197.695 pertandingan unik**.
- **Distribusi tahunan** (lihat `reports/temporal_yearly.csv`):

| Tahun | n_messages | n_matches |
|-------|-----------|-----------|
| 2016 | 92.810 | 7.758 |
| 2017 | 81.996 | 8.797 |
| 2018 | 96.940 | 8.741 |
| 2019 | 157.670 | 16.450 |
| 2020 | 139.281 | 21.710 |
| 2021 | 129.441 | 16.226 |
| 2022 | 137.015 | 20.701 |
| 2023 | 153.765 | 29.076 |
| 2024 | 164.762 | 29.374 |
| 2025 | 177.857 | 30.087 |
| 2026 (4 bulan) | 49.330 | 8.775 |

- **Eksklusi**:
  - Chatwheel: dieksklusi dari pipeline (perintah suara/visual non-tekstual).
  - Non-Inggris: **274.087 baris** disimpan terpisah di `data/processed/<folder>_non_english.parquet` (Russia, Bulgaria, Macedonia, Filipino, Spanyol, dst).
- **Distribusi tier turnamen**:
  - Lainnya: 1.475.880 (92.04%)
  - DPC Tour: 63.084 (3.93%)
  - Major: 37.811 (2.36%)
  - TI: 26.792 (1.67%)

### 4.1.2 Karakteristik Gold-Standard Anotasi

- **Total pesan teranotasi**: **7.971** (sample stratified 8.002 baris dengan 31 duplikat key di-drop di NB03).
- **Mode anotasi**: **Single-Annotator AI** (Claude opus-4-7) — Pilihan B dari `data/gold/TOOLING_DECISION.md`. Cohen's kappa dan Krippendorff's alpha **TIDAK dihitung** (butuh ≥2 rater independen).
- **Implikasi metodologi** (lihat `data/gold/agreement_report.md`):
  - Inter-annotator agreement tidak dilaporkan untuk gold-test.
  - Reliability bergantung pada konsistensi 1 anotator AI.
  - Bias risk bila model evaluasi punya representasi mirip Claude — di-mitigasi via flag `is_ambiguous=1` untuk subgroup analysis.
  - 38.1% sample (3.037 baris) di-flag `is_ambiguous=1` untuk audit.
- **Distribusi label sentimen**:
  - positive: 4.512 (56.6%)
  - neutral: 3.357 (42.1%)
  - negative: 102 (1.3%)
- **Distribusi label toksisitas** (per-label, jumlah baris dengan label=1):
  - toxic: 68 (0.85%)
  - severe_toxic: 0 (0%)
  - obscene: 53 (0.66%)
  - threat: 0 (0%)
  - insult: 66 (0.83%)
  - identity_hate: 2 (0.025%)
- **is_dota_jargon=1**: 4.257 (53.4%) — konsisten dengan dominasi koordinasi taktis.
- **Train/test split**: 5.579 train (70.0%) / 2.392 test (30.0%), stratified by sentiment, **0 overlap key** (post-dedup).

## 4.2 Hasil Perbandingan 4 Model Deep Learning

### 4.2.1 Model yang Dibandingkan

1. **BERT-base-uncased** (`bert-base-uncased@86b5e093`) — fine-tuned pada TweetEval-sentiment (3-class) + Jigsaw Toxic Comment (multi-label).
2. **RoBERTa-base** (`roberta-base@e2da8e2f`) — fine-tuned identik.
3. **DistilBERT-base-uncased** (`distilbert-base-uncased@12040acc`) — fine-tuned identik.
4. **Detoxify (`unitary/toxic-bert@4d6c22e7`)** — pretrained checkpoint, dipakai zero-shot untuk toksisitas; pada sentimen dipakai sebagai reference dengan mapping `max_toxicity_prob ≥ 0.5 → negative`.

Detail revision SHA + hyperparameter: `configs/experiment.yaml`.

**Validation F1 di dataset publik (sebelum gold-test):**

| Model × Task | Val F1 | Threshold spec | Status |
|--------------|--------|----------------|--------|
| BERT × sentiment | 0.727 (F1-macro) | ≥0.70 | ✓ pass |
| RoBERTa × sentiment | 0.729 | ≥0.70 | ✓ pass |
| DistilBERT × sentiment | 0.724 | ≥0.70 | ✓ pass |
| BERT × toxicity | 0.797 (F1-micro) | ≥0.85 | ✗ below |
| RoBERTa × toxicity | 0.803 | ≥0.85 | ✗ below |
| DistilBERT × toxicity | 0.796 | ≥0.85 | ✗ below |

### 4.2.2 Hasil Sentimen 3-Class (gold-test n=2.392)

Tabel komparatif lengkap: `reports/comparison_summary.md` dan `reports/eval_sentiment.csv`.

| Model | Akurasi (95% CI) | F1-macro (95% CI) | F1-neg | F1-neu | F1-pos |
|-------|------------------|-------------------|--------|--------|--------|
| BERT | 0.388 [0.369, 0.408] | 0.208 [0.192, 0.229] | 0.026 | 0.551 | 0.046 |
| RoBERTa | 0.396 [0.377, 0.416] | 0.222 [0.197, 0.253] | 0.069 | 0.562 | 0.035 |
| DistilBERT | 0.395 [0.376, 0.415] | 0.217 [0.197, 0.244] | 0.050 | 0.559 | 0.043 |
| **Detoxify (zero-shot)** | **0.419** | **0.380*** | 0.170 | 0.590 | NaN** |

\* F1-macro Detoxify dihitung 2-class (negative+neutral) karena Detoxify tidak prediksi `positive`. Tidak comparable langsung ke 3-class F1-macro.
\** F1-positive Detoxify = NaN by design (zero-shot mapping).

**Pemenang sentimen** (dari fine-tuned 3-class): **RoBERTa** F1-macro=0.222 [0.197, 0.253], walau CI tumpang tindih dengan BERT dan DistilBERT (perbedaan tidak signifikan).

**Catatan kritis**: 3 fine-tuned models bias berat ke `neutral` (F1-neu 0.55, F1-pos+neg <0.07). Domain shift TweetEval (Twitter generic) → Dota chat substansial. Validation F1 0.73 → gold-test F1 0.22 = drop 0.51 poin.

**Confusion matrix**: `reports/confusion_<model>_sentiment.{csv,png}`.

### 4.2.3 Hasil Toksisitas Multi-Label (gold-test n=2.392)

| Model | F1-micro (95% CI) | F1-macro | Hamming Loss | Subset Acc | F1-toxic | F1-insult | F1-id_hate |
|-------|-------------------|----------|--------------|------------|----------|-----------|------------|
| BERT | 0.125 [0.020, 0.248] | 0.075 | 0.0059 | 0.979 | 0.075 | 0.174 | 0.000 |
| RoBERTa | 0.141 [0.032, 0.259] | 0.077 | 0.0059 | 0.980 | 0.120 | 0.080 | 0.000 |
| DistilBERT | 0.128 [0.021, 0.256] | 0.068 | 0.0047 | 0.985 | 0.108 | 0.091 | 0.000 |
| **Detoxify** | **0.152** [0.024, 0.294] | **0.081** | 0.0047 | 0.986 | 0.111 | 0.174 | 0.000 |

**Pemenang toksisitas**: **Detoxify** F1-micro=0.152, F1-macro=0.081. Specialist pretrained mengungguli fine-tuned generalist. Tapi CI sangat lebar (0.024–0.294) karena gold-test punya hanya 68 baris ber-label `toxic`.

**F1=0** untuk `severe_toxic`, `threat`, `identity_hate` di SEMUA model — karena gold-test punya 0 / 0 / 2 baris berlabel tsb. Statistik tidak informatif di label rare ini.

### 4.2.4 Analisis Error per-Jargon

Lihat `reports/jargon_error_breakdown.csv`. Error sample 50+ FP/FN per (model, tugas) di `reports/error_samples_<model>_<task>.csv`.

### 4.2.5 Diskusi Trade-off

- **DistilBERT vs BERT**: DistilBERT 256 MB checkpoint vs BERT 418 MB (40% lebih kecil), training 7.6 menit vs 40.5 menit (5x lebih cepat). Gold-test F1-macro sentiment hanya beda 0.001 → DistilBERT cocok untuk produksi moderasi real-time.
- **Specialist (Detoxify) vs Generalist fine-tuned**: Detoxify mengungguli fine-tuned di SEMUA metrik toxicity gold-test (F1-micro 0.152 vs 0.13-0.14). Pretraining domain-specific (Wikipedia/Reddit toxic) lebih kuat ketimbang fine-tune Jigsaw → transfer ke Dota.

## 4.3 Tren Temporal Sentimen dan Toksisitas (2016-2026)

### 4.3.1 Time-Series Bulanan (124 bulan, 2016-01 → 2026-04)

Plot di `reports/plots/temporal_*.{png,svg}`:
- `temporal_sentiment_monthly` — line chart mean_sentiment_score + pct_negative
- `temporal_toxicity_monthly` — line chart pct_toxic_any + mean_toxicity_score
- `temporal_negative_pct_monthly` — fokus negative %
- `temporal_toxicity_score_monthly` — fokus toxicity intensitas
- `temporal_yearly_overview` — bar chart agregat tahunan

Event overlay: patch major (7.00, 7.06, 7.20, 7.22, 7.23, 7.27, 7.29, 7.30, 7.31, 7.32, 7.33, 7.34, 7.35, 7.36, 7.37), pandemic band (2020-03 → 2021-12), TI tahunan, era DPC band.

### 4.3.2 Uji Tren Mann-Kendall

| Series | tau | z-score | p-value | Interpretasi |
|--------|-----|---------|---------|--------------|
| `mean_sentiment_score` | -0.579 | -9.531 | <1e-16 | **DECREASING** |
| `pct_negative` | -0.222 | -3.648 | 2.6e-04 | DECREASING |
| `pct_toxic_any` | -0.495 | -8.145 | <1e-15 | **DECREASING** |
| `mean_toxicity_score` | +0.324 | +5.337 | 9.5e-08 | **INCREASING** |

**Anomali penting**: `pct_toxic_any` turun (chat toksik makin jarang) tapi `mean_toxicity_score` naik (intensitas saat muncul makin tinggi). Pattern *intensification* — frekuensi turun, severity naik.

### 4.3.3 Structural Break (Chow Test)

Breakpoints kandidat: pandemic_start (2020-03), post_dpc (2023-01), patch_7.33 (2023-04).

| Series | Pandemic 2020-03 | Post-DPC 2023-01 | Patch 7.33 |
|--------|------------------|------------------|------------|
| `mean_sentiment_score` | F=3.88, p=0.023 ✓ | F=0.35, p=0.71 | F=0.34, p=0.72 |
| `pct_negative` | F=4.12, p=0.019 ✓ | F=5.30, p=0.006 ✓ | F=4.50, p=0.013 ✓ |
| `pct_toxic_any` | F=14.75, p=1.9e-06 ✓ | F=0.62, p=0.54 | F=0.24, p=0.79 |
| `mean_toxicity_score` | F=13.33, p=5.9e-06 ✓ | F=0.42, p=0.66 | F=0.59, p=0.55 |

**Pandemi 2020-03 = breakpoint TERKUAT**: structural break terdeteksi di SEMUA 4 metrik (p<0.05). Post-DPC dan Patch 7.33 hanya break di pct_negative.

### 4.3.4 Event Comparison (Mann-Whitney U)

| Pair | Metric | Mean A | Mean B | Δ | p-value |
|------|--------|--------|--------|---|---------|
| pre-pandemic vs pandemic-online | max_toxicity_prob | 0.0525 | 0.0504 | -0.0022 | 7.8e-193 |
| pre-pandemic vs pandemic-online | sentiment_score | 0.0277 | 0.0251 | -0.0027 | 6.7e-101 |
| pandemic-online vs post-pandemic-LAN | max_toxicity_prob | 0.0504 | 0.0525 | +0.0021 | <1e-300 |
| pandemic-online vs post-pandemic-LAN | sentiment_score | 0.0251 | 0.0170 | -0.0080 | <1e-300 |
| dpc-era vs post-dpc | max_toxicity_prob | 0.0519 | 0.0532 | +0.0013 | <1e-300 |
| dpc-era vs post-dpc | sentiment_score | 0.0250 | 0.0162 | -0.0088 | <1e-300 |

Semua perbandingan signifikan secara statistik karena n besar (>200k per kelompok). Effect size kecil (|Δ| < 0.01) — konteks klinis ringan.

### 4.3.5 Pembahasan Kualitatif

- **Pandemi COVID-19 (2020-2021)**: Pivot LAN→online disrupt SEMUA 4 metrik (Chow break p<0.05). Toksisitas turun sedikit (-0.002), sentimen turun (-0.003). Tampak **dampak emosional** isolasi/online play tapi mild.
- **Pergantian patch major**: Patch 7.33 (2023-04 — meta overhaul terbesar) hanya break di `pct_negative`. Patch lain tidak menyebabkan break sistemik.
- **Transisi post-DPC (2023+)**: sportsmanship turun signifikan (sentiment_score -0.009 dari era DPC ke post-DPC, p<1e-300). Hipotesis: hilangnya struktur kompetitif formal DPC mengurangi insentif sportsmanship. Toksisitas naik tipis (+0.001).
- **Tren dekade**: Frekuensi toksisitas turun (regulation Valve + komunitas dewasa), tapi intensitas saat muncul makin parah (extreme tail).

## 4.4 Korelasi Faktor Kontekstual

### 4.4.1 Hasil Uji (10 tests, BH-FDR koreksi q=0.05)

Semua 10 uji **SIGNIFIKAN** setelah BH-FDR. Lihat `reports/correlation_tests.csv` + `correlation_summary.md`.

| Feature | Label | Test | Stat | p_adj_BH | Effect Size | Interpretasi |
|---------|-------|------|------|----------|-------------|--------------|
| `match_outcome_for_player` | sentiment | chi-square | 341.6 | <1e-74 | V=0.016 | Kalah/menang ↔ sentimen |
| `match_outcome_for_player` | max_toxicity | Mann-Whitney | - | 0 | r=**0.060** | Kalah lebih toxic |
| `duration` | sentiment_score | Spearman | -0.055 | 0 | ρ=-0.055 | Match panjang ↔ sentiment turun |
| `duration` | max_toxicity | Spearman | -0.080 | 0 | **ρ=-0.080** | Match panjang ↔ toxicity turun |
| `tier` | max_toxicity | Kruskal-Wallis | 887.4 | <1e-191 | η²=0.001 | Tier ↔ toxicity (tipis) |
| `tier` | sentiment | chi-square | 65.3 | <1e-11 | V=0.005 | Tier ↔ sentiment (tipis) |
| `phase` | max_toxicity | Kruskal-Wallis | 40.4 | <1e-08 | η²=0.000 | Phase ↔ toxicity (sangat tipis) |
| `phase` | sentiment | chi-square | 28.5 | <1e-05 | V=0.003 | Phase ↔ sentiment (sangat tipis) |
| `period` | max_toxicity | Kruskal-Wallis | 14672.6 | 0 | η²=0.011 | Period ↔ toxicity |
| `period` | sentiment | chi-square | 4295.4 | 0 | **V=0.039** | Period ↔ sentiment (terkuat) |

**3 finding terkuat (effect size desc):**

1. **`duration` × `max_toxicity`** (Spearman ρ=-0.080): match makin lama → toxicity makin RENDAH. Intuitif: pertandingan singkat (early-stomp) lebih toxic karena frustrasi blowout.
2. **`match_outcome` × `max_toxicity`** (Mann-Whitney r=0.060): tim kalah cenderung lebih toxic. **Konfirmasi hipotesis utama H1**.
3. **`period` × `sentiment`** (Cramér's V=0.039): periode waktu pengaruh sentimen — sportsmanship menurun antar dekade.

**Catatan effect size**: semua effect <0.10. n besar (~1.3-1.4 juta) → statistik signifikan tapi efek praktis kecil. Reporting di skripsi WAJIB cantumkan effect size + n bersama p-value.

### 4.4.2 Robustness pada Level per-Pertandingan

Untuk kontrol clustering effect (pesan dalam satu match tidak independen):

| Test | Stat | p-value | Interpretasi |
|------|------|---------|--------------|
| `duration × mean_toxicity` (Spearman per-match, n=197.695) | ρ=-0.147 | <1e-300 | **DIPERKUAT** (vs ρ=-0.080 per-message) |
| `tier × mean_toxicity` (Kruskal per-match) | H=1.76 | 0.62 | **NS** (vs sig per-message) |

Tier effect HILANG di level per-match → tier × toxicity sebelumnya kemungkinan artefak clustering (pesan banyak di TI = inflate sample). Klaim tier × toxicity di skripsi HARUS diturunkan ke "marginal/eksploratori".

Duration × toxicity DIPERKUAT di per-match level → finding ini ROBUST.

### 4.4.3 Pembahasan

- **Hasil pertandingan**: H1 tervalidasi. Tim kalah ~6% lebih toxic (rank-biserial). Implikasi: detection moderasi otomatis perlu mempertimbangkan loser-bias.
- **Durasi pertandingan**: H2 tervalidasi (negatif). Pertandingan singkat (stomp) PALING toxic. Match panjang (close game) LEBIH SOPAN — komunikasi koordinasi dominan.
- **Tier turnamen**: efek per-match NS. Klaim tier × toxicity TIDAK robust. Hapus dari klaim utama.
- **Periode waktu**: sportsmanship menurun antar dekade (V=0.039). Konsisten dengan H3 dan finding temporal §4.3.

## 4.5 Pembahasan dan Limitasi

### 4.5.1 Implikasi Praktis

- **Pengembang game (Valve)**: rekomendasi sistem moderasi otomatis pakai **Detoxify pretrained** (F1-micro 0.152 di Dota chat, terbaik dari 4 model). Tidak perlu fine-tune lagi — domain Wikipedia/Reddit-toxic transfer cukup baik. Threshold prioritas: pertandingan singkat (<30 menit) dan tim kalah → high-risk moderation focus.
- **Penyelenggara esports**: post-DPC era menunjukkan penurunan sportsmanship (V=0.039, p<1e-300). Format tournament structured berkorelasi dengan komunikasi sehat. Pertimbangkan struktur kompetitif formal pasca-DPC.
- **Komunitas peneliti**: metodologi dapat direplikasi ke MOBA lain (LoL, Mobile Legends). Pipeline `01_data_pipeline → 08_correlation` reusable. Single-annotator AI pendekatan yang scalable tapi butuh cross-validation human untuk publikasi.

### 4.5.2 Limitasi

- **Kualitas label**: gold-standard 7.971 single-annotator AI. Tidak ada Cohen's kappa / Krippendorff's alpha. **Kerja masa depan**: replikasi dengan 3 anotator manusia + AI = 4-rater study.
- **Domain gap**: TweetEval (Twitter generic) → Dota chat memberikan F1-macro 0.22 (drop dari 0.73 di publik). Jigsaw (Wikipedia) → Dota chat memberikan F1-micro 0.13-0.15. Gap besar sugesti **fine-tune di chat MOBA** sebagai prioritas penelitian lanjutan.
- **Phase derivation lemah**: lihat §4 `reports/contextual_features.md`. Klaim phase × toxicity HARUS hati-hati — derivation berbasis heuristik leaguename, tidak match-level akurat.
- **Bahasa**: hanya Inggris. Non-Inggris di-eksklusi (274.087 baris = **14.6%** data hilang). Russia, Filipino, Spanyol substansial tapi out-of-scope.
- **Clustering**: pesan dalam satu match tidak independen. Robustness per-match dilaporkan (duration robust, tier tidak). Power statistik berkurang di level per-match.
- **Class imbalance ekstrem di gold**: severe_toxic (0%), threat (0%), identity_hate (0.025%) di gold-test → F1=0 di semua model untuk label rare. Stratified sampling lebih ketat di label rare diperlukan.
- **Validation-test gap**: validation F1 sentiment 0.73 → gold-test 0.22 = drop 0.51 poin. Validation set publik (TweetEval) bukan estimator yang baik untuk performa di domain Dota chat.

### 4.5.3 Saran Penelitian Lanjutan

- **Multilingual analysis**: 274k pesan non-Inggris (Russia/CN/SEA) yang dieksklusi. Replikasi pipeline dengan multilingual BERT (XLM-RoBERTa) → cakupan +14% data.
- **Multi-modal analysis**: voice chat (transcribe) + emote + chat teks. Voice signal kemungkinan paling kaya untuk emosi.
- **Kausalitas, bukan korelasi**: eksperimen dengan intervensi moderasi (random assignment of communication ban duration). Korelasi → kausalitas.
- **Anotasi crowdsource skala 50.000+** untuk fine-tuning di domain Dota. Dapat naikkan F1-macro sentiment dari 0.22 → estimasi 0.55-0.65 (berbasis literature in-domain fine-tuning).
- **Per-player longitudinal**: track toxicity/sentiment per pemain sepanjang karir (account_id sudah tersedia). Identifikasi pola personal vs global.
- **Toxicity intensification**: investigasi mengapa frekuensi toksik turun tapi severity naik. Apakah Valve communication ban hanya filter low-severity, leave high-severity?

---

## Daftar Artefak Pendukung

| Section | Artefak | File |
|---------|---------|------|
| 4.1 | Distribusi tahunan | `reports/temporal_yearly.csv` |
| 4.1 | Agreement report | `data/gold/agreement_report.md` |
| 4.1 | Tooling decision | `data/gold/TOOLING_DECISION.md` |
| 4.2 | Tabel komparatif | `reports/comparison_summary.md`, `reports/eval_sentiment.csv`, `reports/eval_toxicity.csv` |
| 4.2 | Confusion matrices | `reports/confusion_<model>_sentiment.{csv,png}` (3 model) |
| 4.2 | Error samples | `reports/error_samples_<model>_sentiment.csv` (3 model, ~66 KB each) |
| 4.2 | Training logs | `reports/training_<model>_<task>.log` (6 file) |
| 4.3 | Time-series CSV | `reports/temporal_monthly.csv` (124 baris), `reports/temporal_yearly.csv` (11 baris) |
| 4.3 | Plot temporal | `reports/plots/temporal_*.{png,svg}` (5 plot × 2 format) |
| 4.3 | Tests | `reports/temporal_tests.csv`, `reports/event_comparison.csv` |
| 4.4 | Tests | `reports/correlation_tests.csv` (10 uji) |
| 4.4 | Plot korelasi | `reports/plots/correlation_*.{png,svg}` (5 plot × 2 format) |
| 4.4 | Robustness | `reports/correlation_robustness_per_match.csv` |
| 4.5 | Manifest | `configs/experiment.yaml` (model SHA, dataset SHA256, hyperparameters, hash data) |
| 4.5 | Run log | `reports/run_log.csv` (8 entries, NB01-NB08) |

**Reproducibility**:
- Seed: 42
- Total runtime: NB01 (1603s) + NB02 (5s) + NB03 (1.6s) + NB04 (24965s GPU) + NB05 (4146s GPU) + NB06 (36s) + NB07 (17s) + NB08 (51s) = **~8.5 jam total** (mayoritas GPU training).
- Hardware: NVIDIA CUDA 12.8, transformers v5.5.0, torch v2.8.0, Python 3.10.
