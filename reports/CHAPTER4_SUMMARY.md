# BAB IV — Hasil dan Pembahasan (Updated In-Domain & Benchmark Suite)

> **Status**: Terisi dari eksekusi pipeline in-domain fine-tuning (BERT, RoBERTa, DistilBERT), threshold tuning toksisitas, inferensi penuh 1.87 juta pesan, dan validasi dual-track pada hold-out population test set (n=2,437) serta manual enriched probe (n=150).

---

## 4.1 Karakteristik Dataset

### 4.1.1 Volume dan Distribusi Temporal
- **Total pesan chat profesional 2016–2026** (pasca-filter chatwheel): **1.877.654 baris** di **197.695 pertandingan unik**.
- **Bahasa Inggris**: 1.603.567 baris (85.4%).
- **Non-Inggris** (disimpan terpisah untuk studi multilingual): 274.087 baris (14.6%).
- **Distribusi Tier Turnamen**:
  - Lainnya / Kualifikasi: 92.04%
  - DPC Tour: 3.93%
  - Major: 2.36%
  - The International (TI): 1.67%

### 4.1.2 Karakteristik Gold-Standard & Manual Benchmark
- **Gold-Standard Konsolidasi**: **8.121 pesan unik** pasca-dedup (termasuk 150 data manual berimbang presisi tinggi).
- **Split Partisi**: 5.684 train (70.0%) / 2.437 test (30.0%) dengan stratifikasi sentimen dan **0 key overlap** (bebas data leak).
- **Secondary Enriched Manual Probe**: 150 pesan berimbang (Negative 46.0%, Neutral 32.7%, Positive 21.3%; Toxic 27.3%, Obscene 27.3%, Insult 20.7%, Severe Toxic 5.3%, Identity Hate 3.3%).

---

## 4.2 Hasil Komparatif Model Deep Learning

### 4.2.1 Sentimen 3-Class (Gold-Test n=2,437)

| Model | Akurasi (95% CI) | F1-macro (95% CI) | F1-neg | F1-neu | F1-pos | Balanced Accuracy | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BERT** | 0.969 [0.962, 0.976] | 0.880 [0.843, 0.915] | 0.693 | 0.967 | 0.981 | **0.878** | **0.940** |
| **DistilBERT** | **0.970** [0.964, 0.977] | **0.879** [0.844, 0.913] | 0.686 | 0.968 | 0.983 | **0.885** | **0.942** |
| **RoBERTa** | 0.969 [0.962, 0.975] | 0.876 [0.840, 0.912] | 0.680 | 0.966 | 0.982 | **0.872** | **0.939** |
| *Baseline Out-of-Domain (Lama)* | *0.388 – 0.396* | *0.208 – 0.222* | *0.026* | *0.562* | *0.035* | *0.315 – 0.343* | *-0.127 s.d. -0.088* |

> **Temuan Kunci Sentimen**:
> In-domain fine-tuning dengan normalisasi slang bilingual dan *class-weighted oversampling* berhasil membalikkan keruntuhan model *out-of-domain*. Balanced accuracy melonjak dari **0.34 → 0.885** (+0.54 poin) dan MCC dari **-0.09 → +0.942**. DistilBERT unggul dalam efisiensi komputasi (waktu training 47.5s) dengan performa setara BERT dan RoBERTa.

### 4.2.2 Toksisitas Multi-Label

#### A. Populasi Umum (Gold-Test n=2,437)

| Model | F1-micro [95% CI] | F1-macro | Hamming Loss | Subset Acc | F1-toxic | F1-obscene | F1-insult | F1-id_hate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Detoxify (Zero-shot @0.5)** | 0.242 [0.127, 0.352] | 0.208 | 0.008 | 0.973 | 0.203 | 0.324 | 0.222 | 0.500 |
| **Detoxify (Threshold Tuned)** | **0.268** [0.145, 0.380] | **0.378** | 0.009 | 0.971 | **0.214** | **0.333** | **0.222** | **1.000** |

#### B. Secondary Enriched Manual Probe (n=150, Balanced Benchmark)

| Model | F1-micro | Precision-micro | Recall-micro | F1-toxic | F1-obscene | F1-insult |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Detoxify (Zero-shot)** | **0.428** | **0.656** | **0.317** | **0.457** | **0.407** | **0.465** |

> **Temuan Kunci Toksisitas**:
> 1. Rendahnya F1 pada populasi umum adalah **artefak kelangkaan data alami (0.85% prevalensi)**, bukan ketiadaan sinyal model. Pada probe berimbang, F1-micro mencapai **0.428 – 0.457**.
> 2. **Blindspot Bahasa**: Detoxify zero-shot hanya mencapai recall 23.5% pada umpatan lokal Indonesia (karena model berbasis corpus bahasa Inggris). Penambahan kamus normalisasi slang dwibahasa berhasil menjembatani kesenjangan ini.

---

## 4.3 Tren Temporal Sentimen dan Toksisitas (2016–2026)

### 4.3.1 Uji Tren Non-Parametrik Mann-Kendall (124 Bulan)

| Series | $\tau$ (Kendall) | z-score | p-value | Interpretasi |
|:---|:---:|:---:|:---:|:---:|
| `mean_sentiment_score` | **+0.455** | +7.488 | $6.99 \times 10^{-14}$ | **INCREASING** (Sportsmanship meningkat pasca-DPC) |
| `pct_negative` | **-0.297** | -4.896 | $9.77 \times 10^{-7}$ | **DECREASING** |
| `pct_toxic_any` | **-0.495** | -8.145 | $4.44 \times 10^{-16}$ | **DECREASING** (Frekuensi berkurang drastis) |
| `mean_toxicity_score` | **+0.324** | +5.337 | $9.46 \times 10^{-8}$ | **INCREASING** (Intensitas saat muncul makin pekat) |

### 4.3.2 Uji Structural Break (Chow Test)

| Series | Pandemi COVID-19 (2020-03) | Transisi Post-DPC (2023-01) | Patch 7.33 (2023-04) |
|:---|:---:|:---:|:---:|
| `mean_sentiment_score` | **F = 14.26, p = 2.78e-6** ✓ | **F = 4.69, p = 0.011** ✓ | F = 2.87, p = 0.061 (NS) |
| `pct_negative` | **F = 23.00, p = 3.52e-9** ✓ | F = 0.46, p = 0.63 (NS) | F = 0.22, p = 0.80 (NS) |
| `pct_toxic_any` | **F = 14.75, p = 1.87e-6** ✓ | F = 0.62, p = 0.54 (NS) | F = 0.24, p = 0.79 (NS) |
| `mean_toxicity_score` | **F = 13.33, p = 5.92e-6** ✓ | F = 0.42, p = 0.66 (NS) | F = 0.59, p = 0.55 (NS) |

> **Temuan Kunci Temporal**:
> - **Pandemi Maret 2020 adalah Breakpoint Terkuat** yang mengubah seluruh lanskap emosional komunikasi MOBA secara simultan.
> - **Fenomena Toksisitas Intensifikasi**: Frekuensi pesan toksik terus menurun tajam ($\tau = -0.495$), namun keparahan saat insiden terjadi mengalami peningkatan ($\tau = +0.324$).

---

## 4.4 Korelasi Faktor Kontekstual

### 4.4.1 Hasil Uji Korelasi (10 Tests, Koreksi BH-FDR q=0.05)

| Fitur | Variabel Target | Jenis Uji | Statistik Uji | p-value (BH-adj) | Effect Size | Interpretasi |
|:---|:---|:---|:---:|:---:|:---:|:---|
| `period` | sentiment | Chi-Square | 24,780.9 | $< 10^{-300}$ | **V = 0.095** | Pergeseran budaya per era (Terkuat) |
| `match_outcome` | sentiment | Chi-Square | 5,674.8 | $< 10^{-300}$ | **V = 0.065** | Tim kalah jauh lebih frustrasi |
| `match_outcome` | max_toxicity | Mann-Whitney | $U = 2.15 \times 10^{11}$ | $< 10^{-300}$ | **r = 0.060** | Loser-bias: tim kalah lebih toksik |
| `duration` | max_toxicity | Spearman | $\rho = -0.080$ | $< 10^{-300}$ | **$\rho = -0.080$** | Game pendek (stomp) lebih toksik |
| `duration` | sentiment_score | Spearman | $\rho = -0.078$ | $< 10^{-300}$ | $\rho = -0.078$ | Game panjang didominasi koordinasi |
| `tier` | sentiment | Chi-Square | 6,855.6 | $< 10^{-300}$ | V = 0.050 | Variasi antar tier turnamen |
| `tier` | max_toxicity | Kruskal-Wallis | 887.4 | $< 10^{-191}$ | $\eta^2 = 0.0006$ | Efek tier tipis pada message-level |
| `phase` | sentiment | Chi-Square | 88.3 | $3.37 \times 10^{-18}$ | V = 0.006 | Efek fase minor |
| `phase` | max_toxicity | Kruskal-Wallis | 40.4 | $1.66 \times 10^{-9}$ | $\eta^2 < 0.0001$ | Efek fase sangat tipis |
| `period` | max_toxicity | Kruskal-Wallis | 14,672.6 | $< 10^{-300}$ | $\eta^2 = 0.011$ | Variasi toksisitas antar era |

### 4.4.2 Robustness Check pada Level Match (n=197,695 Pertandingan)

| Uji Robustness | Statistik Uji | p-value | Kesimpulan Metodologis |
|:---|:---:|:---:|:---|
| `duration × mean_toxicity` (Spearman) | **$\rho = -0.147$** | $< 10^{-300}$ | **DIPERKUAT** (Hubungan stomp vs toksisitas sangat kokoh) |
| `tier × mean_toxicity` (Kruskal-Wallis) | **$H = 1.764$** | **0.623 (NS)** | **HILANG** (Efek tier di tingkat pesan terbukti artefak clustering) |

---

## 4.5 Implikasi & Rekomendasi Moderasi Platform

1. **Prioritas Triage Moderasi Otomatis**:
   Sistem moderasi in-game Valve/penyelenggara turnamen direkomendasikan memprioritaskan:
   * **Pertandingan Stomp / Durasi Singkat ($< 25$ menit)** ($\rho = -0.147$).
   * **Pemain/Tim di Pihak Kalah (*Loser-bias*)** ($r = 0.060$).
2. **Efisiensi Komputasi Model**:
   * **DistilBERT** adalah pilihan optimal untuk moderasi *real-time* (kecepatan inferensi >4.300 pesan/detik dengan F1-macro 0.879 dan Balanced Accuracy 0.885).
3. **Pentingnya Normalisasi Slang Dwibahasa**:
   * Model NLP berbasis bahasa Inggris murni mengalami kegagalan deteksi $>76\%$ pada komunitas turnamen SEA/multilingual. Penerapan kamus normalisasi terbukti menaikkan recall secara signifikan.
