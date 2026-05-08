## ADDED Requirements

### Requirement: Stratified sampling untuk gold-standard set

Sistem SHALL memilih sample 8.000 pesan chat (target ±10%) dari `data/processed/*.parquet` dengan stratifikasi pada tiga dimensi: `year` (2016-2026), `match_outcome_for_player` (win/loss), dan `tier` turnamen (TI / Major / DPC Tour / Lainnya).

Setiap stratum MUST memiliki minimal 50 pesan. Jika populasi suatu stratum < 50 (mis. era 2016 untuk tier "Lainnya"), seluruh populasi diambil dan kekurangan dilaporkan di log.

Sampling MUST deterministik — diberikan seed di `configs/experiment.yaml`, eksekusi ulang menghasilkan sample yang identik.

#### Scenario: Sample stratified terdistribusi merata
- **WHEN** proses sampling dijalankan dengan seed yang ditetapkan
- **THEN** output `data/gold/sample.csv` berisi ~8000 baris dengan distribusi minimal 50 pesan per stratum (year × outcome × tier) bila populasi mencukupi

#### Scenario: Sampling deterministik
- **WHEN** proses sampling dijalankan dua kali dengan seed yang sama
- **THEN** kedua eksekusi menghasilkan sample dengan baris yang identik (cek hash baris)

#### Scenario: Stratum kecil dilaporkan
- **WHEN** suatu stratum memiliki populasi < 50 pesan
- **THEN** seluruh populasi diambil dan log mencatat `[WARN] stratum=<year>x<outcome>x<tier> populasi=N target=50`

### Requirement: Skema label dual-axis (sentiment + toxicity)

Sistem SHALL menyediakan skema anotasi dengan dua sumbu untuk setiap pesan:

- **Sentimen (single-label, 3-class)**: `positive`, `neutral`, `negative`.
- **Toksisitas (multi-label, 6 label biner)**: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Skema mengikuti Detoxify/Jigsaw Toxic Comment supaya hasil model specialist dapat dievaluasi setara.

Setiap pesan MUST diannotasi pada KEDUA sumbu — tidak ada pesan yang hanya di-label sentimen atau hanya di-label toksisitas.

Sistem MUST menyediakan kolom tambahan: `is_dota_jargon` (boolean — apakah pesan didominasi jargon Dota), `is_ambiguous` (boolean — anotator tidak yakin), `notes` (string opsional).

#### Scenario: Pesan diannotasi dual-axis
- **WHEN** anotator menyelesaikan satu baris di file anotasi
- **THEN** baris tersebut memiliki nilai non-null di kolom `sentiment` dan ke-6 kolom toksisitas (0/1)

#### Scenario: Pesan toksik berat
- **WHEN** anotator membaca pesan dengan ujaran kebencian rasial eksplisit
- **THEN** label valid: `sentiment=negative`, `toxic=1`, `severe_toxic=1`, `identity_hate=1`, sisa label opsional

### Requirement: Guideline anotasi yang terdokumentasi

Sistem SHALL menyediakan dokumen guideline anotasi di `data/gold/ANNOTATION_GUIDELINE.md` yang berisi:
- Definisi operasional setiap label.
- Contoh positif dan negatif untuk setiap label (minimal 5 contoh).
- Aturan disambiguasi (mis. sarcasm dianggap apa, jargon Dota dianggap apa, tilt non-spesifik dianggap apa).
- Instruksi penanganan kasus ambigu.

Dokumen guideline MUST diversionkan via git dan setiap perubahan signifikan setelah anotasi dimulai memerlukan re-kalibrasi.

#### Scenario: Anotator merujuk guideline
- **WHEN** anotator menemui pesan yang ambigu
- **THEN** guideline memberikan aturan eksplisit (atau instruksi `is_ambiguous=1`)

#### Scenario: Guideline diubah saat anotasi berjalan
- **WHEN** definisi label diubah setelah anotator mulai produksi
- **THEN** sample minimal 100 pesan terdahulu di-review ulang dengan definisi baru, dan log perubahan dicatat di guideline

### Requirement: Pilot kalibrasi sebelum produksi

Sistem SHALL menjalankan tahap pilot 200 pesan sebelum anotasi produksi 8.000 pesan dimulai. Pilot diannotasi oleh ketiga anotator secara penuh (overlap 100%), kemudian dilakukan diskusi adjudikasi untuk setiap disagreement.

Cohen's kappa pair-wise dari pilot MUST dilaporkan dan dijadikan dasar GO/NO-GO produksi. Threshold rilis: kappa minimal **0.6** untuk sentimen dan Krippendorff's alpha minimal **0.55** untuk toksisitas multi-label.

Jika threshold tidak tercapai, guideline MUST direvisi dan pilot diulang dengan 100 pesan baru.

#### Scenario: Pilot lolos threshold
- **WHEN** kappa pair-wise sentimen di pilot ≥ 0.6 untuk semua pasangan anotator
- **THEN** anotasi produksi dimulai

#### Scenario: Pilot di bawah threshold
- **WHEN** kappa pair-wise di salah satu pasangan < 0.6
- **THEN** anotasi produksi DITUNDA, guideline direvisi, pilot diulang dengan 100 pesan baru, dan kappa dihitung ulang

### Requirement: Inter-annotator agreement di production set

Sistem SHALL mengukur dan melaporkan inter-annotator agreement di production set 8.000 pesan dengan modus salah satu dari:
- **Mode A (full overlap)**: ketiga anotator menganotasi 100% sample. Cohen's kappa pair-wise + Fleiss' kappa.
- **Mode B (2-of-3 overlap dengan adjudikasi)**: setiap pesan diannotasi 2 anotator; jika berbeda, anotator ketiga sebagai adjudikator. Kappa pair-wise dilaporkan dari sub-sample yang overlap.

Mode dipilih setelah pilot dan didokumentasikan di laporan eksperimen.

#### Scenario: Kappa final dilaporkan
- **WHEN** anotasi produksi selesai
- **THEN** `data/gold/agreement_report.md` berisi tabel kappa pair-wise (sentimen) + Krippendorff's alpha (toksisitas) + jumlah disagreement yang di-adjudikasi

### Requirement: Output gold final dengan label konsensus

Sistem SHALL menghasilkan `data/gold/gold_final.csv` (atau `.parquet`) berisi setiap pesan sample beserta:
- Label sentimen konsensus (mayoritas atau hasil adjudikasi).
- Label toksisitas konsensus (per-label, mayoritas).
- Kolom `confidence`: `unanimous` (semua anotator setuju) / `majority` (2-of-3) / `adjudicated` (dipakai keputusan adjudikator).
- Reference ke `match_id`, `time`, `player_slot`, dan kolom kontekstual dari pipeline.

#### Scenario: Konsensus mayoritas
- **WHEN** 2 dari 3 anotator menetapkan `sentiment=negative` dan 1 menetapkan `neutral`
- **THEN** label konsensus `sentiment=negative`, `confidence=majority`

#### Scenario: Konsensus penuh
- **WHEN** ketiga anotator menetapkan `toxic=1` dan `insult=1`
- **THEN** label konsensus `toxic=1`, `insult=1`, `confidence=unanimous`

#### Scenario: Adjudikasi dipakai
- **WHEN** anotator A=positive, B=negative, C=neutral pada satu pesan
- **THEN** adjudikator menetapkan label final dan `confidence=adjudicated`
