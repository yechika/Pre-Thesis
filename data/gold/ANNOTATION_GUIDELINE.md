# Pedoman Anotasi — Gold Standard Sentimen + Toksisitas Chat Dota 2

**Versi**: 1.0 (2026-04-28)
**Anotator**: Daniel Dirgantara, Muhammad Dhitan Imam Sakti, Aldiaz Kusuma Ramadhan
**Sample**: `data/gold/sample.csv` (~8.000 pesan, stratified per `year × outcome × tier`).

Tujuan dokumen: memastikan ketiga anotator melabelkan pesan dengan kriteria yang sama, sehingga inter-annotator agreement (Cohen's kappa, Krippendorff's alpha) terukur dengan reliable.

> **Sebelum mulai**: baca seluruh dokumen ini, lalu kerjakan **pilot 200 pesan** terlebih dahulu (lihat `notebooks/03_annotation_review.ipynb`). Jangan mulai produksi 8.000 sebelum kappa pilot ≥ 0.6.

---

## 1. Skema Label Dual-Axis

Setiap pesan diannotasi pada **dua sumbu independen**:

### 1.1 Sentimen (single-label, 3-class)

Pilih SATU dari:

| Label      | Definisi                                                                 |
|------------|--------------------------------------------------------------------------|
| `positive` | Pesan menunjukkan emosi positif: pujian, dukungan, antusiasme, humor non-sarkastik. |
| `neutral`  | Pesan informatif/koordinasi tanpa muatan emosi yang jelas.                |
| `negative` | Pesan menunjukkan emosi negatif: frustrasi, tilt, kekecewaan, kemarahan, putus asa, sinis. |

> Sentimen ≠ toksisitas. Pesan bisa `negative` tanpa toksik (mis. "we're losing this badly"), atau `neutral` tapi toksik (mis. ujaran kebencian dingin), atau `positive` dan toksik (mis. sarcasm + insult yang dibalut "good job").

### 1.2 Toksisitas (multi-label, 6 label biner)

Untuk setiap label, beri 0 (tidak ada) atau 1 (ada). Lebih dari satu label bisa = 1 sekaligus.

| Label          | Definisi                                                                                  |
|----------------|--------------------------------------------------------------------------------------------|
| `toxic`        | Pesan secara umum bersifat menyerang/destruktif terhadap orang lain.                       |
| `severe_toxic` | Pesan ekstrim — ujaran kebencian eksplisit, ancaman serius, pelecehan berat. (subset dari `toxic`, tapi level berat) |
| `obscene`      | Mengandung kata kasar, makian, atau kotor (fuck, shit, dll.) — terlepas dari niat menyerang. |
| `threat`       | Mengancam kekerasan/celaka kepada orang lain ("I'll kill you", "kys").                    |
| `insult`       | Menghina pemain lain secara langsung (idiot, trash, noob, retard).                        |
| `identity_hate`| Ujaran kebencian berbasis identitas: ras, agama, gender, orientasi seksual, kebangsaan.    |

**Aturan kombinasi**:
- Pesan `severe_toxic=1` HARUS juga `toxic=1`.
- `identity_hate` SHARUSNYA juga memicu `toxic=1` dan `insult=1`.
- `threat` SHARUSNYA juga memicu `toxic=1`.
- Pesan murni `obscene` (mis. "fuck this lag") TANPA target orang BISA `toxic=0`.

### 1.3 Kolom tambahan

| Kolom            | Nilai      | Definisi                                                                  |
|------------------|------------|---------------------------------------------------------------------------|
| `is_dota_jargon` | 0/1        | Pesan didominasi jargon Dota (>50% token jargon) — penanda untuk analisis error per-jargon. |
| `is_ambiguous`   | 0/1        | Anotator tidak yakin akan label sentimen — beri keterangan di `notes`.    |
| `notes`          | string     | Bebas — alasan ambiguitas, konteks tambahan, atau catatan untuk adjudikator. |

---

## 2. Contoh Berlabel

### 2.1 Sentiment

**Positive (5 contoh)**
- "gg wp guys" → `positive` (selama bukan sarkastik)
- "nice play mid" → `positive`
- "we got this team, push together" → `positive`
- "thanks for the heal" → `positive`
- "lol that ulti tho 😄" → `positive`

**Neutral (5 contoh)**
- "rosh in 2 min" → `neutral`
- "missing top" → `neutral`
- "bot rune" → `neutral`
- "buy back ready" → `neutral`
- "smoke at 5 min" → `neutral`

**Negative (5 contoh)**
- "this lag is unbearable" → `negative` (frustrasi tanpa target orang)
- "we threw that fight" → `negative`
- "lost so bad lmao" → `negative`
- "i can't carry this trash team" → `negative` (juga `toxic=1, insult=1`)
- "ff at 25, no point" → `negative` (defeatism)

### 2.2 Toksisitas

**toxic + insult**:
- "you're literally the worst pos5 i've ever seen" → `toxic=1, insult=1`
- "uninstall noob" → `toxic=1, insult=1`

**obscene only (no target)**:
- "wtf this rng" → `obscene=1, toxic=0`
- "fucking lag" → `obscene=1, toxic=0`

**toxic + obscene + insult**:
- "shut the fuck up trash" → `toxic=1, obscene=1, insult=1`

**threat**:
- "i'll find your house" → `threat=1, toxic=1, severe_toxic=1`
- "kys" → `threat=1, toxic=1, insult=1, severe_toxic=1`

**identity_hate**:
- "[ujaran rasial eksplisit]" → `identity_hate=1, toxic=1, severe_toxic=1, insult=1`
- "typical [nationality] feeders" → `identity_hate=1, toxic=1`

**clean (semua 0)**:
- "buy back ready", "rosh in 2 min", "gg wp" → semua label toksisitas = 0.

---

## 3. Aturan Disambiguasi

### 3.1 Sarcasm

Sarcasm SERING sulit dideteksi tanpa konteks. Aturan:
- Jika pesan secara permukaan positif tapi dari konteks pertandingan jelas sarkastik (mis. "wow great pick" setelah teammate pick hero off-meta dan kalah lane), label = `negative`.
- Jika tidak ada cukup konteks untuk yakin → set `is_ambiguous=1` dan beri `notes`.

### 3.2 Jargon Dota

Jargon Dota yang sering ditemui dan label default-nya:
| Jargon          | Default sentimen | Toksik? |
|-----------------|------------------|---------|
| "gg"            | positive (akhir match) atau negative (mid-match early-GG = tilt) | tidak |
| "ez"            | tergantung konteks: kalau menang → neutral/positive; kalau menyerang → toxic + insult | bisa |
| "ff"            | negative (defeatism) | tidak |
| "noob"          | toxic + insult (offensif) | ya |
| "retard", "tard"| toxic + insult, kemungkinan severe_toxic | ya |
| "feeder", "fed" | tergantung: deskriptif neutral; kalau menyerang teammate → insult | bisa |
| "smurf"         | neutral (deskriptif) atau toxic (tuduhan) | bisa |
| "jungler", "mid", "carry" | neutral (peran) | tidak |
| "bot/top/mid lane" | neutral (lokasi) | tidak |

Beri `is_dota_jargon=1` jika lebih dari setengah token adalah jargon Dota (mis. "rosh smoke 5 mins", "bot rune up").

### 3.3 Ambigu murni

Jika setelah membaca pesan + kolom konteks (year, tier, outcome) anotator masih tidak yakin label sentimen:
1. Set `is_ambiguous=1`.
2. Pilih label sentimen terbaik berdasarkan tebakan terdekat (jangan kosongkan).
3. Tulis di `notes` mengapa ambigu (mis. "tone tidak jelas, bisa positif atau sarkastik").

Untuk toksisitas, label SELALU 0/1 (tidak boleh kosong) — jika ragu, default ke 0 (konservatif).

### 3.4 Bahasa campuran / non-Inggris

Sample sudah difilter ke bahasa Inggris saja oleh pipeline. Jika menemui pesan non-Inggris (false positive dari language detector):
- Set `notes='non-english'`.
- Tetap beri label sentimen + toksisitas berdasarkan kata yang dikenal (mis. "ggwp" cross-language).

### 3.5 Pesan singkat (`gg`, `ez`, `?`, `!`, `noob`)

- `"gg"` → `positive` (default, akhir game) atau `negative` (mid-game early-GG, lihat `match_minute`).
- `"?"` / `"!"` → `neutral`.
- `"ez"` → tergantung outcome di kolom: kalau menang dan tier rendah → mungkin sarkastik trash-talk → `toxic=1`.
- `"noob"` → `toxic=1, insult=1, negative`.

### 3.6 Spam karakter / autis-text

Pesan seperti `"aaaaaaaaaaa"`, `"......."`, `"!!!!"`:
- Sentimen: bergantung konteks. Default `negative` jika di akhir match yang dimenangkan lawan; `neutral` lainnya.
- Toksisitas: 0 kecuali jelas marah/bermusuhan.

---

## 4. Workflow Anotasi

### 4.1 Pilot (200 pesan, full overlap 3 anotator)

1. Buka `data/gold/sample.csv`, ambil 200 baris pertama (sudah di-shuffle deterministik).
2. Anotator A, B, C masing-masing menganotasi seluruh 200 pesan **secara independen** (tidak diskusi sebelum selesai).
3. Setelah selesai, jalankan `notebooks/03_annotation_review.ipynb` untuk hitung kappa pair-wise.
4. Diskusi adjudikasi setiap disagreement; revisi dokumen ini jika perlu.
5. Jika kappa pair-wise sentimen ≥ 0.6 dan Krippendorff's alpha toksisitas ≥ 0.55 → lanjut produksi. Kalau tidak, ulangi pilot dengan 100 pesan baru setelah revisi guideline.

### 4.2 Produksi (sisa 7.800 pesan)

Mode produksi ditentukan dari hasil pilot. Lihat `data/gold/agreement_report.md` untuk keputusan mode (full-overlap atau 2-of-3 dengan adjudikasi).

### 4.3 Pencatatan

- Isi kolom `annotator_id` dengan inisial Anda (`DD`, `DI`, `AK`).
- Simpan file Anda dengan suffix: `sample_<initial>.csv` (mis. `sample_DD.csv`).
- Commit file Anda ke folder `data/gold/raw_annotations/` (gitignored — ini file kerja, bukan output final).
- Jangan modifikasi kolom konten (`match_id`, `time`, `key`, dst.) — hanya isi kolom anotasi.

### 4.4 Pause & istirahat

Direkomendasikan: anotasi maksimal **2 jam berturut-turut** lalu istirahat 30 menit. Kelelahan kognitif menurunkan reliability. Target produksi: ~50-100 pesan/hari/anotator (selesai dalam 2-3 minggu).

---

## 5. FAQ Cepat

**Q**: Pesan "ez" tanpa konteks lebih lanjut, beri label apa?
**A**: Default `neutral` jika tidak ada konteks pertandingan jelas. Cek kolom `match_outcome_for_player` dan `match_minute` untuk konteks. Jika menang dan kolom outcome=`win`, kemungkinan sarkastik → `toxic=1, insult=1`.

**Q**: Pesan dalam huruf besar semua "WTF GG"?
**A**: Capitalization meningkatkan intensitas. `"WTF"` = `obscene=1`. `"GG"` di capslock akhir match menang = `positive`; di mid-match kalah = `negative` defeatism.

**Q**: Emoticon teks `:D`, `:(`, `xD`?
**A**: Tambah weight ke sentimen (positif untuk smile, negatif untuk frown). Tidak mempengaruhi toksisitas.

**Q**: Pesan satu kata jargon teknis Dota?
**A**: `is_dota_jargon=1`. Sentimen `neutral` kecuali jelas emosional.

**Q**: Quote/kutipan pesan teammate?
**A**: Anotasi sesuai isi yang dikutip. Jika anotator quote untuk mengejek, bisa toksik tergantung tone.

---

## 6. Riwayat Revisi

| Versi | Tanggal     | Perubahan                                          |
|-------|-------------|----------------------------------------------------|
| 1.0   | 2026-04-28  | Versi awal — definisi 3-class sentimen, 6-label toxicity. |
