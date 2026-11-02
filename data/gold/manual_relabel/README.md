# Anotasi Manual — Petunjuk Annotator

File: `annotate_daniel.xlsx`, `annotate_dhitan.xlsx`, `annotate_aldiaz.xlsx` (±1.010 pesan per orang).

1. Baca dulu `../ANNOTATION_GUIDELINE.md` (definisi label, contoh, aturan jargon & sarkasme).
2. Kerjakan **sendiri-sendiri**. Jangan diskusi label, jangan lihat file orang lain, jangan buka `gold_final.csv`.
3. Kolom konteks (jangan diubah): `message`, `end_of_match` (yes = 20% terakhir durasi match), `match_minute` (negatif = sebelum horn / pre-game), `match_length_min`, `player_team_result` (win/loss tim pengirim), `year`, `tier`, `n_occurrences` (berapa kali pesan ini muncul dengan konteks sama — label kamu berlaku untuk semuanya).
4. Isi (dropdown):
   - `sentiment`: positive / neutral / negative
   - `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`: 0 / 1
   - `is_dota_jargon`: 1 kalau >50% token adalah jargon Dota (mis. "rosh smoke 5 min", "bot rune up")
   - `is_ambiguous`: 1 kalau ragu, lalu tulis alasan di `notes`
5. Semua baris wajib terisi (kecuali `notes`). Jangan sort/hapus baris; `row_id` jangan diubah.
6. Selesai → kirim file `.xlsx` apa adanya.

Sebagian pesan sengaja muncul di ketiga file (untuk menghitung kesepakatan antar-annotator). Kerjakan seperti biasa.
