import kagglehub
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Muat kredensial Kaggle dari file .env
load_dotenv()

# Validasi
if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
    raise EnvironmentError(
        "KAGGLE_USERNAME atau KAGGLE_KEY tidak ditemukan di file .env!"
    )

# Download
path = kagglehub.competition_download('jigsaw-toxic-comment-classification-challenge')
print("Path to competition files:", path)

# Pindahkan ke data/external/jigsaw_toxic/
output_dir = Path("data/external/jigsaw_toxic")
output_dir.mkdir(parents=True, exist_ok=True)

src = Path(path)
for f in src.rglob("*"):
    if f.is_file():
        target = output_dir / f.name
        if not target.exists():
            shutil.copy2(f, target)
            print(f"  -> {f.name} [OK]")
        else:
            print(f"  -> {f.name} [SKIP, sudah ada]")

print(f"\nProses selesai. Output: {output_dir.resolve()}")
