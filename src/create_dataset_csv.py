from pathlib import Path
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Dataset directories
REAL_DIR = ROOT / "dataset" / "raw" / "content" / "real_images"
FAKE_DIR = ROOT / "dataset" / "raw" / "content" / "fake_images"

# Output CSV
OUTPUT_FILE = ROOT / "dataset" / "dataset.csv"

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

records = []

# --------------------------------------------------
# REAL IMAGES
# --------------------------------------------------
print("Scanning real images...")

for image_path in REAL_DIR.rglob("*"):
    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
        records.append({
            "image_path": image_path.relative_to(ROOT).as_posix(),
            "label": 0,
            "class": "real",
            "generator": "real"
        })

print(f"Real images found: {sum(r['label'] == 0 for r in records)}")


# --------------------------------------------------
# AI-GENERATED IMAGES
# --------------------------------------------------
print("\nScanning AI-generated images...")

fake_count = 0

for generator_dir in FAKE_DIR.iterdir():

    if not generator_dir.is_dir():
        continue

    generator_name = generator_dir.name

    for image_path in generator_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:

            records.append({
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "label": 1,
                "class": "fake",
                "generator": generator_name
            })

            fake_count += 1

    print(f"{generator_name}: completed")


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------
df = pd.DataFrame(records)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n========================================")
print("DATASET CSV CREATED SUCCESSFULLY")
print("========================================")

print(f"Total images : {len(df)}")
print(f"Real images  : {(df['label'] == 0).sum()}")
print(f"Fake images  : {(df['label'] == 1).sum()}")

print("\nGenerator distribution:")
print(df["generator"].value_counts())

print(f"\nCSV saved at:")
print(OUTPUT_FILE)