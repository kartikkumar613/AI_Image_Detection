from pathlib import Path
from PIL import Image
from collections import Counter

REAL_DIR = Path("dataset/raw/content/real_images")
FAKE_DIR = Path("dataset/raw/content/fake_images")


def inspect_folder(folder, name):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"\n{name}")
    print("-" * 50)
    print(f"Images found: {len(files)}")

    dimensions = Counter()
    corrupted = 0

    for file in files[:500]:
        try:
            with Image.open(file) as img:
                dimensions[img.size] += 1
        except Exception:
            corrupted += 1

    print("Sample dimensions:")
    for size, count in dimensions.most_common(10):
        print(f"  {size}: {count}")

    print(f"Corrupted images in sample: {corrupted}")


inspect_folder(REAL_DIR, "REAL IMAGES")
inspect_folder(FAKE_DIR / "stylegan1", "STYLEGAN1")