import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# =========================
# CONFIG
# =========================

CSV_PATH = Path("results/dataset.csv")
OUTPUT_DIR = Path("results")

RANDOM_STATE = 42


# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Total images: {len(df)}")
print("\nClass distribution:")
print(df["class"].value_counts())


# =========================
# TRAIN / TEMP SPLIT
# =========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=RANDOM_STATE
)


# =========================
# VALIDATION / TEST SPLIT
# =========================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)


# =========================
# SAVE
# =========================

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)


# =========================
# RESULTS
# =========================

print("\n==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")
print(f"Total:      {len(train_df) + len(val_df) + len(test_df)}")

print("\nTrain distribution:")
print(train_df["class"].value_counts())

print("\nValidation distribution:")
print(val_df["class"].value_counts())

print("\nTest distribution:")
print(test_df["class"].value_counts())