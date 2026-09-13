import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_CSV = os.path.join(ROOT_DIR, "results", "train.csv")
VAL_CSV = os.path.join(ROOT_DIR, "results", "val.csv")
TEST_CSV = os.path.join(ROOT_DIR, "results", "test.csv")

MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

FC_LR = 1e-4
BACKBONE_LR = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AI IMAGE DETECTION - TRAINING")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE.type == "cpu":
    print("WARNING: CUDA is not available.")
    print("Training will run on CPU and may take longer.")


# ============================================================
# DATASET
# ============================================================

class AIImageDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        print(f"\nLoaded: {csv_file}")
        print(f"Images: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["image_path"]
        label = int(row["label"])

        if not os.path.isabs(image_path):
            image_path = os.path.join(
                ROOT_DIR,
                image_path
            )

        image_path = os.path.normpath(image_path)

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"\nCould not load image:\n"
                f"{image_path}\n"
                f"Error: {e}"
            )

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = AIImageDataset(
    TRAIN_CSV,
    train_transform
)

val_dataset = AIImageDataset(
    VAL_CSV,
    val_transform
)

test_dataset = AIImageDataset(
    TEST_CSV,
    val_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(
    f"Training images   : {len(train_dataset)}"
)

print(
    f"Validation images : {len(val_dataset)}"
)

print(
    f"Testing images    : {len(test_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# ------------------------------------------------------------
# FREEZE EVERYTHING FIRST
# ------------------------------------------------------------

for parameter in model.parameters():
    parameter.requires_grad = False


# ------------------------------------------------------------
# UNFREEZE LAST RESNET BLOCK
# ------------------------------------------------------------

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# ------------------------------------------------------------
# REPLACE CLASSIFIER
# ------------------------------------------------------------

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    [
        {
            "params": model.layer4.parameters(),
            "lr": 1e-5
        },

        {
            "params": model.fc.parameters(),
            "lr": 1e-4
        }
    ]
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    predictions = []
    actual_labels = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        predictions.extend(
            predicted.cpu().numpy()
        )

        actual_labels.extend(
            labels.cpu().numpy()
        )

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    average_loss = (
        total_loss / len(train_loader)
    )

    return average_loss, accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0

    predictions = []
    actual_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            actual_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    average_loss = (
        total_loss / len(val_loader)
    )

    return average_loss, accuracy


# ============================================================
# TRAIN MODEL
# ============================================================

best_val_accuracy = 0.0

model_path = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)


print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = (
        validate()
    )

    scheduler.step(
        val_accuracy
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"Best model saved -> {model_path}"
        )


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("TESTING BEST MODEL")
print("=" * 60)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=DEVICE
    )
)

model.eval()

predictions = []
actual_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        predictions.extend(
            predicted.cpu().numpy()
        )

        actual_labels.extend(
            labels.numpy()
        )


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    actual_labels,
    predictions
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predictions,
        target_names=[
            "Real",
            "Fake"
        ],
        digits=4
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Model saved at: "
    f"{model_path}"
)