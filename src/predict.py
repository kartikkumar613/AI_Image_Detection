import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "best_model.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["Real", "Fake"]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading ResNet18...")

    model = models.resnet18(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded successfully!")

    return model


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path, model):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    prediction = CLASS_NAMES[predicted_class]

    confidence = (
        probabilities[0][predicted_class].item()
        * 100
    )

    real_probability = (
        probabilities[0][0].item()
        * 100
    )

    fake_probability = (
        probabilities[0][1].item()
        * 100
    )

    return (
        prediction,
        confidence,
        real_probability,
        fake_probability
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/predict.py path_to_image"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):

        print(
            f"\nERROR: Image not found:"
        )

        print(image_path)

        sys.exit(1)

    print("\n" + "=" * 60)
    print("AI IMAGE DETECTION - PREDICTION")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    model = load_model()

    prediction, confidence, real_prob, fake_prob = (
        predict_image(
            image_path,
            model
        )
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"Image      : {image_path}")

    print(f"Prediction : {prediction}")

    print(f"Confidence : {confidence:.2f}%")

    print("\nClass probabilities:")

    print(f"Real : {real_prob:.2f}%")

    print(f"Fake : {fake_prob:.2f}%")

    print("=" * 60)