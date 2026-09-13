import os
import torch
import torch.nn as nn
import streamlit as st

from PIL import Image
from torchvision import transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "best_model.pth"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Real",
    "Fake"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔍 AI Image Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether an image is Real or AI-generated'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

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
# LOAD MODEL
# ============================================================

def load_model():
    model = models.resnet18(
        weights=None
    )

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

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image, model):

    image = image.convert("RGB")

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
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Model file not found!"
    )

    st.write(
        f"Expected model location:"
    )

    st.code(MODEL_PATH)

    st.stop()


model = load_model()


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# DISPLAY + PREDICT
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    if st.button(
        "🔍 Analyze Image",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            prediction, confidence, real_prob, fake_prob = (
                predict_image(
                    image,
                    model
                )
            )

        st.markdown("---")

        st.subheader(
            "Prediction Result"
        )

        if prediction == "Real":

            st.success(
                f"✅ REAL IMAGE"
            )

        else:

            st.error(
                f"⚠️ AI-GENERATED / FAKE IMAGE"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.markdown(
            "### Class Probabilities"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Real",
                f"{real_prob:.2f}%"
            )

        with col2:

            st.metric(
                "Fake",
                f"{fake_prob:.2f}%"
            )

        st.progress(
            int(confidence)
        )

        st.caption(
            "Prediction generated using your "
            "trained ResNet18 model."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Image Detection System | ResNet18"
)