import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(
    page_title="Orange Fruit Disease Detection",
    page_icon="🍊",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_LABELS = {
    "blackspotted_orange": "Black-spotted Orange",
    "cankered_orange": "Cankered Orange",
    "fresh_orange": "Fresh Orange",
    "green_orange": "Green Orange",
    "rotten_orange": "Rotten Orange",
}

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return YOLO(str(MODEL_PATH))

st.title("🍊 Orange Fruit Disease Detection")
st.write(
    "Upload an orange image to predict its dataset category. "
    "This is an academic prototype, not a substitute for expert inspection."
)

model = load_model()
if model is None:
    st.error(
        "Model file not found. Place your trained YOLOv8 classification "
        "checkpoint named best.pt in the same folder as app.py."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an orange image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict category", type="primary", use_container_width=True):
        with st.spinner("Analysing image..."):
            result = model.predict(image, imgsz=224, verbose=False)[0]
            probs = result.probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            raw_name = model.names[class_id]
            friendly_name = CLASS_LABELS.get(raw_name, raw_name.replace("_", " ").title())

        st.subheader("Prediction")
        st.success(f"**{friendly_name}**")
        st.metric("Model confidence", f"{confidence * 100:.2f}%")

        if probs is not None:
            st.subheader("Class scores")
            scores = []
            for idx, score in enumerate(probs.data.tolist()):
                label = model.names[idx]
                scores.append((CLASS_LABELS.get(label, label.replace("_", " ").title()), float(score)))
            for label, score in sorted(scores, key=lambda x: x[1], reverse=True):
                st.write(f"{label}: {score * 100:.2f}%")
                st.progress(max(0.0, min(1.0, score)))
