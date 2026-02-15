"""
Aplicación web con Streamlit para clasificación de residuos.
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.detection import WasteDetector

MODEL_PATH = Path("models/waste_classifier.h5")
CLASSES_PATH = Path("models/class_indices.json")


def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    image_rgb = uploaded_image.convert("RGB")
    image_np = np.array(image_rgb)
    return cv2.resize(image_np, (224, 224))


@st.cache_resource
def load_detector(model_path: Path, classes_path: Path):
    if not model_path.exists() or not classes_path.exists():
        return None
    return WasteDetector(
        model_path=str(model_path),
        class_mapping_path=str(classes_path),
        img_size=(224, 224),
        enable_smoothing=False
    )


def main() -> None:
    st.set_page_config(page_title="Clasificador de Residuos", page_icon="🗑️", layout="centered")
    st.title("🗑️ Clasificador de Residuos con IA")
    st.write("Carga una imagen y el sistema predecirá la categoría del residuo con su confianza.")

    detector = load_detector(MODEL_PATH, CLASSES_PATH)

    if detector is None:
        st.error("No se encontró el modelo o el mapeo de clases en models/. Entrena primero con train_model.py.")
        return

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)

        image_np = preprocess_image(image)
        predictions = detector.predict_array(image_np, top_k=3)

        top_prediction = predictions[0]
        predicted_class = top_prediction["class"]
        confidence = top_prediction["percentage"]

        st.subheader("Resultado")
        st.success(f"Clase predicha: {predicted_class}")
        st.info(f"Confianza: {confidence:.2f}%")

        st.subheader("Distribución de probabilidades")
        chart_data = {item["class"]: float(item["confidence"]) for item in predictions}

        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
