"""
Aplicación web con Streamlit para clasificación de residuos.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/waste_classifier.h5")
CLASSES_PATH = Path("models/class_indices.json")


def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    image_rgb = uploaded_image.convert("RGB")
    image_np = np.array(image_rgb)
    image_resized = cv2.resize(image_np, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    return np.expand_dims(image_normalized, axis=0)


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path)


def load_class_names(classes_path: Path):
    if not classes_path.exists():
        return None
    with open(classes_path, "r", encoding="utf-8") as f:
        index_to_class = json.load(f)
    return [index_to_class[str(i)] for i in range(len(index_to_class))]


def main() -> None:
    st.set_page_config(page_title="Clasificador de Residuos", page_icon="🗑️", layout="centered")
    st.title("🗑️ Clasificador de Residuos con IA")
    st.write("Carga una imagen y el sistema predecirá la categoría del residuo con su confianza.")

    model = load_model(MODEL_PATH)
    class_names = load_class_names(CLASSES_PATH)

    if model is None:
        st.error("No se encontró el modelo en models/waste_classifier.h5. Entrena primero con train_model.py.")
        return

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)

        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor, verbose=0)[0]

        top_index = int(np.argmax(predictions))
        confidence = float(predictions[top_index]) * 100

        if class_names and top_index < len(class_names):
            predicted_class = class_names[top_index]
        else:
            predicted_class = f"Clase_{top_index}"

        st.subheader("Resultado")
        st.success(f"Clase predicha: {predicted_class}")
        st.info(f"Confianza: {confidence:.2f}%")

        st.subheader("Distribución de probabilidades")
        if class_names and len(class_names) == len(predictions):
            chart_data = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
        else:
            chart_data = {f"Clase_{i}": float(predictions[i]) for i in range(len(predictions))}

        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
