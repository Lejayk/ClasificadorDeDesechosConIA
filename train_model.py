"""
Entrenamiento de clasificador de residuos con Transfer Learning (MobileNetV2).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador de residuos")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directorio con dataset por carpetas de clase")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--model-output", type=str, default="models/waste_classifier.h5", help="Ruta de salida del modelo")
    parser.add_argument("--history-output", type=str, default="models/training_history.csv", help="Ruta de salida del historial")
    parser.add_argument("--classes-output", type=str, default="models/class_indices.json", help="Ruta de salida del mapeo de clases")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_dir}")

    model_output = Path(args.model_output)
    history_output = Path(args.history_output)
    classes_output = Path(args.classes_output)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    history_output.parent.mkdir(parents=True, exist_ok=True)
    classes_output.parent.mkdir(parents=True, exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    model = build_model(num_classes=train_generator.num_classes)
    model.summary()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        verbose=1
    )

    model.save(str(model_output))

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_output, index=False)

    index_to_class = {v: k for k, v in train_generator.class_indices.items()}
    with open(classes_output, "w", encoding="utf-8") as f:
        json.dump(index_to_class, f, indent=2, ensure_ascii=False)

    print("\nEntrenamiento finalizado")
    print(f"Modelo guardado en: {model_output}")
    print(f"Historial guardado en: {history_output}")
    print(f"Clases guardadas en: {classes_output}")


if __name__ == "__main__":
    main()
