"""
Entrenamiento de clasificador de residuos con Transfer Learning (MobileNetV2).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
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

    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


def create_callbacks(patience: int) -> list:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1
        )
    ]


def merge_histories(first: dict, second: dict) -> dict:
    merged = {key: list(value) for key, value in first.items()}

    for key, value in second.items():
        if key not in merged:
            merged[key] = []
        merged[key].extend(list(value))

    return merged


def compute_class_weights(train_generator) -> dict[int, float]:
    classes = train_generator.classes
    unique_classes = np.array(sorted(set(classes.tolist())), dtype=np.int64)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes
    )
    return {int(class_id): float(weight) for class_id, weight in zip(unique_classes, weights)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador de residuos")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directorio con dataset por carpetas de clase")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--base-learning-rate", type=float, default=1e-3, help="Learning rate fase 1 (cabeza)")
    parser.add_argument("--fine-tune", action=argparse.BooleanOptionalAction, default=True, help="Activa/desactiva fine-tuning")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="Épocas fase 2 (fine-tuning)")
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5, help="Learning rate fase 2")
    parser.add_argument("--unfreeze-layers", type=int, default=30, help="Número de capas finales de MobileNetV2 a descongelar")
    parser.add_argument("--patience", type=int, default=6, help="Paciencia para EarlyStopping")
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True, help="Usar pesos de clase balanceados")
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
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        brightness_range=(0.85, 1.15),
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

    model, base_model = build_model(num_classes=train_generator.num_classes)
    compile_model(model, args.base_learning_rate)
    model.summary()

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(train_generator)
        print("\nPesos de clase (balanceados):")
        for class_name, class_index in sorted(train_generator.class_indices.items(), key=lambda item: item[1]):
            print(f"- {class_name}: {class_weight[class_index]:.4f}")

    print("\nFase 1/2: entrenamiento de la cabeza clasificadora")
    phase_1_history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=create_callbacks(args.patience),
        class_weight=class_weight,
        verbose=1
    )

    full_history = phase_1_history.history

    if args.fine_tune and args.fine_tune_epochs > 0:
        total_layers = len(base_model.layers)
        unfreeze_layers = min(max(args.unfreeze_layers, 1), total_layers)

        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        compile_model(model, args.fine_tune_learning_rate)

        print("\nFase 2/2: fine-tuning de capas finales de MobileNetV2")
        print(f"Capas descongeladas: {unfreeze_layers}/{total_layers}")

        phase_2_history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
            callbacks=create_callbacks(args.patience),
            class_weight=class_weight,
            verbose=1
        )

        full_history = merge_histories(full_history, phase_2_history.history)

    model.save(str(model_output))

    history_df = pd.DataFrame(full_history)
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
