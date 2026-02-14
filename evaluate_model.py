"""
Evaluación del clasificador de residuos.
Genera matriz de confusión, precisión, recall, f-score y gráficas de accuracy/loss.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_class_names(classes_path: Path, generator) -> list:
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            index_to_class = json.load(f)
        return [index_to_class[str(i)] for i in range(len(index_to_class))]

    class_indices = generator.class_indices
    return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def plot_history(history_path: Path, output_dir: Path) -> None:
    if not history_path.exists():
        print(f"Aviso: no se encontró historial en {history_path}. Se omiten gráficas de entrenamiento.")
        return

    history_df = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if "accuracy" in history_df.columns:
        axes[0].plot(history_df["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history_df.columns:
        axes[0].plot(history_df["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Evolución de Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    if "loss" in history_df.columns:
        axes[1].plot(history_df["loss"], label="Train Loss")
    if "val_loss" in history_df.columns:
        axes[1].plot(history_df["val_loss"], label="Val Loss")
    axes[1].set_title("Evolución de Loss")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plot_path = output_dir / "training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfica de historial guardada en: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del clasificador de residuos")
    parser.add_argument("--test-dir", type=str, required=True, help="Directorio de test por carpetas de clase")
    parser.add_argument("--model", type=str, default="models/waste_classifier.h5", help="Ruta del modelo entrenado")
    parser.add_argument("--classes", type=str, default="models/class_indices.json", help="Ruta del archivo de clases")
    parser.add_argument("--history", type=str, default="models/training_history.csv", help="Ruta del historial de entrenamiento")
    parser.add_argument("--output-dir", type=str, default="models/evaluation", help="Directorio de salida")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    model_path = Path(args.model)
    classes_path = Path(args.classes)
    history_path = Path(args.history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de test: {test_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False
    )

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = load_class_names(classes_path, test_generator)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nMétricas globales")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F-score:   {fscore:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print("\nReporte de clasificación:\n")
    print(report)

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    metrics_df = pd.DataFrame([
        {"precision": precision, "recall": recall, "fscore": fscore}
    ])
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    plot_history(history_path, output_dir)

    print("\nEvaluación finalizada")
    print(f"Reporte guardado en: {report_path}")
    print(f"Matriz de confusión guardada en: {cm_path}")


if __name__ == "__main__":
    main()
