"""
Orquestador de pipeline completo: split train/test, entrenamiento y evaluación.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_raw_dataset(raw_dir: Path) -> dict[str, int]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de dataset: {raw_dir}")

    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(
            f"No se encontraron carpetas de clases en {raw_dir}. "
            "Debes organizar imágenes por clase (ej: plastico/, papel/, vidrio/)."
        )

    summary: dict[str, int] = {}
    for class_dir in class_dirs:
        files = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        if len(files) < 2:
            raise ValueError(
                f"La clase '{class_dir.name}' necesita al menos 2 imágenes para crear split train/test."
            )
        summary[class_dir.name] = len(files)

    return summary


def create_split(
    raw_dir: Path,
    split_root: Path,
    test_size: float,
    random_seed: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_dir = split_root / "train"
    test_dir = split_root / "test"

    if overwrite and split_root.exists():
        shutil.rmtree(split_root)

    split_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    for class_dir in class_dirs:
        files = sorted([
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ])

        train_files, test_files = train_test_split(
            files,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
        )

        class_train = train_dir / class_dir.name
        class_test = test_dir / class_dir.name
        class_train.mkdir(parents=True, exist_ok=True)
        class_test.mkdir(parents=True, exist_ok=True)

        for file_path in train_files:
            shutil.copy2(file_path, class_train / file_path.name)

        for file_path in test_files:
            shutil.copy2(file_path, class_test / file_path.name)

    return train_dir, test_dir


def run_command(command: list[str], step_name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"{step_name}")
    print(f"{'=' * 70}")
    print("Comando:")
    print(" ".join(command))

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Falló el paso '{step_name}' con código {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo de residuos: split + train + evaluate")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Dataset base por carpetas de clase")
    parser.add_argument("--split-root", type=str, default="data/processed/split", help="Directorio de split train/test")
    parser.add_argument("--test-size", type=float, default=0.25, help="Proporción de test")
    parser.add_argument("--random-seed", type=int, default=42, help="Semilla para split")
    parser.add_argument("--overwrite-split", action="store_true", help="Regenera split eliminando el anterior")
    parser.add_argument("--epochs", type=int, default=20, help="Épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--model-output", type=str, default="models/waste_classifier.h5", help="Ruta de modelo")
    parser.add_argument("--history-output", type=str, default="models/training_history.csv", help="Ruta historial")
    parser.add_argument("--classes-output", type=str, default="models/class_indices.json", help="Ruta de clases")
    parser.add_argument("--evaluation-output", type=str, default="models/evaluation", help="Directorio resultados de evaluación")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    split_root = Path(args.split_root)

    summary = validate_raw_dataset(raw_dir)
    train_dir, test_dir = create_split(
        raw_dir=raw_dir,
        split_root=split_root,
        test_size=args.test_size,
        random_seed=args.random_seed,
        overwrite=args.overwrite_split,
    )

    split_report = {
        "raw_dir": str(raw_dir),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "class_distribution_raw": summary,
        "test_size": args.test_size,
        "random_seed": args.random_seed,
    }

    split_report_path = Path("models") / "split_report.json"
    split_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_report_path, "w", encoding="utf-8") as report_file:
        json.dump(split_report, report_file, indent=2, ensure_ascii=False)

    python_exec = sys.executable

    train_cmd = [
        python_exec,
        "train_model.py",
        "--data-dir", str(train_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--model-output", args.model_output,
        "--history-output", args.history_output,
        "--classes-output", args.classes_output,
    ]

    eval_cmd = [
        python_exec,
        "evaluate_model.py",
        "--test-dir", str(test_dir),
        "--model", args.model_output,
        "--classes", args.classes_output,
        "--history", args.history_output,
        "--output-dir", args.evaluation_output,
        "--batch-size", str(args.batch_size),
    ]

    run_command(train_cmd, "PASO 1/2 - ENTRENAMIENTO")
    run_command(eval_cmd, "PASO 2/2 - EVALUACIÓN")

    print("\nPipeline completado exitosamente")
    print(f"Split report: {split_report_path}")
    print(f"Modelo: {args.model_output}")
    print(f"Evaluación: {args.evaluation_output}")


if __name__ == "__main__":
    main()
