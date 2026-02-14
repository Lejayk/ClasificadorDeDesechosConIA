"""
Script para evaluar el modelo de clasificación de residuos.
"""

import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from evaluation import ModelEvaluator
from preprocessing import DataPreprocessor


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo de clasificación de residuos')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Directorio con datos de test')
    parser.add_argument('--model', type=str, default='models/waste_classifier_custom_cnn_best.h5',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--classes', type=str, default='models/waste_classifier_custom_cnn_classes.json',
                       help='Ruta al archivo de clases')
    parser.add_argument('--output-dir', type=str, default='models/evaluation',
                       help='Directorio para guardar resultados de evaluación')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño de lote')
    parser.add_argument('--img-size', type=int, default=64,
                       help='Tamaño de imagen')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EVALUACIÓN DEL MODELO DE CLASIFICACIÓN DE RESIDUOS")
    print("="*70)
    
    # Verificar archivos
    if not Path(args.test_dir).exists():
        print(f"\nERROR: El directorio {args.test_dir} no existe")
        return
    
    if not Path(args.model).exists():
        print(f"\nERROR: El modelo {args.model} no existe")
        return
    
    if not Path(args.classes).exists():
        print(f"\nERROR: El archivo de clases {args.classes} no existe")
        return
    
    # Crear directorio de salida
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo
    print(f"\nCargando modelo desde: {args.model}")
    evaluator = ModelEvaluator(args.model, args.classes)
    
    # Crear generador de datos de test
    print(f"Cargando datos de test desde: {args.test_dir}")
    preprocessor = DataPreprocessor(
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    test_generator = preprocessor.create_test_generator(args.test_dir)
    
    # Evaluar modelo
    print("\nEvaluando modelo en datos de test...")
    metrics = evaluator.evaluate(test_generator)
    
    # Generar reporte de clasificación
    print("\nGenerando reporte de clasificación...")
    report_path = output_path / "classification_report.txt"
    evaluator.generate_classification_report(test_generator, save_path=str(report_path))
    
    # Graficar matriz de confusión
    print("\nGenerando matriz de confusión...")
    cm_path = output_path / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(test_generator, save_path=str(cm_path))
    
    # Analizar accuracy por clase
    print("\nAnalizando accuracy por clase...")
    per_class_path = output_path / "per_class_accuracy.png"
    evaluator.plot_per_class_accuracy(test_generator, save_path=str(per_class_path))

    # Analizar confusiones más frecuentes
    print("\nAnalizando confusiones más frecuentes...")
    confusion_report_path = output_path / "top_confusions.txt"
    evaluator.analyze_top_confusions(test_generator, top_n=8, save_path=str(confusion_report_path))
    
    print("\n" + "="*70)
    print("EVALUACIÓN COMPLETADA")
    print("="*70)
    print(f"\nResultados guardados en: {args.output_dir}/")
    print(f"  - Reporte de clasificación: {report_path.name}")
    print(f"  - Matriz de confusión: {cm_path.name}")
    print(f"  - Accuracy por clase: {per_class_path.name}")
    print(f"  - Top confusiones: {confusion_report_path.name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
