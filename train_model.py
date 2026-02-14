"""
Script principal para entrenar el modelo de clasificación de residuos.
"""

import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from model import WasteClassificationModel
from preprocessing import DataPreprocessor
from train import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación de residuos')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directorio con datos de entrenamiento')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directorio para guardar modelos')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño de lote')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Tamaño de imagen')
    parser.add_argument('--architecture', type=str, default='custom_cnn',
                       choices=['custom_cnn', 'mobilenet', 'resnet', 'efficientnet'],
                       help='Arquitectura del modelo')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Tasa de aprendizaje')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Proporción de datos para validación')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Desactivar data augmentation')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SISTEMA DE CLASIFICACIÓN DE RESIDUOS CON IA")
    print("="*70)
    print("\nConfiguración:")
    print(f"  Directorio de datos: {args.data_dir}")
    print(f"  Directorio de salida: {args.output_dir}")
    print(f"  Épocas: {args.epochs}")
    print(f"  Tamaño de lote: {args.batch_size}")
    print(f"  Tamaño de imagen: {args.img_size}x{args.img_size}")
    print(f"  Arquitectura: {args.architecture}")
    print(f"  Tasa de aprendizaje: {args.learning_rate}")
    print(f"  Data augmentation: {not args.no_augmentation}")
    print("="*70 + "\n")
    
    # Verificar que existe el directorio de datos
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"ERROR: El directorio {args.data_dir} no existe")
        print("Por favor, organiza tus datos en el directorio especificado")
        return
    
    # Crear preprocesador
    print("Inicializando preprocesador...")
    preprocessor = DataPreprocessor(
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Obtener número de clases
    try:
        class_names = preprocessor.get_class_names(args.data_dir)
        num_classes = len(class_names)
        print(f"Clases detectadas ({num_classes}): {class_names}\n")
    except Exception as e:
        print(f"ERROR: No se pudieron obtener las clases: {e}")
        return
    
    # Crear generadores de datos
    print("Creando generadores de datos...")
    try:
        train_gen, val_gen = preprocessor.create_data_generators(
            args.data_dir,
            validation_split=args.validation_split,
            use_augmentation=not args.no_augmentation
        )
    except Exception as e:
        print(f"ERROR: No se pudieron crear los generadores: {e}")
        return
    
    # Crear modelo
    print("\nCreando modelo...")
    model_builder = WasteClassificationModel(
        num_classes=num_classes,
        img_size=(args.img_size, args.img_size),
        architecture=args.architecture
    )
    
    # Construir y compilar modelo
    model = model_builder.build()
    model_builder.compile_model(learning_rate=args.learning_rate)
    
    print("\nResumen del modelo:")
    model_builder.get_model_summary()
    
    # Crear entrenador
    print("\nPreparando entrenamiento...")
    trainer = ModelTrainer(model_builder, output_dir=args.output_dir)
    
    # Entrenar modelo
    history = trainer.train(
        train_gen,
        val_gen,
        epochs=args.epochs,
        model_name=f"waste_classifier_{args.architecture}"
    )
    
    # Graficar historial
    print("\nGenerando gráficas de entrenamiento...")
    plot_path = Path(args.output_dir) / f"training_history_{args.architecture}.png"
    trainer.plot_training_history(save_path=str(plot_path))
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
    print("="*70)
    print(f"\nModelo guardado en: {args.output_dir}/")
    print(f"Gráficas guardadas en: {plot_path}")
    print("\nPuedes usar el modelo entrenado con el script 'predict.py'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
