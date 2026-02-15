"""
Script principal para realizar predicciones con el modelo entrenado.
"""

import argparse
from pathlib import Path

from src.detection import WasteDetector


def main():
    parser = argparse.ArgumentParser(description='Clasificar residuos en imágenes')
    parser.add_argument('--image', type=str, required=True,
                       help='Ruta a la imagen a clasificar')
    parser.add_argument('--model', type=str, default='models/waste_classifier.h5',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--classes', type=str, default='models/class_indices.json',
                       help='Ruta al archivo de clases')
    parser.add_argument('--output', type=str, default=None,
                       help='Ruta para guardar imagen con predicción')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Umbral de confianza mínimo (0.0-1.0)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Número de predicciones principales a mostrar')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Tamaño de imagen para inferencia')
    parser.add_argument('--disable-smoothing', action='store_true',
                       help='Desactivar suavizado defensivo en inferencia')
    parser.add_argument('--smoothing-method', type=str, default='gaussian',
                       choices=['gaussian', 'median'],
                       help='Método de suavizado defensivo')
    
    args = parser.parse_args()

    
    print("\n" + "="*70)
    print("CLASIFICADOR DE RESIDUOS CON IA - PREDICCIÓN")
    print("="*70)
    
    # Verificar que existe la imagen
    if not Path(args.image).exists():
        print(f"\nERROR: La imagen {args.image} no existe")
        return
    
    # Verificar que existe el modelo
    if not Path(args.model).exists():
        print(f"\nERROR: El modelo {args.model} no existe")
        print("Por favor, entrena el modelo primero con 'train_model.py'")
        return
    
    # Verificar que existe el archivo de clases
    if not Path(args.classes).exists():
        print(f"\nERROR: El archivo de clases {args.classes} no existe")
        return
    
    print(f"\nCargando modelo desde: {args.model}")
    print(f"Cargando clases desde: {args.classes}")
    
    # Crear detector
    try:
        detector = WasteDetector(
            model_path=args.model,
            class_mapping_path=args.classes,
            img_size=(args.img_size, args.img_size),
            enable_smoothing=not args.disable_smoothing,
            smoothing_method=args.smoothing_method
        )
    except Exception as e:
        print(f"\nERROR: No se pudo cargar el modelo: {e}")
        return
    
    print(f"\nClasificando imagen: {args.image}")
    print("="*70)
    
    # Realizar predicción
    try:
        if args.threshold > 0:
            # Predicción con umbral
            result = detector.classify_with_threshold(
                args.image,
                confidence_threshold=args.threshold
            )
            
            if result:
                print(f"\n✓ Clasificación exitosa:")
                print(f"  Clase: {result['class'].upper()}")
                print(f"  Confianza: {result['percentage']:.2f}%")
            else:
                print(f"\n✗ La confianza no supera el umbral de {args.threshold*100}%")
        else:
            # Predicción normal
            results = detector.predict(args.image, top_k=args.top_k)
            
            print(f"\n✓ Top {args.top_k} Predicciones:")
            print("-" * 50)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['class'].upper():15} - {result['percentage']:6.2f}%")
            print("-" * 50)
            
            # Mostrar resultado principal
            top_result = results[0]
            print(f"\n🎯 Resultado: {top_result['class'].upper()}")
            print(f"   Confianza: {top_result['percentage']:.2f}%")
        
        # Visualizar si se especifica output
        if args.output:
            print(f"\nGenerando visualización...")
            detector.predict_and_display(args.image, save_path=args.output)
    
    except Exception as e:
        print(f"\nERROR: No se pudo realizar la predicción: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
