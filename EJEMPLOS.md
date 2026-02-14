# Ejemplos de Uso - Sistema de Clasificación de Residuos

Este archivo contiene ejemplos prácticos de cómo usar el sistema.

## Ejemplo 1: Entrenamiento Rápido

Para entrenar rápidamente con configuración por defecto:

```bash
python train_model.py --data-dir data/raw --epochs 20
```

## Ejemplo 2: Entrenamiento con Transfer Learning

Usar MobileNetV2 para un modelo más preciso:

```bash
python train_model.py \
    --data-dir data/raw \
    --architecture mobilenet \
    --epochs 30 \
    --learning-rate 0.0001
```

## Ejemplo 3: Clasificar una Imagen

Clasificar una imagen y ver las top 3 predicciones:

```bash
python predict.py \
    --image test_images/botella_plastico.jpg \
    --top-k 3
```

## Ejemplo 4: Clasificar con Visualización

Generar una imagen con la predicción visualizada:

```bash
python predict.py \
    --image test_images/botella_plastico.jpg \
    --output resultado_prediccion.png
```

## Ejemplo 5: Clasificación con Umbral

Solo clasificar si la confianza es mayor a 80%:

```bash
python predict.py \
    --image test_images/lata_metal.jpg \
    --threshold 0.8
```

## Ejemplo 6: Evaluación del Modelo

Evaluar el modelo en un conjunto de test:

```bash
python evaluate_model.py \
    --test-dir data/test \
    --output-dir models/evaluation
```

## Ejemplo 7: Usar el Sistema en Python

### Predicción Simple

```python
from src.detection import WasteDetector

# Inicializar detector
detector = WasteDetector(
    model_path='models/waste_classifier_custom_cnn_best.h5',
    class_mapping_path='models/waste_classifier_custom_cnn_classes.json'
)

# Clasificar imagen
results = detector.predict('imagen.jpg', top_k=3)

# Mostrar resultados
for i, result in enumerate(results, 1):
    print(f"{i}. {result['class']}: {result['percentage']:.2f}%")
```

### Predicción en Lote

```python
from src.detection import WasteDetector

detector = WasteDetector(
    model_path='models/waste_classifier_custom_cnn_best.h5',
    class_mapping_path='models/waste_classifier_custom_cnn_classes.json'
)

# Lista de imágenes
images = [
    'imagen1.jpg',
    'imagen2.jpg',
    'imagen3.jpg'
]

# Procesar en lote
results = detector.batch_predict(images)

# Mostrar resultados
for result in results:
    if 'error' not in result:
        pred = result['prediction']
        print(f"{result['image_path']}: {pred['class']} ({pred['percentage']:.1f}%)")
```

### Entrenamiento Personalizado

```python
from src.model import WasteClassificationModel
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer

# Crear preprocesador
preprocessor = DataPreprocessor(
    img_size=(224, 224),
    batch_size=32
)

# Crear generadores
train_gen, val_gen = preprocessor.create_data_generators(
    data_dir='data/raw',
    validation_split=0.2,
    use_augmentation=True
)

# Crear modelo
model_builder = WasteClassificationModel(
    num_classes=6,
    img_size=(224, 224),
    architecture='custom_cnn'
)

# Construir y compilar
model = model_builder.build()
model_builder.compile_model(learning_rate=0.001)

# Entrenar
trainer = ModelTrainer(model_builder, output_dir='models')
history = trainer.train(
    train_gen,
    val_gen,
    epochs=50,
    model_name='mi_modelo'
)

# Visualizar entrenamiento
trainer.plot_training_history(save_path='training_plot.png')
```

### Evaluación Detallada

```python
from src.evaluation import ModelEvaluator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar evaluador
evaluator = ModelEvaluator(
    model_path='models/waste_classifier_custom_cnn_best.h5',
    class_mapping_path='models/waste_classifier_custom_cnn_classes.json'
)

# Crear generador de test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluar
metrics = evaluator.evaluate(test_generator)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Generar reporte
evaluator.generate_classification_report(
    test_generator,
    save_path='classification_report.txt'
)

# Matriz de confusión
evaluator.plot_confusion_matrix(
    test_generator,
    save_path='confusion_matrix.png'
)

# Accuracy por clase
evaluator.plot_per_class_accuracy(
    test_generator,
    save_path='per_class_accuracy.png'
)
```

## Ejemplo 8: Pipeline Completo

Script completo de entrenamiento a predicción:

```python
#!/usr/bin/env python3
"""Pipeline completo de entrenamiento y predicción."""

import sys
sys.path.append('src')

from data_collection import DataCollector
from preprocessing import DataPreprocessor
from model import WasteClassificationModel
from train import ModelTrainer
from detection import WasteDetector

# 1. Verificar datos
print("Verificando datos...")
collector = DataCollector('data/raw')
total, counts = collector.validate_dataset()
print(f"Total de imágenes: {total}")

# 2. Preprocesar
print("\nPreparando datos...")
preprocessor = DataPreprocessor(img_size=(224, 224), batch_size=32)
train_gen, val_gen = preprocessor.create_data_generators('data/raw')

# 3. Crear y entrenar modelo
print("\nEntrenando modelo...")
model_builder = WasteClassificationModel(num_classes=6)
model = model_builder.build()
model_builder.compile_model()

trainer = ModelTrainer(model_builder)
history = trainer.train(train_gen, val_gen, epochs=30)

# 4. Usar modelo para predicción
print("\nProbando modelo...")
detector = WasteDetector(
    'models/waste_classifier_best.h5',
    'models/waste_classifier_classes.json'
)

# Probar con una imagen
result = detector.predict('test_image.jpg', top_k=1)[0]
print(f"\nResultado: {result['class'].upper()} ({result['percentage']:.1f}%)")
```

## Consejos y Mejores Prácticas

1. **Datos balanceados**: Intenta tener similar cantidad de imágenes por clase
2. **Validación**: Siempre reserva datos para test (nunca usados en entrenamiento)
3. **Augmentación**: Úsala para aumentar la variabilidad de los datos
4. **Early stopping**: El sistema para automáticamente si no hay mejora
5. **Learning rate**: Empieza con 0.001, reduce si el modelo no converge
6. **Batch size**: Ajusta según tu memoria RAM/GPU (16, 32, 64)
7. **Épocas**: Empieza con 30-50, el early stopping parará si es necesario

## Solución de Problemas Comunes

### Error: No module named 'tensorflow'
```bash
pip install tensorflow>=2.13.0
```

### Error: Out of memory
```bash
# Reducir batch size
python train_model.py --batch-size 16
```

### Bajo accuracy
- Recopilar más datos
- Entrenar por más épocas
- Probar transfer learning (mobilenet, resnet)
- Verificar que los datos estén bien etiquetados

### Modelo muy lento
- Usar MobileNetV2 para inferencia rápida
- Reducir tamaño de imagen a 128x128
- Optimizar modelo con TensorFlow Lite
