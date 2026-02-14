# Guía de Usuario - Sistema de Clasificación de Residuos con IA

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Instalación](#instalación)
4. [Preparación de Datos](#preparación-de-datos)
5. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
6. [Evaluación del Modelo](#evaluación-del-modelo)
7. [Uso del Sistema de Detección](#uso-del-sistema-de-detección)
8. [API de Referencia](#api-de-referencia)
9. [Solución de Problemas](#solución-de-problemas)

---

## Introducción

Este sistema utiliza técnicas de inteligencia artificial y visión artificial para detectar y clasificar diferentes tipos de residuos a través de imágenes. El sistema es capaz de reconocer las siguientes categorías de residuos:

- **Plástico**: Botellas, envases, bolsas plásticas
- **Papel**: Periódicos, documentos, cartulina
- **Vidrio**: Botellas, frascos, cristales
- **Orgánico**: Restos de comida, cáscaras, desechos biodegradables
- **Metal**: Latas, alambres, envases metálicos
- **Cartón**: Cajas, empaques de cartón

## Requisitos del Sistema

### Hardware
- CPU: Procesador multinúcleo (Intel Core i5 o superior recomendado)
- RAM: Mínimo 8 GB (16 GB recomendado)
- GPU: Opcional pero recomendado para entrenamiento (NVIDIA con soporte CUDA)
- Almacenamiento: Mínimo 5 GB de espacio libre

### Software
- Python 3.8 o superior
- Sistema operativo: Windows 10/11, Linux, o macOS

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Lejayk/ClasificadorDeDesechosConIA.git
cd ClasificadorDeDesechosConIA
```

### 2. Crear Entorno Virtual (Recomendado)

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## Preparación de Datos

### Estructura de Directorios

Organiza tus imágenes de entrenamiento en la siguiente estructura:

```
data/
└── raw/
    ├── plastico/
    │   ├── imagen1.jpg
    │   ├── imagen2.jpg
    │   └── ...
    ├── papel/
    │   ├── imagen1.jpg
    │   └── ...
    ├── vidrio/
    │   └── ...
    ├── organico/
    │   └── ...
    ├── metal/
    │   └── ...
    └── carton/
        └── ...
```

### Recomendaciones para Datos

- **Cantidad mínima**: Al menos 100-200 imágenes por categoría
- **Cantidad recomendada**: 500+ imágenes por categoría para mejores resultados
- **Formato**: JPG, PNG, o JPEG
- **Resolución**: No es crítica, las imágenes se redimensionan automáticamente
- **Variedad**: Incluye diferentes ángulos, iluminaciones y contextos

### Crear Estructura de Directorios

```bash
python -c "from src.data_collection import DataCollector; DataCollector().create_category_directories()"
```

## Entrenamiento del Modelo

### Entrenamiento Básico

```bash
python train_model.py --data-dir data/raw --epochs 50
```

### Opciones Avanzadas

```bash
python train_model.py \
    --data-dir data/raw \
    --output-dir models \
    --epochs 100 \
    --batch-size 32 \
    --img-size 224 \
    --architecture custom_cnn \
    --learning-rate 0.001 \
    --validation-split 0.2
```

### Arquitecturas Disponibles

1. **custom_cnn** (por defecto): CNN personalizada, ligera y eficiente
2. **mobilenet**: Transfer learning con MobileNetV2, rápida en inferencia
3. **resnet**: Transfer learning con ResNet50, mayor precisión
4. **efficientnet**: Transfer learning con EfficientNetB0, balance óptimo

### Ejemplo con Transfer Learning

```bash
python train_model.py \
    --data-dir data/raw \
    --architecture mobilenet \
    --epochs 30
```

### Monitoreo del Entrenamiento

Durante el entrenamiento, se generan:
- Logs de TensorBoard en `models/logs/`
- Modelos guardados en `models/`
- Historial de entrenamiento en formato JSON
- Gráficas de métricas (accuracy, loss, precision, recall)

Para visualizar con TensorBoard:

```bash
tensorboard --logdir models/logs
```

## Evaluación del Modelo

### Evaluación Básica

```bash
python evaluate_model.py --test-dir data/test
```

### Opciones de Evaluación

```bash
python evaluate_model.py \
    --test-dir data/test \
    --model models/waste_classifier_custom_cnn_best.h5 \
    --classes models/waste_classifier_custom_cnn_classes.json \
    --output-dir models/evaluation
```

### Métricas Generadas

- **Accuracy global**: Precisión general del modelo
- **Precision y Recall**: Por clase
- **Matriz de confusión**: Visualización de errores
- **Accuracy por clase**: Rendimiento individual de cada categoría

## Uso del Sistema de Detección

### Clasificar una Imagen

```bash
python predict.py --image ruta/a/imagen.jpg
```

### Opciones de Predicción

```bash
# Con visualización
python predict.py \
    --image test_image.jpg \
    --output resultado.png \
    --top-k 3

# Con umbral de confianza
python predict.py \
    --image test_image.jpg \
    --threshold 0.8
```

### Usar Modelo Personalizado

```bash
python predict.py \
    --image test_image.jpg \
    --model models/mi_modelo.h5 \
    --classes models/mi_modelo_classes.json
```

## API de Referencia

### Módulo de Detección (detection.py)

```python
from src.detection import WasteDetector

# Inicializar detector
detector = WasteDetector(
    model_path='models/waste_classifier_custom_cnn_best.h5',
    class_mapping_path='models/waste_classifier_custom_cnn_classes.json'
)

# Clasificar imagen
results = detector.predict('imagen.jpg', top_k=3)

# Resultado con umbral
result = detector.classify_with_threshold('imagen.jpg', confidence_threshold=0.7)

# Predicción en lote
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.batch_predict(image_paths)
```

### Módulo de Entrenamiento (train.py)

```python
from src.model import WasteClassificationModel
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer

# Crear preprocesador
preprocessor = DataPreprocessor(img_size=(224, 224), batch_size=32)
train_gen, val_gen = preprocessor.create_data_generators('data/raw')

# Crear y entrenar modelo
model_builder = WasteClassificationModel(num_classes=6)
model = model_builder.build()
model_builder.compile_model()

trainer = ModelTrainer(model_builder)
history = trainer.train(train_gen, val_gen, epochs=50)
```

## Solución de Problemas

### Error: "No se pudo cargar la imagen"

**Causa**: Archivo de imagen corrupto o formato no soportado

**Solución**: 
- Verifica que la imagen sea JPG, PNG, o JPEG
- Intenta abrir la imagen con otro programa para verificar su integridad

### Error: "El modelo no ha sido construido"

**Causa**: Intentar compilar o entrenar sin construir el modelo primero

**Solución**: 
```python
model = model_builder.build()  # Primero construir
model_builder.compile_model()  # Luego compilar
```

### Bajo Rendimiento del Modelo

**Causas posibles**:
1. Datos insuficientes
2. Datos de baja calidad
3. Clases desbalanceadas

**Soluciones**:
1. Recopilar más imágenes (500+ por clase)
2. Usar data augmentation (activado por defecto)
3. Entrenar por más épocas
4. Probar diferentes arquitecturas (mobilenet, resnet)

### Error de Memoria (Out of Memory)

**Solución**:
```bash
# Reducir batch size
python train_model.py --batch-size 16

# Reducir tamaño de imagen
python train_model.py --img-size 128
```

### GPU No Detectada

**Verificar CUDA**:
```python
import tensorflow as tf
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
```

**Solución**: Instalar CUDA Toolkit y cuDNN compatibles con tu versión de TensorFlow

## Mejores Prácticas

1. **División de datos**: 70% entrenamiento, 15% validación, 15% test
2. **Augmentación**: Siempre activada para entrenamiento
3. **Early stopping**: El sistema para automáticamente si no hay mejora
4. **Guardar modelos**: El mejor modelo se guarda automáticamente
5. **Validación cruzada**: Evalúa con datos nunca vistos por el modelo

## Contacto y Soporte

Para reportar problemas o solicitar ayuda:
- Crear un issue en GitHub
- Revisar la documentación técnica en `docs/`

---

**Versión**: 1.0.0  
**Última actualización**: Febrero 2026
