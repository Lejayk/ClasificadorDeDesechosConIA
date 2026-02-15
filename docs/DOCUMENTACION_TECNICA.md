# Documentación Técnica - Sistema de Clasificación de Residuos

## Arquitectura del Sistema

### Visión General

El sistema está diseñado siguiendo principios de modularidad y separación de responsabilidades:

```
ClasificadorDeDesechosConIA/
├── src/                      # Módulos principales del sistema
│   ├── __init__.py
│   ├── data_collection.py   # Recopilación y organización de datos
│   ├── preprocessing.py     # Preprocesamiento de imágenes
│   ├── model.py             # Definición de arquitecturas de modelos
│   ├── train.py             # Lógica de entrenamiento
│   ├── evaluation.py        # Evaluación y métricas
│   └── detection.py         # Sistema de inferencia
├── data/                    # Datos del proyecto
│   ├── raw/                 # Datos sin procesar
│   └── processed/           # Datos procesados (generados)
├── models/                  # Modelos entrenados y artefactos
├── notebooks/               # Jupyter notebooks para exploración
├── docs/                    # Documentación
├── train_model.py           # Script principal de entrenamiento
├── predict.py               # Script principal de predicción
├── evaluate_model.py        # Script principal de evaluación
├── run_pipeline.py          # Pipeline completo (split + train + evaluate)
└── requirements.txt         # Dependencias del proyecto
```

## Módulos del Sistema

### 1. data_collection.py

**Propósito**: Gestión de recopilación y organización de datos de entrenamiento.

**Clase Principal**: `DataCollector`

**Métodos Clave**:
- `create_category_directories()`: Crea estructura de directorios para categorías
- `validate_dataset()`: Valida integridad del dataset
- `print_dataset_summary()`: Muestra resumen estadístico del dataset

**Categorías Soportadas**:
1. Plástico
2. Papel
3. Vidrio
4. Orgánico
5. Metal
6. Cartón

### 2. preprocessing.py

**Propósito**: Preprocesamiento y augmentación de imágenes.

**Clase Principal**: `DataPreprocessor`

**Funcionalidades**:
- Redimensionamiento de imágenes a tamaño uniforme
- Normalización de píxeles (0-1)
- Conversión de espacios de color (BGR → RGB)
- Data augmentation para aumentar variabilidad

**Técnicas de Augmentación**:
- Rotación (±20°)
- Desplazamiento horizontal/vertical (±20%)
- Zoom (±20%)
- Flip horizontal
- Transformaciones de corte

### 3. model.py

**Propósito**: Definición y construcción de arquitecturas de modelos.

**Clase Principal**: `WasteClassificationModel`

**Arquitecturas Disponibles**:

#### CNN Personalizada (custom_cnn)
```
Input (64x64x3)
    ↓
Conv2D(32, 2x2, same, stride=1) → ReLU → MaxPool(2x2) → Dropout(0.2)
    ↓
Conv2D(32, 2x2, same, stride=1) → ReLU → MaxPool(2x2) → Dropout(0.2)
    ↓
Conv2D(32, 2x2, same, stride=1) → ReLU → MaxPool(2x2) → Dropout(0.2)
    ↓
Flatten
    ↓
Dense(512) → ReLU
    ↓
Dense(num_classes) → Softmax
```

**Parámetros aproximados**: ~5M parámetros

#### Transfer Learning

- **MobileNetV2**: Ligero, optimizado para dispositivos móviles
- **ResNet50**: Profundo, alta precisión
- **EfficientNetB0**: Balance óptimo eficiencia/precisión

### 4. train.py

**Propósito**: Orquestación del proceso de entrenamiento.

**Clase Principal**: `ModelTrainer`

**Callbacks Implementados**:

1. **ModelCheckpoint**: Guarda el mejor modelo
    - Monitor: `val_loss`
    - Modo: Minimizar
   
2. **EarlyStopping**: Detiene entrenamiento si no hay mejora
   - Paciencia: 10 épocas
   - Restaura mejores pesos
   
3. **ReduceLROnPlateau**: Reduce learning rate dinámicamente
   - Factor: 0.5
   - Paciencia: 5 épocas
   
4. **TensorBoard**: Logging para visualización

**Robustez adversarial (FGSM)**:
- Entrenamiento adversarial opcional con `epsilon`, `adv_ratio` y `adv_start_epoch`
- Mezcla de muestras limpias y adversariales por batch
- Clipping de entradas al rango `[0, 1]`

**Proceso de Entrenamiento**:
1. Validación de datos
2. Creación de callbacks
3. Entrenamiento con validación
4. Guardado de modelo y artefactos
5. Generación de gráficas

### 5. evaluation.py

**Propósito**: Evaluación exhaustiva del modelo entrenado.

**Clase Principal**: `ModelEvaluator`

**Métricas Calculadas**:
- **Accuracy**: Precisión global
- **Precision**: Verdaderos positivos / (VP + Falsos positivos)
- **Recall**: Verdaderos positivos / (VP + Falsos negativos)
- **F1-Score**: Media armónica de Precision y Recall
- **Matriz de Confusión**: Visualización de predicciones vs realidad

**Visualizaciones Generadas**:
1. Matriz de confusión (heatmap)
2. Accuracy por clase (gráfico de barras)
3. Reporte de clasificación detallado
4. Top confusiones entre pares de clases

### 6. detection.py

**Propósito**: Sistema de inferencia para clasificación en tiempo real.

**Clase Principal**: `WasteDetector`

**Métodos de Predicción**:

1. `predict()`: Predicción básica con top-k resultados
2. `predict_array()`: Predicción desde imagen en memoria (ideal para UI/API)
3. `predict_and_display()`: Predicción con visualización
4. `batch_predict()`: Predicción en lote
5. `classify_with_threshold()`: Clasificación con umbral de confianza

**Pipeline de Inferencia**:
```
Imagen de entrada
    ↓
Carga de imagen (OpenCV)
    ↓
Validación de imagen y saneamiento de canales
    ↓
Suavizado defensivo (gaussian/median)
    ↓
Redimensionamiento (224x224)
    ↓
Normalización (0-1)
    ↓
Expansión de dimensión batch
    ↓
Predicción del modelo
    ↓
Extracción de top-k clases
    ↓
Formato de resultados
```

## Optimizaciones y Consideraciones

### Rendimiento

**Entrenamiento**:
- Uso de `batch_size` para procesamiento paralelo
- Data augmentation on-the-fly para eficiencia de memoria
- Callbacks para early stopping y ahorro de cómputo
- Entrenamiento adversarial FGSM configurable para robustez

**Inferencia**:
- Modelos optimizados para diferentes escenarios (MobileNet para móviles)
- Preprocesamiento eficiente con NumPy y OpenCV
- Batch prediction para múltiples imágenes

### Escalabilidad

**Datos**:
- Sistema modular permite agregar nuevas categorías fácilmente
- Estructura de directorios extensible
- Validación automática de datasets

**Modelos**:
- Arquitecturas intercambiables
- Soporte para transfer learning
- Fine-tuning de modelos pre-entrenados

### Mantenibilidad

**Código**:
- Documentación exhaustiva (docstrings)
- Separación clara de responsabilidades
- Configuración mediante argumentos CLI
- Logging detallado de procesos

## Formato de Datos

### Imágenes de Entrada

- **Formatos soportados**: JPG, PNG, JPEG
- **Resolución**: Flexible (se redimensiona automáticamente)
- **Canales**: RGB (3 canales)
- **Tamaño procesado**: 64x64 píxeles (configurable)
- **Tamaño procesado**: 224x224 píxeles (configurable)

### Modelos Guardados

**Archivos generados**:
1. `{model_name}_best.h5`: Mejor modelo (val_loss)
2. `{model_name}_final.h5`: Modelo final de entrenamiento
3. `{model_name}_classes.json`: Mapeo índice → clase
4. `{model_name}_history.json`: Historial de entrenamiento

**Formato de clases.json**:
```json
{
    "0": "cardboard",
    "1": "glass",
    "2": "metal",
    "3": "paper",
    "4": "plastic",
    "5": "trash"
}
```

### Salida de Predicción

```python
[
  {
    "class": "plastico",
    "confidence": 0.8523,
    "percentage": 85.23
  },
  {
    "class": "metal",
    "confidence": 0.0892,
    "percentage": 8.92
  },
  ...
]
```

## API REST (Futuro)

Para desplegar como servicio web, se recomienda:

```python
from flask import Flask, request, jsonify
from src.detection import WasteDetector

app = Flask(__name__)
detector = WasteDetector('models/model.h5', 'models/classes.json')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    # Guardar temporalmente
    temp_path = 'temp.jpg'
    image.save(temp_path)
    
    # Predecir
    results = detector.predict(temp_path)
    
    # Limpiar
    os.remove(temp_path)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Testing

### Tests Unitarios (Recomendado)

```python
import unittest
from src.preprocessing import DataPreprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
    
    def test_image_loading(self):
        img = self.preprocessor.load_and_preprocess_image('test.jpg')
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertTrue(img.max() <= 1.0)
        self.assertTrue(img.min() >= 0.0)
```

### Validación de Modelos

Siempre validar en conjunto de test separado:
- Train/Test: 75% / 25%
- Validación interna adicional desde train para callbacks

## Mejoras Futuras

1. **Detección de objetos**: Localizar residuos en imágenes complejas
2. **Multi-label**: Clasificar múltiples tipos de residuos en una imagen
3. **Modelo en tiempo real**: Optimización para video en vivo
4. **Aplicación móvil**: Despliegue en Android/iOS
5. **Base de datos**: Almacenamiento de predicciones y métricas
6. **API REST**: Servicio web para integración
7. **Interfaz web**: Dashboard para usuarios

## Referencias

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Applications: https://keras.io/api/applications/
- Computer Vision Best Practices: https://github.com/microsoft/computervision-recipes

---

**Mantenedor**: Sistema de IA  
**Versión**: 1.0.0  
**Licencia**: MIT (o la que corresponda)
