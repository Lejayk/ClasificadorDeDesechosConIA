<div align="center">

# 🌍♻️ Sistema de Detección y Clasificación de Residuos con IA

### *Clasificación Inteligente de Residuos mediante Visión Artificial*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

Proyecto de visión artificial para la **Universidad Rafael Urdaneta** orientado a la clasificación automática de residuos (plástico, papel, vidrio, orgánicos y más) a partir de imágenes.

[Características](#-características) • [Instalación](#-instalación) • [Uso Rápido](#-uso-rápido) • [Documentación](#-documentación) • [Ejemplos](#-ejemplos)

---

</div>

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Instalación](#-instalación)
- [Estructura de Datos](#-estructura-de-datos)
- [Uso Rápido](#-uso-rápido)
- [Entrenamiento](#-entrenamiento-del-modelo)
- [Pipeline Completo](#-pipeline-completo)
- [Evaluación](#-evaluación-del-modelo)
- [Interfaz Web](#-interfaz-web)
- [Ejemplos de Uso](#-ejemplos-de-uso)
- [Documentación Adicional](#-documentación-adicional)
- [Contribuir](#-contribuir)

---

## ✨ Características

<table>
<tr>
<td width="50%">

### 🎯 Funcionalidades Principales

- ✅ **Múltiples Arquitecturas CNN**
  - Custom CNN (ligero, ~5M parámetros)
  - MobileNetV2 (optimizado para móviles)
  - ResNet50 (alta precisión)
  - EfficientNetB0 (balanceado)

- ✅ **Transfer Learning Avanzado**
  - Pre-entrenamiento con ImageNet
  - Fine-tuning configurable
  - Descongelamiento de capas selectivo

</td>
<td width="50%">

### 🚀 Capacidades

- ✅ **Data Augmentation Inteligente**
  - Rotación, zoom, flip
  - Normalización automática
  - Split train/validation/test

- ✅ **Interfaz Completa**
  - CLI para entrenamiento y predicción
  - Interfaz web con Streamlit
  - API lista para integración

</td>
</tr>
</table>

### 🏷️ Categorías de Residuos Soportadas

| Categoría | Emoji | Descripción |
|-----------|-------|-------------|
| **Plástico** | 🔷 | Botellas, envases, bolsas |
| **Papel** | 📄 | Hojas, revistas, periódicos |
| **Vidrio** | 🔳 | Botellas, frascos |
| **Orgánico** | 🌱 | Restos de comida, plantas |
| **Metal** | ⚙️ | Latas, aluminio |
| **Cartón** | 📦 | Cajas, empaques |

---

## 🛠️ Tecnologías

<div align="center">

| Tecnología | Uso | Versión |
|------------|-----|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Lenguaje principal | 3.8+ |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep Learning | 2.13+ |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | API de alto nivel | Incluido |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | Procesamiento de imágenes | 4.8+ |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Computación numérica | 1.24+ |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Análisis de datos | 2.0+ |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white) | Visualización | 3.7+ |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine Learning | 1.3+ |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Interfaz web | 1.30+ |

</div>

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA DE CLASIFICACIÓN                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Entrada de  │   │ Preprocesa-  │   │   Modelo     │
│   Imagen     │──▶│    miento    │──▶│     CNN      │
│              │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │ Clasificación│
                                     │  + Confianza │
                                     └──────────────┘
```

### Componentes Principales

```
src/
├── 📊 data_collection.py    # Organización y validación del dataset
├── 🔄 preprocessing.py      # Preprocesamiento y data augmentation
├── 🧠 model.py              # Arquitecturas CNN y transfer learning
├── 🎓 train.py              # Pipeline de entrenamiento
├── 📈 evaluation.py         # Evaluación y métricas
└── 🎯 detection.py          # Sistema de inferencia
```

---

## 📥 Instalación

### Requisitos Previos

> **⚠️ Importante:** Se recomienda Python 3.8+ y al menos 8GB de RAM (16GB recomendado para entrenamiento).

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/Lejayk/ClasificadorDeDesechosConIA.git
cd ClasificadorDeDesechosConIA
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar Instalación

```bash
python setup.py
```

---

## 📁 Estructura de Datos

Organiza el dataset por carpetas de clase:

```text
data/
  raw/
    plastico/
    papel/
    vidrio/
    organico/
```


Organiza tu dataset por carpetas de clase. El sistema detectará automáticamente las categorías:

```
📂 data/
├── 📂 raw/                    # Datos originales para entrenamiento
│   ├── 📂 plastico/          # Imágenes de plástico
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── 📂 papel/             # Imágenes de papel
│   ├── 📂 vidrio/            # Imágenes de vidrio
│   ├── 📂 organico/          # Imágenes de orgánicos
│   ├── 📂 metal/             # Imágenes de metal (opcional)
│   └── 📂 carton/            # Imágenes de cartón (opcional)
│
└── 📂 processed/             # Datos procesados (generado automáticamente)
    └── 📂 split/
        ├── 📂 train/         # 80% entrenamiento
        └── 📂 test/          # 20% prueba
```

> **💡 Consejo:** Se recomienda un mínimo de 100-200 imágenes por categoría (idealmente 500+) para mejores resultados.

---

## 🚀 Uso Rápido

### Opción 1: Pipeline Completo (Recomendado)

Ejecuta todo el proceso en un solo comando:

```bash
python run_pipeline.py \
    --raw-dir data/raw \
    --epochs 20 \
    --batch-size 32 \
    --overwrite-split
```

Este comando ejecutará:
1. ✅ Validación del dataset
2. ✅ División train/test (80/20)
3. ✅ Entrenamiento del modelo
4. ✅ Evaluación automática
5. ✅ Generación de reportes

### Opción 2: Paso a Paso

#### 1️⃣ Entrenar el Modelo

```bash
python train_model.py \
    --data-dir data/raw \
    --epochs 20 \
    --batch-size 32
```

#### 2️⃣ Evaluar el Modelo

```bash
python evaluate_model.py \
    --test-dir data/test \
    --model models/waste_classifier.h5
```

#### 3️⃣ Clasificar una Imagen

```bash
python predict.py --image path/to/image.jpg
```

---

## 🎓 Entrenamiento del Modelo

### Entrenamiento Básico

El script `train_model.py` implementa **Transfer Learning** con MobileNetV2 pre-entrenado en ImageNet:

```bash
python train_model.py --data-dir data/raw --epochs 20 --batch-size 32
```

### Entrenamiento Avanzado con Fine-Tuning

Para mejores resultados, usa entrenamiento en dos fases:

```bash
python train_model.py \
    --data-dir data/raw \
    --architecture mobilenet \
    --epochs 30 \
    --fine-tune-epochs 10 \
    --learning-rate 0.001 \
    --fine-tune-learning-rate 0.00001 \
    --batch-size 32
```

### Características del Entrenamiento

| Característica | Descripción |
|----------------|-------------|
| **Data Augmentation** | Rotación, zoom, flip horizontal, shifts |
| **Preprocesamiento** | Resize a 224x224, normalización |
| **Split** | 80% entrenamiento, 20% validación |
| **Callbacks** | Early stopping, reducción de LR, checkpoints |
| **Transfer Learning** | ImageNet pre-entrenado |

### Artefactos Generados

Después del entrenamiento, se generan:

- `models/waste_classifier.h5` - Modelo entrenado
- `models/training_history.csv` - Historial de métricas
- `models/class_indices.json` - Mapeo de índices a clases

---

## 🔄 Pipeline Completo

### Ejecución Básica

```bash
python run_pipeline.py \
    --raw-dir data/raw \
    --epochs 20 \
    --batch-size 32 \
    --overwrite-split
```

### Ejecución con Fine-Tuning (Recomendado)

```bash
python run_pipeline.py \
    --raw-dir data/raw/dataset-resized \
    --epochs 20 \
    --fine-tune-epochs 10 \
    --base-learning-rate 0.001 \
    --fine-tune-learning-rate 0.00001 \
    --unfreeze-layers 30 \
    --batch-size 32 \
    --overwrite-split
```

### Proceso del Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  1. Validación del Dataset por Carpetas de Clase       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  2. Split Train/Test Reproducible (80/20)              │
│     └─▶ data/processed/split/                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  3. Entrenamiento con train_model.py                   │
│     └─▶ sobre data/processed/split/train               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  4. Evaluación Automática con evaluate_model.py        │
│     └─▶ sobre data/processed/split/test                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  5. Reporte de Split en models/split_report.json       │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluación del Modelo


El script `evaluate_model.py` realiza una evaluación completa del modelo:

```bash
python evaluate_model.py \
    --test-dir data/test \
    --model models/waste_classifier.h5
```

### Métricas Calculadas

<table>
<tr>
<td width="50%">

**📈 Métricas Globales**
- Accuracy total
- Pérdida (Loss)
- Precision macro/micro
- Recall macro/micro
- F1-Score

</td>
<td width="50%">

**📊 Métricas por Clase**
- Precision por categoría
- Recall por categoría
- F1-Score por categoría
- Support (muestras)

</td>
</tr>
</table>

### Visualizaciones Generadas

Todas las visualizaciones se guardan en `models/evaluation/`:

| Archivo | Descripción |
|---------|-------------|
| `confusion_matrix.png` | Matriz de confusión normalizada |
| `classification_report.txt` | Reporte detallado por clase |
| `metrics_summary.csv` | Resumen de métricas en CSV |
| `training_history.png` | Gráficas de Accuracy y Loss |

### Ejemplo de Salida

```
Evaluación del Modelo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy Global: 94.32%
Loss: 0.1845

Métricas por Clase:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plástico   - Precision: 96.1%  Recall: 94.8%
Papel      - Precision: 93.5%  Recall: 95.2%
Vidrio     - Precision: 92.8%  Recall: 91.5%
Orgánico   - Precision: 95.2%  Recall: 96.0%
```

---

## 🌐 Interfaz Web

### Lanzar la Aplicación Streamlit

La aplicación web permite clasificar imágenes de forma interactiva:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Características de la Interfaz

- 📤 **Carga de imágenes** (JPG, JPEG, PNG)
- 🎯 **Predicción en tiempo real**
- 📊 **Visualización de confianza** (top-3 predicciones)
- 📈 **Gráfico de barras** con distribución de probabilidades
- 🖼️ **Vista previa** de la imagen cargada

### Guía Rápida de Uso

1. **Inicia la aplicación**
   ```bash
   streamlit run app.py
   ```

2. **Verifica que existe el modelo**
   - Debe existir: `models/waste_classifier.h5`
   - Debe existir: `models/class_indices.json`

3. **Carga una imagen**
   - Clic en "Selecciona una imagen"
   - Elige una foto de residuo

4. **Obtén la predicción**
   - La clase predicha aparecerá con su porcentaje de confianza
   - Verás un gráfico con las top-3 predicciones

> **💡 Tip:** Si la aplicación no se abre automáticamente, copia la URL que aparece en la terminal y pégala en tu navegador.

---

## 💻 Ejemplos de Uso

### Ejemplo 1: Clasificación Simple

```bash
python predict.py --image test_images/botella_plastico.jpg
```

**Salida:**
```
🎯 Predicción: PLÁSTICO
📊 Confianza: 98.45%
```

### Ejemplo 2: Top-K Predicciones

```bash
python predict.py --image test_images/lata.jpg --top-k 3
```

**Salida:**
```
Top 3 Predicciones:
1. Metal    - 95.2%
2. Plástico - 3.1%
3. Vidrio   - 1.2%
```

### Ejemplo 3: Con Visualización

```bash
python predict.py \
    --image test_images/botella.jpg \
    --output resultado.png \
    --top-k 3
```

Genera una imagen con la predicción visualizada.

### Ejemplo 4: Uso Programático en Python

```python
from src.detection import WasteDetector

# Inicializar detector
detector = WasteDetector(
    model_path='models/waste_classifier.h5',
    class_mapping_path='models/class_indices.json'
)

# Clasificar imagen
results = detector.predict('imagen.jpg', top_k=3)

# Mostrar resultados
for i, result in enumerate(results, 1):
    print(f"{i}. {result['class']}: {result['percentage']:.2f}%")
```

### Ejemplo 5: Predicción en Lote

```python
from src.detection import WasteDetector

detector = WasteDetector(
    model_path='models/waste_classifier.h5',
    class_mapping_path='models/class_indices.json'
)

# Lista de imágenes
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']

# Procesar en lote
results = detector.batch_predict(images)

for result in results:
    if 'error' not in result:
        pred = result['prediction']
        print(f"{result['image_path']}: {pred['class']} ({pred['percentage']:.1f}%)")
```

---

## 📚 Documentación Adicional

Para información más detallada, consulta:

| Documento | Descripción |
|-----------|-------------|
| 📖 [EJEMPLOS.md](EJEMPLOS.md) | Ejemplos detallados de uso y código |
| 🔧 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Resumen técnico del proyecto |
| 📓 [notebooks/](notebooks/) | Jupyter notebooks interactivos |
| 📁 [docs/](docs/) | Documentación técnica adicional |

---

## 🔧 Parámetros de Configuración

### `train_model.py`

```bash
python train_model.py [OPTIONS]
```

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--data-dir` | Directorio del dataset | `data/raw` |
| `--epochs` | Número de épocas | `50` |
| `--batch-size` | Tamaño del lote | `32` |
| `--architecture` | Arquitectura CNN | `mobilenet` |
| `--learning-rate` | Tasa de aprendizaje | `0.001` |
| `--model-output` | Ruta del modelo `.h5` | `models/waste_classifier.h5` |
| `--history-output` | Historial `.csv` | `models/training_history.csv` |
| `--classes-output` | Mapeo de clases `.json` | `models/class_indices.json` |

### `evaluate_model.py`

```bash
python evaluate_model.py [OPTIONS]
```

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--test-dir` | Dataset de prueba | `data/test` |
| `--model` | Modelo entrenado | `models/waste_classifier.h5` |
| `--classes` | Archivo de clases | `models/class_indices.json` |
| `--history` | Historial de entrenamiento | `models/training_history.csv` |
| `--output-dir` | Carpeta de resultados | `models/evaluation/` |

### `predict.py`

```bash
python predict.py [OPTIONS]
```

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--image` | Ruta de la imagen | *Requerido* |
| `--model` | Modelo a usar | `models/waste_classifier.h5` |
| `--classes` | Archivo de clases | `models/class_indices.json` |
| `--top-k` | Top K predicciones | `1` |
| `--threshold` | Umbral de confianza | `0.0` |
| `--output` | Guardar resultado visualizado | `None` |

---

## 💡 Consejos y Mejores Prácticas

<table>
<tr>
<td width="50%">

### 📊 Datos

- ✅ Mantén datasets **balanceados** (similar cantidad por clase)
- ✅ Mínimo **100-200 imágenes** por categoría
- ✅ Ideal **500+ imágenes** por categoría
- ✅ Usa **imágenes variadas** (diferentes ángulos, iluminación)
- ✅ Separa datos de **test** (nunca usados en entrenamiento)

</td>
<td width="50%">

### 🎓 Entrenamiento

- ✅ Usa **data augmentation** para más variabilidad
- ✅ Comienza con **learning rate de 0.001**
- ✅ Ajusta **batch size** según tu RAM/GPU (16, 32, 64)
- ✅ Empieza con **30-50 épocas**
- ✅ El **early stopping** parará si no hay mejora

</td>
</tr>
</table>

---

## ⚠️ Solución de Problemas

### Error: No module named 'tensorflow'

```bash
pip install tensorflow>=2.13.0
```

### Error: Out of memory

```bash
# Reducir batch size
python train_model.py --batch-size 16
```

### Bajo Accuracy

- 📊 Recopilar más datos
- 🔄 Entrenar por más épocas
- 🧠 Probar transfer learning (mobilenet, resnet)
- ✅ Verificar que los datos estén bien etiquetados
- 🔍 Revisar balance de clases

### Modelo Muy Lento

- 📱 Usar **MobileNetV2** para inferencia rápida
- 🔽 Reducir tamaño de imagen a **128x128**
- ⚡ Optimizar modelo con **TensorFlow Lite**
- 🖥️ Usar GPU para entrenamiento

---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Si deseas contribuir:

1. 🍴 Fork el repositorio
2. 🌿 Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push a la rama (`git push origin feature/AmazingFeature`)
5. 🔄 Abre un Pull Request

---

## 📝 Notas Importantes

> **ℹ️ Detección Automática de Clases**
> 
> Si cambias las clases del dataset, el sistema las detecta automáticamente desde las carpetas. No necesitas modificar el código.

> **🔧 Configuración por Defecto**
> 
> La aplicación web usa `models/waste_classifier.h5` y `models/class_indices.json` por defecto. Asegúrate de que estos archivos existan antes de ejecutar la interfaz.

> **🎯 Reutilización del Código**
> 
> La inferencia está centralizada en `src/detection.py` (`WasteDetector`), por lo que puedes reutilizar la misma lógica en:
> - Streamlit (`app.py`)
> - CLI (`predict.py`)
> - API REST (FastAPI/Flask)
> - Aplicación de escritorio (Tkinter/PyQt)

---

## 📜 Licencia

Este proyecto está desarrollado para la **Universidad Rafael Urdaneta** como proyecto académico.

---

## 🙏 Agradecimientos

- Universidad Rafael Urdaneta
- TensorFlow y Keras por las herramientas de Deep Learning
- La comunidad de código abierto

---

<div align="center">

### ⭐ Si este proyecto te fue útil, considera darle una estrella ⭐

**Desarrollado con ❤️ para la clasificación inteligente de residuos**

[🔝 Volver arriba](#-sistema-de-detección-y-clasificación-de-residuos-con-ia)

</div>
