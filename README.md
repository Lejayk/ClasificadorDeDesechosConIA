# Sistema de Detección y Clasificación de Residuos con IA

Proyecto de visión artificial para la Universidad Rafael Urdaneta orientado a la clasificación automática de residuos (por ejemplo: plástico, papel, vidrio y orgánicos) a partir de imágenes.

## Tecnologías

- Python
- TensorFlow / Keras
- OpenCV
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit

## Estructura esperada de datos

Organiza el dataset por carpetas de clase:

```text
data/
  raw/
    plastico/
    papel/
    vidrio/
    organico/
```

Para evaluación, usa una carpeta separada de prueba con la misma estructura:

```text
data/
  test/
    plastico/
    papel/
    vidrio/
    organico/
```

## Instalación

1. Crear y activar entorno virtual (recomendado).
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Entrenamiento del modelo

El script `train_model.py` implementa Transfer Learning con MobileNetV2 (ImageNet), `ImageDataGenerator` con augmentación (rotación, zoom y volteo horizontal), preprocesamiento a `224x224` con normalización y split entrenamiento/validación `80/20`.

Ejemplo:

```bash
python train_model.py --data-dir data/raw --epochs 20 --batch-size 32
```

Artefactos generados:

- `models/waste_classifier.h5` (modelo entrenado)
- `models/training_history.csv` (historial de métricas)
- `models/class_indices.json` (índice ↔ clase)

## Pipeline completo (recomendado)

Para cumplir todo el flujo solicitado (split reproducible train/test + entrenamiento + evaluación) en un solo comando:

```bash
python run_pipeline.py --raw-dir data/raw --epochs 20 --batch-size 32 --overwrite-split
```

Ejemplo recomendado con fine-tuning (2 fases):

```bash
python run_pipeline.py --raw-dir data/raw/dataset-resized --epochs 20 --fine-tune-epochs 10 --base-learning-rate 0.001 --fine-tune-learning-rate 0.00001 --unfreeze-layers 30 --batch-size 32 --overwrite-split
```

Este pipeline realiza:

- Validación del dataset por carpetas de clase.
- Split train/test reproducible en `data/processed/split/`.
- Entrenamiento con `train_model.py` sobre `data/processed/split/train`.
- Evaluación automática con `evaluate_model.py` sobre `data/processed/split/test`.
- Reporte de split en `models/split_report.json`.

## Evaluación del modelo

El script `evaluate_model.py` carga el modelo entrenado y el conjunto de prueba para calcular:

- Matriz de confusión
- Precision
- Recall
- F-score
- Reporte de clasificación por clase

Además, grafica la evolución de `Accuracy` y `Loss` desde el historial de entrenamiento.

Ejemplo:

```bash
python evaluate_model.py --test-dir data/test --model models/waste_classifier.h5
```

Salidas en `models/evaluation/`:

- `confusion_matrix.png`
- `classification_report.txt`
- `metrics_summary.csv`
- `training_history.png`

## Interfaz web (Streamlit)

`app.py` permite cargar una imagen y obtener la clase predicha con porcentaje de confianza.

La inferencia está centralizada en `src/detection.py` (`WasteDetector`), por lo que puedes reutilizar la misma lógica en:

- Streamlit (`app.py`)
- CLI (`predict.py`)
- API REST (FastAPI/Flask)
- Aplicación de escritorio (Tkinter/PyQt)

Ejecutar:

```bash
streamlit run app.py
```

## Guía rápida para levantar la web local (paso a paso)

1. Abre terminal en la carpeta del proyecto.
2. Verifica que el modelo exista en `models/waste_classifier.h5`.
3. Inicia la web local:

```bash
streamlit run app.py
```

4. Se abrirá una URL local (normalmente `http://localhost:8501`).
5. Pulsa "Selecciona una imagen" y carga una foto de residuo.
6. Verás la clase predicha y la confianza.

Si no abre sola, copia la URL local que aparece en terminal y pégala en tu navegador.

## Parámetros útiles

### `train_model.py`

- `--data-dir`: dataset de entrenamiento
- `--epochs`: número de épocas
- `--batch-size`: tamaño de lote
- `--model-output`: ruta del modelo `.h5`
- `--history-output`: ruta del historial `.csv`
- `--classes-output`: ruta del mapeo de clases `.json`

### `evaluate_model.py`

- `--test-dir`: dataset de prueba
- `--model`: modelo entrenado
- `--classes`: archivo de clases
- `--history`: historial de entrenamiento
- `--output-dir`: carpeta de resultados

## Notas

- Si cambias las clases del dataset, el sistema las detecta automáticamente desde las carpetas.
- La app usa `models/waste_classifier.h5` y `models/class_indices.json` por defecto.
