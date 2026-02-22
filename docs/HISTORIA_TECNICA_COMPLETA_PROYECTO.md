# Historia técnica completa del proyecto (de principio a fin)

Este documento explica **cómo se construyó el proyecto completo**, en orden, desde sus bases hasta su estado actual. Incluye:

- el planteamiento inicial,
- el orden real de creación de archivos (según historial Git),
- la influencia de cada módulo y carpeta,
- el uso práctico de librerías,
- las decisiones de arquitectura, entrenamiento, optimización y evaluación,
- y cómo se conecta todo el sistema extremo a extremo.

---

## 1) Base conceptual: qué problema se decidió resolver

El proyecto se planteó para resolver un caso de visión por computadora: **clasificar residuos en imágenes** (6 clases) para facilitar separación y reciclaje.

### Objetivo funcional

Dado un archivo de imagen, el sistema debe devolver:

1. clase predicha (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`),
2. confianza de predicción,
3. posibilidad de mostrar top-k resultados y usar umbrales.

### Objetivo técnico

Construir una solución **modular, reusable y desplegable**:

- reusable en CLI, notebook y web,
- mantenible por separación de responsabilidades,
- entrenable y evaluable de forma reproducible,
- extensible para futuras integraciones (API/servicios).

### Principios tomados desde el inicio

1. **Separar lógica de negocio de interfaces**: la inferencia vive en `src/detection.py`; CLI (`predict.py`) y web (`app.py`) consumen esa clase.
2. **Entrenamiento reproducible**: `seed`, split controlado y artefactos persistentes.
3. **Artefactos explícitos**: modelo, historial, índices de clase y métricas quedan en `models/`.
4. **Automatización del flujo completo**: `run_pipeline.py` para split + entrenamiento + evaluación.

---

## 2) Orden real de construcción (cronología por commits)

> Fuente: historial Git del repositorio en orden cronológico (`git log --reverse --name-status`).

## Fase 0 — Arranque mínimo

### Commit inicial (2026-02-14)

- Se crea `README.md`.

**Influencia**: fija el contrato inicial del proyecto (objetivo, tecnologías, flujo esperado).

---

## Fase 1 — Estructura completa del sistema (2026-02-14)

En un commit grande se incorpora la primera versión funcional completa:

- núcleo de código en `src/`:
  - `src/data_collection.py`
  - `src/preprocessing.py`
  - `src/model.py`
  - `src/train.py`
  - `src/evaluation.py`
  - `src/detection.py`
- scripts ejecutables:
  - `train_model.py`
  - `predict.py`
  - `evaluate_model.py`
  - `setup.py`
- soporte y documentación:
  - `requirements.txt`
  - `EJEMPLOS.md`
  - `docs/DOCUMENTACION_TECNICA.md`
  - `docs/GUIA_USUARIO.md`
  - `notebooks/demo.ipynb`
  - `.gitignore`
- estructura de carpetas con `.gitkeep` para `data/` y `models/`.

**Por qué así**: primero se garantiza una base “vertical completa” (datos→modelo→inferencia), aunque luego se refina.

---

## Fase 2 — Ajustes técnicos y consolidación (2026-02-14)

Cambios relevantes:

- corrección de type hints en `src/detection.py`,
- adición de `PROJECT_SUMMARY.md`,
- mejoras en documentación y en varios módulos (`train_model.py`, `evaluate_model.py`, `src/*`) incluyendo robustez e inferencia defensiva.

**Influencia**: el sistema pasa de “funciona” a “más robusto y mejor documentado”.

---

## Fase 3 — Capa web y refinamiento de entrenamiento (2026-02-14)

Se agrega `app.py` (Streamlit) y se ajustan:

- `train_model.py`,
- `evaluate_model.py`,
- `requirements.txt`,
- `README.md`.

**Influencia**: el proyecto deja de ser sólo técnico/CLI y pasa a tener una interfaz usable localmente.

---

## Fase 4 — Pipeline integral y artefactos de ejecución (2026-02-15)

Se incorpora `run_pipeline.py` y se actualizan scripts para operar con split reproducible.

Se versionan artefactos:

- `models/class_indices.json`
- `models/training_history.csv`
- `models/split_report.json`
- `models/evaluation/classification_report.txt`
- `models/evaluation/metrics_summary.csv`

**Influencia**: el proyecto adopta un flujo productivo “un comando” y deja evidencia reproducible de resultados.

---

## Fase 5 — Fine-tuning y mejoras de usabilidad (2026-02-15)

Se agregan parámetros de fine-tuning y guía web paso a paso:

- `docs/GUIA_WEB_LOCAL_PASO_A_PASO.md`
- mejoras en `run_pipeline.py` y `train_model.py`.

**Influencia**: más control experimental y mejor onboarding para usuario final.

---

## Fase 6 — Ajuste por desbalance y reporte técnico (2026-02-15)

Se integra soporte de pesos de clase balanceados (`class_weight`) en entrenamiento, y se crea:

- `docs/INFORME_TECNICO_COMPLETO_ENTRENAMIENTO.md`.

**Influencia**: se ataca el desbalance (`trash` minoritaria) y se documenta la lectura técnica de resultados.

---

## 3) Estructura de carpetas: función e influencia

## `src/`

Contiene la **lógica reusable** del sistema.

- `data_collection.py`: organización/validación de dataset por clases.
- `preprocessing.py`: carga, normalización, augmentación y split train/test.
- `model.py`: arquitecturas CNN y transfer learning.
- `train.py`: entrenador modular con callbacks y opción adversarial FGSM.
- `evaluation.py`: métricas, matrices, reportes y análisis por clase.
- `detection.py`: inferencia centralizada para producción y UI.

**Influencia**: mantiene el dominio limpio; scripts sólo orquestan.

## `data/`

- `raw/`: fuente original (dataset por carpetas de clase).
- `processed/split/train|test/`: resultado del particionado reproducible.

**Influencia**: separa origen y derivados; evita contaminación de evaluación.

## `models/`

Guarda artefactos de entrenamiento y evaluación:

- modelo `.h5`,
- mapeo de clases,
- historial,
- reportes de evaluación.

**Influencia**: persistencia y trazabilidad de experimentos.

## `docs/`

Documentación de usuario, técnica, web y reportes.

**Influencia**: reduce dependencia del código para comprender/operar el sistema.

## `notebooks/`

Exploración interactiva (`demo.ipynb`).

**Influencia**: soporte didáctico y pruebas rápidas de hipótesis.

---

## 4) Tecnologías elegidas y qué se usa de cada una

Según `requirements.txt`:

- **TensorFlow/Keras**: construcción de red, entrenamiento, fine-tuning, guardado/carga del modelo.
- **NumPy**: manipulación de tensores/imágenes y operaciones numéricas.
- **OpenCV (`cv2`)**: lectura de imágenes, conversiones de color, resize y filtros de suavizado.
- **Pillow**: manejo de imágenes en UI Streamlit.
- **scikit-learn**: `train_test_split`, métricas y soporte de `class_weight`.
- **pandas**: persistencia/lectura de historial y métricas en CSV.
- **matplotlib + seaborn**: curvas de entrenamiento y matriz de confusión.
- **streamlit**: interfaz web local para carga de imagen e inferencia.
- **jupyter**: notebook de demostración.
- **tqdm**: utilidades de progreso (cuando aplica).

### Fundamento de elección

Se privilegió un stack estándar en ML aplicado: rápido de implementar, con buena documentación, y suficiente para pasar de prototipo a uso real local.

---

## 5) Módulo a módulo (orden lógico de construcción y dependencia)

Aunque varios archivos nacieron juntos, el **orden lógico de dependencia** para construir este sistema es:

1. datos,
2. preprocesamiento,
3. arquitectura,
4. entrenamiento,
5. evaluación,
6. inferencia,
7. interfaces/automatización.

## 5.1 `src/data_collection.py`

### Qué resuelve

- crear directorios por clase,
- validar cantidad de imágenes,
- resumir dataset.

### Por qué primero

Sin estructura de datos consistente no hay entrenamiento estable.

### Influencia

Define el “contrato de entrada” para `ImageDataGenerator` y split.

---

## 5.2 `src/preprocessing.py`

### Qué resuelve

- carga imagen,
- BGR→RGB,
- resize,
- normalización a `[0,1]`,
- generadores train/val con augmentación,
- split reproducible train/test.

### Por qué segundo

El modelo necesita tensores homogéneos y pipeline de entrada consistente.

### Influencia

Controla generalización (augmentación) y reduce overfitting.

---

## 5.3 `src/model.py`

### Qué resuelve

Define arquitecturas disponibles:

- `custom_cnn`,
- `mobilenet`,
- `resnet`,
- `efficientnet`.

### Arquitectura usada en entrenamiento principal (`train_model.py`)

Transfer Learning con `MobileNetV2`:

1. Base `MobileNetV2` sin top (`include_top=False`, pesos ImageNet).
2. `GlobalAveragePooling2D`.
3. `Dense(256, relu)`.
4. `Dropout(0.5)`.
5. `Dense(num_classes, softmax)`.

### Por qué esta decisión

- buena relación precisión/costo,
- convergencia más rápida que entrenar desde cero,
- ideal para dataset de tamaño moderado.

---

## 5.4 `src/train.py`

### Qué resuelve

Orquesta entrenamiento y persistencia con:

- `ModelCheckpoint`,
- `EarlyStopping`,
- `ReduceLROnPlateau`,
- `TensorBoard`,
- soporte opcional FGSM adversarial.

### Decisiones de optimización

- **optimizador**: Adam,
- **pérdida**: `categorical_crossentropy`,
- **métrica principal**: `accuracy`.

### Por qué

- Adam acelera convergencia inicial,
- `categorical_crossentropy` corresponde a clasificación multiclase softmax,
- callbacks reducen costo computacional y sobreentrenamiento.

### Influencia

Es el módulo donde se vuelven reales las decisiones de modelado.

---

## 5.5 `src/evaluation.py`

### Qué resuelve

- evaluación cuantitativa y diagnóstica:
  - clasificación por clase,
  - matriz de confusión,
  - accuracy por clase,
  - top confusiones.

### Por qué después de entrenar

Separa entrenamiento de validación final para evitar sesgo de interpretación.

### Influencia

Guía iteraciones: permite decidir si mejorar datos, arquitectura o hiperparámetros.

---

## 5.6 `src/detection.py`

### Qué resuelve

La inferencia reusable del proyecto:

- `predict` (ruta de imagen),
- `predict_array` (imagen en memoria),
- `batch_predict`,
- clasificación con umbral,
- preprocesamiento robusto y suavizado opcional.

### Por qué es clave

Evita duplicar lógica entre CLI y web; una sola implementación para producción local.

### Influencia

Es el punto de conexión entre modelo entrenado y experiencia de usuario.

---

## 6) Scripts principales: para qué sirven y cómo se conectan

## `setup.py`

Configura/verifica entorno, crea directorios base y guía primeros pasos.

## `train_model.py`

Entrenamiento principal actual:

- crea `ImageDataGenerator` con `validation_split=0.2`,
- entrena en 2 fases (cabeza + fine-tuning),
- permite `class_weight`,
- exporta:
  - `models/waste_classifier.h5`,
  - `models/training_history.csv`,
  - `models/class_indices.json`.

## `evaluate_model.py`

Consume modelo + test set y genera en `models/evaluation/`:

- `classification_report.txt`,
- `metrics_summary.csv`,
- `confusion_matrix.png`,
- `training_history.png` (si existe historial).

## `predict.py`

CLI para inferencia. Carga `WasteDetector`, permite top-k, umbral y visualización de salida.

## `app.py`

Interfaz Streamlit. Carga imagen, llama `predict_array` y muestra clase/confianza + barras de probabilidad.

## `run_pipeline.py`

Orquestador E2E:

1. valida dataset,
2. crea split reproducible (`train_test_split`),
3. ejecuta `train_model.py`,
4. ejecuta `evaluate_model.py`,
5. guarda `models/split_report.json`.

### Conexión entre scripts

`run_pipeline.py` -> `train_model.py` -> artefactos en `models/` -> `evaluate_model.py` y (`predict.py`/`app.py`) consumen esos artefactos.

---

## 7) Red neuronal usada: capas, pérdida y optimización

## Entrada

- tamaño: `224x224x3` (RGB),
- normalización: `x / 255`.

## Arquitectura efectiva

`MobileNetV2 base` -> `GlobalAveragePooling2D` -> `Dense(256, relu)` -> `Dropout(0.5)` -> `Dense(6, softmax)`.

## Función de pérdida

- `categorical_crossentropy`.

Razón: problema multiclase exclusivo (una etiqueta por imagen) y salida softmax.

## Optimizador

- Adam con dos tasas de aprendizaje:
  - fase 1 (cabeza): `1e-3`,
  - fase 2 (fine-tuning): `1e-5`.

Razón: aprendizaje rápido al inicio y ajuste fino estable luego.

## Estrategia de regularización

- `Dropout(0.5)` en cabeza,
- augmentación de datos,
- `EarlyStopping`,
- `ReduceLROnPlateau`.

Razón: combatir overfitting con dataset no masivo.

## Manejo de desbalance

- `class_weight` opcional en entrenamiento para aumentar el peso de clases minoritarias (ej. `trash`).

---

## 8) Manejo entre carpetas y flujo de artefactos

## Ruta de datos

1. Se coloca dataset en `data/raw/...` por clase.
2. `run_pipeline.py` crea split en `data/processed/split/train` y `data/processed/split/test`.
3. `train_model.py` consume `train`.
4. `evaluate_model.py` consume `test`.

## Ruta de artefactos

1. Entrenamiento guarda modelo/historial/clases en `models/`.
2. Evaluación guarda reportes y gráficas en `models/evaluation/`.
3. Inferencia (`predict.py`, `app.py`) lee `models/waste_classifier.h5` + `models/class_indices.json`.

### Beneficio

Separación clara entre:

- datos fuente,
- datos procesados,
- artefactos de experimentación,
- interfaces de consumo.

---

## 9) Fundamentos detrás de las decisiones tomadas

1. **Transfer Learning antes que CNN desde cero**: menos datos requeridos y convergencia más rápida.
2. **MobileNetV2 como base principal**: buena eficiencia para despliegue local.
3. **Dos fases de entrenamiento**: estabiliza aprendizaje y evita destruir representaciones preentrenadas al inicio.
4. **Callbacks automáticos**: reducen intervención manual y costo de entrenamiento.
5. **Separación `src/` vs scripts**: favorece reutilización y mantenimiento.
6. **Pipeline unificado (`run_pipeline.py`)**: reproducibilidad operativa.
7. **Documentación extensa**: facilita transferencia de conocimiento académico/técnico.
8. **Inferencia centralizada en `WasteDetector`**: una sola fuente de verdad para predicción.

---

## 10) Conexión completa de principio a fin (resumen operacional)

1. Preparar entorno (`setup.py`, `requirements.txt`).
2. Organizar/validar dataset (`src/data_collection.py`).
3. Preprocesar y particionar (`src/preprocessing.py`, `run_pipeline.py`).
4. Construir modelo (`src/model.py`).
5. Entrenar y ajustar (`train_model.py`, soporte de `src/train.py`).
6. Evaluar y diagnosticar (`evaluate_model.py`, `src/evaluation.py`).
7. Consumir modelo en inferencia (`src/detection.py`).
8. Exponer por CLI (`predict.py`) y web (`app.py`).
9. Analizar artefactos en `models/` y documentación en `docs/`.

---

## 11) Estado actual del proyecto

El sistema quedó en estado funcional con:

- entrenamiento reproducible,
- evaluación formal con métricas persistidas,
- inferencia reutilizable,
- interfaz web local,
- documentación técnica y de usuario.

En términos de ingeniería, la base está correctamente planteada para una siguiente etapa de mejora en precisión (más datos, ajuste de arquitectura y estrategias de pérdida para clases difíciles).
