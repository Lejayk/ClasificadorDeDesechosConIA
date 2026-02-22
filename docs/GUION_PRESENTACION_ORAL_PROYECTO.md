# Guion de presentación oral: Clasificador de Desechos con IA

Este documento está diseñado para exponer el proyecto en voz alta, de forma clara, cronológica y técnica, pero fácil de seguir.

---

## 1. Apertura (qué problema resolvimos)

Buenas, este proyecto se llama **Clasificador de Desechos con IA** y nace para resolver un problema concreto: **identificar automáticamente el tipo de residuo a partir de una imagen**.

La idea central fue construir un sistema que pudiera reconocer seis clases:

- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

El objetivo no era solo entrenar un modelo, sino dejar un flujo completo: desde datos, entrenamiento y evaluación, hasta una interfaz web usable localmente.

---

## 2. Base estratégica (cómo planteamos el proyecto)

Antes de programar, definimos tres bases:

1. **Modularidad**: separar el núcleo técnico de las interfaces.
2. **Reproducibilidad**: poder repetir el entrenamiento y la evaluación con resultados trazables.
3. **Reutilización**: usar la misma inferencia en CLI, notebook y web.

Por eso se tomó la decisión de tener un directorio `src/` con la lógica principal, y scripts aparte para ejecutar tareas específicas.

---

## 3. Inicio real del desarrollo (orden cronológico)

### Etapa 1: arranque del repositorio

Lo primero fue crear `README.md`, para dejar claro el alcance, tecnologías y forma de uso.

### Etapa 2: primera versión completa

Luego se creó la columna vertebral del proyecto en una sola etapa grande:

- módulos en `src/` (`data_collection`, `preprocessing`, `model`, `train`, `evaluation`, `detection`),
- scripts (`train_model.py`, `predict.py`, `evaluate_model.py`, `setup.py`),
- dependencias (`requirements.txt`),
- documentación y notebook.

Aquí el proyecto ya podía entrenar, evaluar y predecir.

### Etapa 3: mejoras de robustez y documentación

Se ajustaron módulos, tipos, y se añadió documentación técnica y resumen de proyecto para hacer el sistema más mantenible.

### Etapa 4: interfaz web

Se agregó `app.py` con Streamlit para que el sistema no dependiera solo de terminal.

### Etapa 5: pipeline integral

Se creó `run_pipeline.py` para ejecutar todo el proceso en cadena:

1. validación de dataset,
2. split train/test,
3. entrenamiento,
4. evaluación,
5. generación de reportes.

### Etapa 6: optimización por desbalance

Se añadió soporte para pesos de clase (`class_weight`) porque `trash` tenía menos ejemplos, y esto afectaba el rendimiento por clase.

---

## 4. Estructura de carpetas (qué hace cada una)

## `data/`

- `raw/`: dataset original organizado por clase.
- `processed/split/train` y `processed/split/test`: datos ya divididos para entrenar y probar.

## `src/`

Es el corazón técnico del sistema:

- `data_collection.py`: valida y organiza el dataset.
- `preprocessing.py`: preprocesa imágenes y crea generadores.
- `model.py`: define arquitecturas de red.
- `train.py`: entrena con callbacks y opciones de robustez.
- `evaluation.py`: calcula métricas y gráficas.
- `detection.py`: inferencia reutilizable.

## `models/`

Guarda artefactos de salida:

- modelo entrenado,
- historial,
- índices de clase,
- reportes de evaluación.

## `docs/`

Documentación de usuario, técnica y resultados.

## `notebooks/`

Demostración interactiva para exploración.

---

## 5. Tecnologías y librerías (qué usamos y para qué)

- **TensorFlow / Keras**: construir, entrenar y guardar la red neuronal.
- **OpenCV**: leer imágenes, cambiar color, redimensionar y suavizar.
- **NumPy**: manipulación de matrices y tensores.
- **scikit-learn**: split train/test, métricas y pesos de clase.
- **Pandas**: manejo de historiales y métricas en CSV.
- **Matplotlib / Seaborn**: visualización de curvas y matriz de confusión.
- **Streamlit**: interfaz web local para subir imagen y clasificar.
- **Pillow**: compatibilidad de imágenes en la app web.

---

## 6. Modelo de IA (arquitectura, pérdida y optimización)

El enfoque principal fue **Transfer Learning con MobileNetV2**.

Arquitectura usada:

1. Base `MobileNetV2` preentrenada en ImageNet (sin capa final).
2. `GlobalAveragePooling2D` para compactar características.
3. `Dense(256, relu)` para aprendizaje específico del dominio.
4. `Dropout(0.5)` para reducir sobreajuste.
5. `Dense(6, softmax)` para clasificación multiclase.

Configuración clave:

- **Pérdida**: `categorical_crossentropy` (correcta para multiclase con softmax).
- **Optimizador**: Adam.
- **Métrica principal**: accuracy.
- **Entrenamiento en dos fases**:
  - fase 1: entrenar cabeza del modelo,
  - fase 2: fine-tuning de capas finales con learning rate menor.

---

## 7. Preprocesamiento y estrategia de generalización

Las imágenes se normalizan a rango `[0,1]` y se redimensionan a `224x224`.

Se aplicó **data augmentation** para mejorar generalización:

- rotación,
- desplazamientos,
- zoom,
- variación de brillo,
- flip horizontal.

Además, se usan callbacks:

- `EarlyStopping` para detener si no mejora,
- `ReduceLROnPlateau` para bajar el learning rate cuando se estanca,
- guardado de mejor modelo.

Esto reduce sobreentrenamiento y hace más eficiente el entrenamiento.

---

## 8. Cómo se conecta todo (flujo extremo a extremo)

El recorrido completo es:

1. Cargar dataset en `data/raw` por clase.
2. Ejecutar `run_pipeline.py`.
3. Se genera split reproducible train/test.
4. `train_model.py` entrena y guarda modelo + clases + historial.
5. `evaluate_model.py` calcula métricas y crea reportes/gráficas.
6. `predict.py` o `app.py` usan `src/detection.py` para inferir sobre imágenes nuevas.

Punto clave: la inferencia está centralizada en `WasteDetector`, por eso la misma lógica funciona en consola y en web.

---

## 9. Decisiones técnicas y su fundamento

1. **Transfer learning** en lugar de entrenar desde cero: más eficiente y estable con dataset moderado.
2. **MobileNetV2**: buen equilibrio entre precisión y costo computacional.
3. **Pipeline automatizado**: evita errores manuales y mejora reproducibilidad.
4. **Separación de carpetas** (`raw`, `processed`, `models`): trazabilidad de datos y resultados.
5. **Class weights**: compensar desbalance de clases, especialmente `trash`.
6. **Documentación amplia**: facilitar uso, mantenimiento y defensa académica del proyecto.

---

## 10. Cierre para exposición

En resumen, este proyecto no se quedó en “un modelo que predice”: se construyó como un sistema completo de ingeniería de ML.

Incluye:

- diseño modular,
- entrenamiento reproducible,
- evaluación formal,
- reportes persistentes,
- y despliegue local con interfaz web.

El siguiente paso natural para escalar precisión sería ampliar datos en clases minoritarias, comparar más arquitecturas y ajustar funciones de pérdida para clases difíciles.

---

## 11. Mini guion de 2 minutos (opcional)

Si necesitas una versión muy corta para presentar rápido:

1. “El proyecto clasifica residuos en 6 clases usando visión por computadora.”
2. “Se diseñó de forma modular: `src` para lógica, scripts para ejecución y `models` para artefactos.”
3. “Usamos MobileNetV2 con transfer learning, `categorical_crossentropy`, Adam y entrenamiento en dos fases.”
4. “El pipeline automatizado hace split, entrenamiento y evaluación en cadena, guardando reportes reproducibles.”
5. “La inferencia se centraliza en `WasteDetector`, por eso funciona igual en CLI y en Streamlit.”
6. “Resultado: sistema completo, documentado y listo para mejoras iterativas de precisión.”
