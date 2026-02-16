# Informe Técnico Completo: Clasificación de Residuos con IA

## 1. Objetivo

Construir, entrenar, ajustar y evaluar un clasificador de residuos multiclase (6 clases) usando Transfer Learning con MobileNetV2, dejando el sistema listo para integración en interfaz web local (Streamlit).

---

## 2. Dataset y particionado

## 2.1 Fuente y estructura
- Dataset usado: `data/raw/dataset-resized` (TrashNet redimensionado).
- Clases:
  - cardboard
  - glass
  - metal
  - paper
  - plastic
  - trash

## 2.2 Split reproducible
- Script: `run_pipeline.py`
- Configuración:
  - train/test = 75/25
  - semilla (`random_seed`) = 42

## 2.3 Distribución de muestras
- Train total: 1892
  - cardboard: 302
  - glass: 375
  - metal: 307
  - paper: 445
  - plastic: 361
  - trash: 102
- Test total: 635
  - cardboard: 101
  - glass: 126
  - metal: 103
  - paper: 149
  - plastic: 121
  - trash: 35

Observación: clase `trash` fuertemente minoritaria (desbalance), lo que impacta precisión por clase.

---

## 3. Arquitectura del modelo

## 3.1 Tipo
- Transfer Learning con `MobileNetV2` + cabeza densa personalizada.

## 3.2 Entrada y salida
- Entrada: `(224, 224, 3)`
- Salida: `(6,)` con `softmax`

## 3.3 Capas (nivel alto)
1. `mobilenetv2_1.00_224` (base preentrenada)
2. `GlobalAveragePooling2D`
3. `Dense(256, relu)`
4. `Dropout(0.5)`
5. `Dense(6, softmax)`

## 3.4 Parámetros y pesos
- Total parámetros: **2,587,462**
- Trainable params (modelo final): **1,855,878**
- Non-trainable params (modelo final): **731,584**

Desglose de perceptrones (capas densas):
- `Dense(256)`:
  - pesos = `1280 x 256 = 327,680`
  - sesgos = `256`
  - total = **327,936**
- `Dense(6)`:
  - pesos = `256 x 6 = 1,536`
  - sesgos = `6`
  - total = **1,542**

## 3.5 Fine-tuning interno
- Capas totales en MobileNetV2: **154**
- Capas trainable tras ajuste fino: **31**
- Capas congeladas: **123**

---

## 4. Preprocesamiento y augmentación

- Reescalado: `1/255`
- Tamaño: `224x224`
- Augmentación aplicada:
  - `rotation_range=20`
  - `width_shift_range=0.15`
  - `height_shift_range=0.15`
  - `zoom_range=0.2`
  - `brightness_range=(0.85,1.15)`
  - `horizontal_flip=True`

---

## 5. Estrategia de entrenamiento

## 5.1 Entrenamiento en 2 fases

### Fase 1 (head training)
- Base congelada inicialmente.
- Épocas objetivo: 20
- Learning rate base: `1e-3`

### Fase 2 (fine-tuning)
- Se descongelan últimas 30 capas de la base.
- Épocas objetivo: 10 adicionales (20→30)
- Learning rate fine-tuning: `1e-5`

## 5.2 Control de optimización
- Optimizador: Adam
- Loss: `categorical_crossentropy`
- Métrica principal: `accuracy`
- Callbacks:
  - `EarlyStopping(patience=6, restore_best_weights=True)`
  - `ReduceLROnPlateau(factor=0.3, patience=3 aprox, min_lr=1e-7)`

## 5.3 Balanceo de clases
- Se activaron `class_weight` automáticos (balanceados):
  - cardboard: 1.0434
  - glass: 0.8417
  - metal: 1.0264
  - paper: 0.7093
  - plastic: 0.8737
  - trash: 3.0793

Interpretación: `trash` recibe peso más alto por su baja frecuencia.

## 5.4 Pasos/ciclos por época
Con `batch_size=32`:
- Train steps por época: `ceil(1515/32)=48`
- Validation steps por época: `ceil(377/32)=12`
- Test steps en evaluación: `ceil(635/32)=20`

---

## 6. Resultados y comparación

## 6.1 Corrida previa (sin class_weight)
- Accuracy test: **0.8157**
- Precision weighted: **0.8217**
- Recall weighted: **0.8157**
- F1 weighted: **0.8176**

## 6.2 Corrida mejorada (con class_weight + augmentación reforzada)
- Accuracy test: **0.8094**
- Precision weighted: **0.8208**
- Recall weighted: **0.8094**
- F1 weighted: **0.8126**

## 6.3 Conclusión técnica sobre precisión
- En este dataset y esta configuración, el ajuste de `class_weight` mejoró recall de clase minoritaria `trash`, pero **no subió la métrica global de accuracy** respecto al mejor punto previo.
- La mejor corrida global, hasta ahora, sigue siendo la previa con **accuracy 0.8157**.

---

## 7. Limitaciones observadas

1. Desbalance de clases, especialmente `trash`.
2. Similitud visual entre `plastic` y `trash`, y entre `paper`/`cardboard` en algunos casos.
3. Posible ruido por fondos, iluminación y ángulos heterogéneos.

---

## 8. Artefactos generados

- Modelo final: `models/waste_classifier.h5`
- Historial entrenamiento: `models/training_history.csv`
- Mapeo clases: `models/class_indices.json`
- Métricas resumidas: `models/evaluation/metrics_summary.csv`
- Reporte clasificación: `models/evaluation/classification_report.txt`
- Matriz de confusión: `models/evaluation/confusion_matrix.png`
- Curvas entrenamiento: `models/evaluation/training_history.png`

---

## 9. Recomendaciones para subir precisión en siguiente iteración

1. Mantener mejor checkpoint por `val_accuracy` y por `val_loss` en paralelo.
2. Usar un split estratificado K-Fold para estimación más robusta.
3. Probar `EfficientNetB0` o `MobileNetV3Small` con misma estrategia 2 fases.
4. Aplicar focal loss para mejorar clases difíciles (`trash`, `plastic`).
5. Incrementar datos de `trash` (objetivo: al menos 2x muestras actuales).

---

## 10. Estado final

- Pipeline completo funcionando (split + train + evaluate).
- Modelo entrenado y validado.
- Proyecto listo para interfaz web local con Streamlit (`app.py`).
- Lógica de inferencia desacoplada y reutilizable para futuras interfaces.
