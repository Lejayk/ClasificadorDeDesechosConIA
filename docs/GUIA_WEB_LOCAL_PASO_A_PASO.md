# Guía paso a paso: interfaz web local (sin experiencia previa)

Esta guía está pensada para que puedas probar tu clasificador en una web local sin saber desarrollo web.

## 1) Requisitos mínimos

- Tener el proyecto descargado.
- Tener Python instalado.
- Tener el entorno del proyecto configurado (`.venv`).

## 2) Abrir el proyecto

1. Abre VS Code.
2. Abre la carpeta del proyecto `ClasificadorDeDesechosConIA`.
3. Abre una terminal integrada (Terminal > New Terminal).

## 3) Instalar dependencias (si no están)

Ejecuta:

```bash
pip install -r requirements.txt
```

## 4) Entrenar modelo (si todavía no lo tienes entrenado)

Si ya tienes `models/waste_classifier.h5`, salta al paso 5.

Comando recomendado (entrenamiento 2 fases con fine-tuning):

```bash
python run_pipeline.py --raw-dir data/raw/dataset-resized --epochs 20 --fine-tune-epochs 10 --base-learning-rate 0.001 --fine-tune-learning-rate 0.00001 --unfreeze-layers 30 --batch-size 32 --overwrite-split
```

Esto genera:
- Modelo: `models/waste_classifier.h5`
- Clases: `models/class_indices.json`
- Métricas y gráficas: `models/evaluation/`

## 5) Levantar la web local

Ejecuta:

```bash
streamlit run app.py
```

Verás en terminal una URL local, normalmente:

- `http://localhost:8501`

Abre esa URL en tu navegador.

## 6) Probar clasificación

1. En la página, pulsa **Selecciona una imagen**.
2. Sube una imagen (`.jpg`, `.jpeg`, `.png`).
3. La app mostrará:
   - clase predicha,
   - confianza,
   - barras con probabilidades.

## 7) Errores comunes y solución rápida

### A) "No se encontró el modelo"
Asegúrate de que existan:
- `models/waste_classifier.h5`
- `models/class_indices.json`

Si faltan, vuelve a ejecutar el pipeline de entrenamiento.

### B) Puerto ocupado
Si `8501` está ocupado:

```bash
streamlit run app.py --server.port 8502
```

### C) No abre el navegador automáticamente
Copia la URL local mostrada en terminal y pégala manualmente en tu navegador.

## 8) Próximo paso recomendado

Cuando esta web local te funcione bien, el siguiente paso natural es crear una API (FastAPI/Flask) para conectar la misma inferencia a una app web más personalizada o móvil.
