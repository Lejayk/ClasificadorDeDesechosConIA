# 🗑️ Clasificador de Residuos con IA

Sistema inteligente de detección y clasificación de residuos utilizando técnicas de inteligencia artificial y visión artificial. El sistema es capaz de reconocer y clasificar diferentes tipos de residuos comunes a través de imágenes.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Descripción

Este proyecto implementa un sistema completo de clasificación de residuos que puede identificar automáticamente las siguientes categorías:

- 🔷 **Plástico**: Botellas, envases, bolsas
- 📄 **Papel**: Documentos, periódicos, cartulina
- 🔳 **Vidrio**: Botellas, frascos, cristales
- 🌱 **Orgánico**: Restos de comida, cáscaras
- ⚙️ **Metal**: Latas, envases metálicos
- 📦 **Cartón**: Cajas, empaques

## ✨ Características

- ✅ **Múltiples Arquitecturas**: CNN personalizada, MobileNetV2, ResNet50, EfficientNetB0
- ✅ **Transfer Learning**: Aprovecha modelos pre-entrenados para mayor precisión
- ✅ **Data Augmentation**: Mejora la generalización con técnicas de augmentación
- ✅ **Evaluación Completa**: Métricas detalladas y visualizaciones
- ✅ **Fácil de Usar**: Scripts CLI intuitivos para entrenamiento y predicción
- ✅ **Documentación Exhaustiva**: Guías de usuario y documentación técnica
- ✅ **Notebooks Interactivos**: Ejemplos en Jupyter para exploración

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Lejayk/ClasificadorDeDesechosConIA.git
cd ClasificadorDeDesechosConIA

# Instalar dependencias
pip install -r requirements.txt
```

### Preparar Datos

Organiza tus imágenes en la siguiente estructura:

```
data/raw/
├── plastico/
├── papel/
├── vidrio/
├── organico/
├── metal/
└── carton/
```

### Entrenar el Modelo

```bash
python train_model.py --data-dir data/raw --epochs 50
```

### Clasificar Imágenes

```bash
python predict.py --image ruta/a/imagen.jpg
```

## 📖 Documentación

- [Guía de Usuario](docs/GUIA_USUARIO.md) - Instrucciones detalladas de uso
- [Documentación Técnica](docs/DOCUMENTACION_TECNICA.md) - Arquitectura y detalles técnicos
- [Demo Notebook](notebooks/demo.ipynb) - Tutorial interactivo

## 🏗️ Estructura del Proyecto

```
ClasificadorDeDesechosConIA/
├── src/                    # Código fuente
│   ├── data_collection.py  # Recopilación de datos
│   ├── preprocessing.py    # Preprocesamiento
│   ├── model.py           # Arquitecturas de modelos
│   ├── train.py           # Entrenamiento
│   ├── evaluation.py      # Evaluación
│   └── detection.py       # Inferencia
├── data/                  # Datos del proyecto
├── models/                # Modelos entrenados
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentación
├── train_model.py         # Script de entrenamiento
├── predict.py            # Script de predicción
├── evaluate_model.py     # Script de evaluación
└── requirements.txt      # Dependencias

```

## 🔧 Requisitos

- Python 3.8 o superior
- TensorFlow 2.13+
- OpenCV
- NumPy, Pandas, Matplotlib
- 8 GB RAM mínimo (16 GB recomendado)
- GPU opcional (recomendada para entrenamiento)

## 📊 Resultados Esperados

Con un dataset bien balanceado de ~500 imágenes por clase, puedes esperar:

- **Accuracy**: 85-95%
- **Precision**: 80-90% por clase
- **Recall**: 80-90% por clase

## 🎯 Casos de Uso

1. **Gestión de Residuos**: Automatización en plantas de reciclaje
2. **Educación Ambiental**: Herramienta para enseñar reciclaje
3. **Smart Bins**: Contenedores inteligentes que clasifican automáticamente
4. **Aplicaciones Móviles**: Apps para ciudadanos sobre clasificación de residuos
5. **Auditorías**: Verificación de correcta separación de residuos

## 🛠️ Uso Avanzado

### Entrenar con Transfer Learning

```bash
python train_model.py \
    --data-dir data/raw \
    --architecture mobilenet \
    --epochs 30 \
    --learning-rate 0.0001
```

### Evaluar el Modelo

```bash
python evaluate_model.py \
    --test-dir data/test \
    --model models/waste_classifier_custom_cnn_best.h5
```

### Predicción con Visualización

```bash
python predict.py \
    --image test.jpg \
    --output resultado.png \
    --top-k 3
```

## 📝 Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `train_model.py` | Entrena el modelo de clasificación |
| `predict.py` | Clasifica imágenes nuevas |
| `evaluate_model.py` | Evalúa rendimiento del modelo |

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👥 Autores

- **Lejayk** - *Desarrollo inicial*

## 🙏 Agradecimientos

- TensorFlow y Keras por las herramientas de deep learning
- La comunidad de código abierto por las librerías utilizadas
- Datasets públicos de residuos para entrenamiento

## 📧 Contacto

Para preguntas, sugerencias o reportar problemas:
- Crear un issue en GitHub
- Revisar la documentación en `docs/`

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**
