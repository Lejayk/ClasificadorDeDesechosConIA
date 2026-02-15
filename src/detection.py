"""
Módulo de inferencia para clasificación de residuos en tiempo real.
Sistema de detección para clasificar imágenes nuevas.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from tensorflow import keras


class WasteDetector:
    """
    Clase para detectar y clasificar residuos en imágenes.
    """
    
    def __init__(self, 
                 model_path: str,
                 class_mapping_path: str,
                 img_size: Tuple[int, int] = (224, 224),
                 enable_smoothing: bool = True,
                 smoothing_method: str = 'gaussian'):
        """
        Inicializa el detector.
        
        Args:
            model_path: Ruta al modelo guardado
            class_mapping_path: Ruta al archivo de mapeo de clases
            img_size: Tamaño de imagen esperado por el modelo
        """
        # Cargar modelo
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size
        self.enable_smoothing = enable_smoothing
        self.smoothing_method = smoothing_method
        
        # Cargar mapeo de clases
        with open(class_mapping_path, 'r') as f:
            self.index_to_class = json.load(f)
        
        # Convertir claves de string a int
        self.index_to_class = {int(k): v for k, v in self.index_to_class.items()}
        
        print(f"Detector inicializado con {len(self.index_to_class)} clases")
        print(f"Clases disponibles: {list(self.index_to_class.values())}")

    def preprocess_image_array(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen en memoria para inferencia.

        Args:
            image_rgb: Imagen RGB en formato numpy

        Returns:
            Tensor de entrada listo para el modelo
        """
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("La imagen en memoria es inválida o está vacía")

        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Se esperaba imagen RGB con 3 canales. shape recibido: {image_rgb.shape}")

        img = self._apply_smoothing(image_rgb)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0

        if not np.isfinite(img).all():
            raise ValueError("La imagen contiene valores inválidos tras preprocesamiento")

        return np.expand_dims(img, axis=0)

    def _validate_input_image(self, img: np.ndarray, image_path: str) -> None:
        """
        Valida la imagen antes de inferencia.

        Args:
            img: Imagen cargada por OpenCV
            image_path: Ruta de la imagen
        """
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        if img.size == 0:
            raise ValueError(f"La imagen está vacía: {image_path}")

        if len(img.shape) not in [2, 3]:
            raise ValueError(f"Formato de imagen no soportado (shape={img.shape}): {image_path}")

    def _sanitize_channels(self, img: np.ndarray) -> np.ndarray:
        """
        Asegura que la imagen tenga 3 canales RGB.
        """
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _apply_smoothing(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica suavizado ligero para mejorar robustez frente a perturbaciones.
        """
        if not self.enable_smoothing:
            return img

        if self.smoothing_method == 'median':
            return cv2.medianBlur(img, 3)

        return cv2.GaussianBlur(img, (3, 3), 0)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocesa una imagen para inferencia.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Imagen preprocesada
        """
        # Leer imagen
        img = cv2.imread(image_path)
        self._validate_input_image(img, image_path)

        # Sanitizar canales y convertir a RGB
        img = self._sanitize_channels(img)

        # Suavizado ligero para robustez
        img = self._apply_smoothing(img)
        
        # Redimensionar
        img = cv2.resize(img, self.img_size)
        
        # Normalizar
        img = img.astype(np.float32) / 255.0

        if not np.isfinite(img).all():
            raise ValueError(f"La imagen contiene valores inválidos tras preprocesamiento: {image_path}")
        
        # Agregar dimensión de batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predice la clase de residuo de una imagen.
        
        Args:
            image_path: Ruta a la imagen
            top_k: Número de predicciones principales a retornar
            
        Returns:
            Lista de diccionarios con predicciones
        """
        # Preprocesar imagen
        img = self.preprocess_image(image_path)
        
        # Realizar predicción
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Obtener top k predicciones
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': self.index_to_class[idx],
                'confidence': float(predictions[idx]),
                'percentage': float(predictions[idx] * 100)
            })
        
        return results

    def predict_array(self, image_rgb: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predice la clase a partir de una imagen RGB en memoria.

        Args:
            image_rgb: Imagen RGB como numpy array
            top_k: Número de predicciones principales

        Returns:
            Lista de predicciones ordenadas por confianza
        """
        img = self.preprocess_image_array(image_rgb)
        predictions = self.model.predict(img, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'class': self.index_to_class[idx],
                'confidence': float(predictions[idx]),
                'percentage': float(predictions[idx] * 100)
            })

        return results
    
    def predict_and_display(self, 
                           image_path: str,
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Predice y muestra resultado visualmente.
        
        Args:
            image_path: Ruta a la imagen
            save_path: Ruta para guardar imagen con predicción (opcional)
            
        Returns:
            Resultado de la predicción
        """
        import matplotlib.pyplot as plt
        
        # Realizar predicción
        results = self.predict(image_path, top_k=1)
        top_prediction = results[0]
        
        # Leer imagen original
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crear visualización
        plt.figure(figsize=(10, 6))
        
        # Mostrar imagen
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Imagen Original')
        plt.axis('off')
        
        # Mostrar predicción
        plt.subplot(1, 2, 2)
        plt.barh(range(len(results)), [r['percentage'] for r in results])
        plt.yticks(range(len(results)), [r['class'].capitalize() for r in results])
        plt.xlabel('Confianza (%)')
        plt.title('Predicciones')
        plt.xlim([0, 100])
        
        # Agregar valores
        for i, result in enumerate(results):
            plt.text(result['percentage'] + 2, i, f"{result['percentage']:.1f}%")
        
        plt.tight_layout()
        
        # Guardar o mostrar
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resultado guardado en: {save_path}")
        else:
            plt.show()
        
        return top_prediction
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Realiza predicciones en lote.
        
        Args:
            image_paths: Lista de rutas a imágenes
            
        Returns:
            Lista de predicciones
        """
        results = []
        
        print(f"Procesando {len(image_paths)} imágenes...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                prediction = self.predict(image_path, top_k=1)[0]
                results.append({
                    'image_path': image_path,
                    'prediction': prediction
                })
                
                if i % 10 == 0:
                    print(f"Procesadas {i}/{len(image_paths)} imágenes")
            
            except Exception as e:
                print(f"Error procesando {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        print("Procesamiento completado")
        return results
    
    def classify_with_threshold(self,
                               image_path: str,
                               confidence_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """
        Clasifica una imagen solo si la confianza supera un umbral.
        
        Args:
            image_path: Ruta a la imagen
            confidence_threshold: Umbral mínimo de confianza
            
        Returns:
            Predicción si supera el umbral, None si no
        """
        results = self.predict(image_path, top_k=1)
        top_prediction = results[0]
        
        if top_prediction['confidence'] >= confidence_threshold:
            return top_prediction
        else:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.index_to_class),
            'classes': list(self.index_to_class.values()),
            'total_parameters': self.model.count_params()
        }


if __name__ == "__main__":
    print("Módulo de detección cargado")
