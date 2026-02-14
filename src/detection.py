"""
Módulo de inferencia para clasificación de residuos en tiempo real.
Sistema de detección para clasificar imágenes nuevas.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras


class WasteDetector:
    """
    Clase para detectar y clasificar residuos en imágenes.
    """
    
    def __init__(self, 
                 model_path: str,
                 class_mapping_path: str,
                 img_size: Tuple[int, int] = (224, 224)):
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
        
        # Cargar mapeo de clases
        with open(class_mapping_path, 'r') as f:
            self.index_to_class = json.load(f)
        
        # Convertir claves de string a int
        self.index_to_class = {int(k): v for k, v in self.index_to_class.items()}
        
        print(f"Detector inicializado con {len(self.index_to_class)} clases")
        print(f"Clases disponibles: {list(self.index_to_class.values())}")
    
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
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img = cv2.resize(img, self.img_size)
        
        # Normalizar
        img = img.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path: str, top_k: int = 3) -> List[Dict[str, any]]:
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
    
    def predict_and_display(self, 
                           image_path: str,
                           save_path: Optional[str] = None) -> Dict[str, any]:
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
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, any]]:
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
                               confidence_threshold: float = 0.7) -> Optional[Dict[str, any]]:
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
    
    def get_model_info(self) -> Dict[str, any]:
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
