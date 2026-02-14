"""
Módulo de preprocesamiento de datos para imágenes de residuos.
Incluye normalización, augmentación y preparación de datos para entrenamiento.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Clase para preprocesar imágenes de residuos.
    """
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32):
        """
        Inicializa el preprocesador.
        
        Args:
            img_size: Tamaño de imagen (altura, ancho)
            batch_size: Tamaño de lote para entrenamiento
        """
        self.img_size = img_size
        self.batch_size = batch_size
        
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Carga y preprocesa una imagen individual.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Imagen preprocesada como array numpy
        """
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img = cv2.resize(img, self.img_size)
        
        # Normalizar píxeles a rango [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def create_data_generators(self, 
                               data_dir: str,
                               validation_split: float = 0.2,
                               use_augmentation: bool = True) -> Tuple:
        """
        Crea generadores de datos para entrenamiento y validación.
        
        Args:
            data_dir: Directorio con datos organizados por categoría
            validation_split: Proporción de datos para validación
            use_augmentation: Si aplicar data augmentation
            
        Returns:
            Tupla con (train_generator, validation_generator)
        """
        if use_augmentation:
            # Augmentación para entrenamiento
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=validation_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        # Generador de validación (sin augmentación)
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Crear generadores
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocesa un lote de imágenes.
        
        Args:
            images: Array de imágenes
            
        Returns:
            Imágenes preprocesadas
        """
        # Normalizar si es necesario
        if images.max() > 1.0:
            images = images / 255.0
        
        return images
    
    def get_class_names(self, data_dir: str) -> List[str]:
        """
        Obtiene los nombres de las clases del directorio de datos.
        
        Args:
            data_dir: Directorio con datos
            
        Returns:
            Lista de nombres de clases
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"El directorio {data_dir} no existe")
        
        classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        return classes


if __name__ == "__main__":
    # Ejemplo de uso
    preprocessor = DataPreprocessor()
    print("Preprocesador inicializado")
    print(f"Tamaño de imagen: {preprocessor.img_size}")
    print(f"Tamaño de lote: {preprocessor.batch_size}")
