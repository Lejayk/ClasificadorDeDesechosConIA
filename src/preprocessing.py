"""
Módulo de preprocesamiento de datos para imágenes de residuos.
Incluye normalización, augmentación y preparación de datos para entrenamiento.
"""

import os
import numpy as np
import cv2
import shutil
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
                 img_size: Tuple[int, int] = (64, 64),
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

    def create_test_generator(self, test_dir: str):
        """
        Crea un generador de test sin augmentación.

        Args:
            test_dir: Directorio con datos de test organizados por clase

        Returns:
            Generador de test
        """
        test_datagen = ImageDataGenerator(rescale=1./255)

        return test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def create_train_test_split(self,
                                data_dir: str,
                                output_dir: str = 'data/processed/trashnet_split',
                                test_split: float = 0.25,
                                random_state: int = 42,
                                overwrite: bool = False) -> Tuple[str, str]:
        """
        Crea una partición reproducible train/test (ej. 75/25) a partir de un directorio por clases.

        Args:
            data_dir: Directorio fuente con imágenes organizadas por clase
            output_dir: Directorio de salida para split generado
            test_split: Proporción para test
            random_state: Semilla para reproducibilidad
            overwrite: Si True, elimina split previo y lo regenera

        Returns:
            Tupla (train_dir, test_dir)
        """
        source_path = Path(data_dir)
        if not source_path.exists():
            raise ValueError(f"El directorio fuente no existe: {data_dir}")

        output_path = Path(output_dir)
        train_path = output_path / 'train'
        test_path = output_path / 'test'

        if overwrite and output_path.exists():
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"No se encontraron clases en: {data_dir}")

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_files = sorted([
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_ext
            ])

            if len(class_files) < 2:
                raise ValueError(
                    f"La clase '{class_name}' requiere al menos 2 imágenes para split train/test"
                )

            train_files, test_files = train_test_split(
                class_files,
                test_size=test_split,
                random_state=random_state,
                shuffle=True
            )

            class_train_path = train_path / class_name
            class_test_path = test_path / class_name
            class_train_path.mkdir(parents=True, exist_ok=True)
            class_test_path.mkdir(parents=True, exist_ok=True)

            for file_path in train_files:
                destination = class_train_path / file_path.name
                if not destination.exists():
                    shutil.copy2(file_path, destination)

            for file_path in test_files:
                destination = class_test_path / file_path.name
                if not destination.exists():
                    shutil.copy2(file_path, destination)

        return str(train_path), str(test_path)
    
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
