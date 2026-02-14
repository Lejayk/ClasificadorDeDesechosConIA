"""
Módulo de diseño y definición del modelo de clasificación de residuos.
Implementa arquitecturas CNN para clasificación de imágenes.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from typing import Tuple, Optional


class WasteClassificationModel:
    """
    Clase para crear y configurar modelos de clasificación de residuos.
    """
    
    def __init__(self, 
                 num_classes: int,
                 img_size: Tuple[int, int] = (64, 64),
                 architecture: str = 'custom_cnn'):
        """
        Inicializa el modelo.
        
        Args:
            num_classes: Número de clases de residuos
            img_size: Tamaño de imagen de entrada
            architecture: Arquitectura del modelo ('custom_cnn', 'mobilenet', 'resnet', 'efficientnet')
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.architecture = architecture
        self.model = None
    
    def build_custom_cnn(self) -> models.Model:
        """
        Construye una CNN personalizada para clasificación de residuos.
        
        Returns:
            Modelo Keras
        """
        model = models.Sequential([
            layers.Input(shape=(*self.img_size, 3)),

            # Bloque 1
            layers.Conv2D(32, (2, 2), strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Bloque 2
            layers.Conv2D(32, (2, 2), strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Bloque 3
            layers.Conv2D(32, (2, 2), strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Clasificador
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model_name: str) -> models.Model:
        """
        Construye un modelo usando transfer learning.
        
        Args:
            base_model_name: Nombre del modelo base ('mobilenet', 'resnet', 'efficientnet')
            
        Returns:
            Modelo Keras
        """
        # Seleccionar modelo base
        if base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == 'resnet':
            base_model = ResNet50(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Modelo base no soportado: {base_model_name}")
        
        # Congelar el modelo base
        base_model.trainable = False
        
        # Construir modelo completo
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build(self) -> models.Model:
        """
        Construye el modelo según la arquitectura especificada.
        
        Returns:
            Modelo Keras compilado
        """
        if self.architecture == 'custom_cnn':
            self.model = self.build_custom_cnn()
        elif self.architecture in ['mobilenet', 'resnet', 'efficientnet']:
            self.model = self.build_transfer_learning_model(self.architecture)
        else:
            raise ValueError(f"Arquitectura no soportada: {self.architecture}")
        
        return self.model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam') -> None:
        """
        Compila el modelo con optimizador y función de pérdida.
        
        Args:
            learning_rate: Tasa de aprendizaje
            optimizer: Nombre del optimizador
        """
        if self.model is None:
            raise ValueError("Primero debe construir el modelo usando build()")
        
        # Seleccionar optimizador
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Optimizador no soportado: {optimizer}")
        
        # Compilar modelo
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall()]
        )
    
    def get_model_summary(self) -> None:
        """
        Imprime el resumen del modelo.
        """
        if self.model is None:
            raise ValueError("Primero debe construir el modelo usando build()")
        
        self.model.summary()
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo desde disco.
        
        Args:
            filepath: Ruta del modelo guardado
        """
        self.model = keras.models.load_model(filepath)
        print(f"Modelo cargado desde: {filepath}")


if __name__ == "__main__":
    # Ejemplo de uso
    model_builder = WasteClassificationModel(num_classes=6)
    model = model_builder.build()
    model_builder.compile_model()
    model_builder.get_model_summary()
