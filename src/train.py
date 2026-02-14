"""
Módulo de entrenamiento del modelo de clasificación de residuos.
Maneja el proceso de entrenamiento, callbacks y guardado del modelo.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import numpy as np

from model import WasteClassificationModel
from preprocessing import DataPreprocessor


class ModelTrainer:
    """
    Clase para entrenar el modelo de clasificación de residuos.
    """
    
    def __init__(self, 
                 model: WasteClassificationModel,
                 output_dir: str = "models"):
        """
        Inicializa el entrenador.
        
        Args:
            model: Modelo a entrenar
            output_dir: Directorio para guardar modelos y resultados
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = None
    
    def create_callbacks(self, model_name: str = "waste_classifier") -> list:
        """
        Crea callbacks para el entrenamiento.
        
        Args:
            model_name: Nombre base para los archivos del modelo
            
        Returns:
            Lista de callbacks
        """
        callbacks = [
            # Guardar mejor modelo
            ModelCheckpoint(
                filepath=str(self.output_dir / f"{model_name}_best.h5"),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reducir learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=str(self.output_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self,
              train_generator,
              validation_generator,
              epochs: int = 50,
              model_name: str = "waste_classifier") -> Dict[str, Any]:
        """
        Entrena el modelo.
        
        Args:
            train_generator: Generador de datos de entrenamiento
            validation_generator: Generador de datos de validación
            epochs: Número de épocas
            model_name: Nombre del modelo
            
        Returns:
            Historial de entrenamiento
        """
        if self.model.model is None:
            raise ValueError("El modelo no ha sido construido ni compilado")
        
        print("\n" + "="*60)
        print("INICIANDO ENTRENAMIENTO")
        print("="*60)
        print(f"Épocas: {epochs}")
        print(f"Muestras de entrenamiento: {train_generator.samples}")
        print(f"Muestras de validación: {validation_generator.samples}")
        print(f"Clases: {list(train_generator.class_indices.keys())}")
        print("="*60 + "\n")
        
        # Crear callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Entrenar modelo
        self.history = self.model.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar modelo final
        final_model_path = self.output_dir / f"{model_name}_final.h5"
        self.model.save_model(str(final_model_path))
        
        # Guardar historial
        self.save_training_history(model_name)
        
        # Guardar mapeo de clases
        self.save_class_mapping(train_generator.class_indices, model_name)
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*60)
        
        return self.history.history
    
    def save_training_history(self, model_name: str) -> None:
        """
        Guarda el historial de entrenamiento.
        
        Args:
            model_name: Nombre del modelo
        """
        if self.history is None:
            return
        
        history_path = self.output_dir / f"{model_name}_history.json"
        
        # Convertir a formato serializable
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Historial guardado en: {history_path}")
    
    def save_class_mapping(self, class_indices: Dict[str, int], model_name: str) -> None:
        """
        Guarda el mapeo de clases.
        
        Args:
            class_indices: Diccionario de clases a índices
            model_name: Nombre del modelo
        """
        mapping_path = self.output_dir / f"{model_name}_classes.json"
        
        # Invertir el mapeo (índice -> clase)
        index_to_class = {v: k for k, v in class_indices.items()}
        
        with open(mapping_path, 'w') as f:
            json.dump(index_to_class, f, indent=2)
        
        print(f"Mapeo de clases guardado en: {mapping_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Grafica el historial de entrenamiento.
        
        Args:
            save_path: Ruta para guardar la gráfica (opcional)
        """
        if self.history is None:
            print("No hay historial de entrenamiento para graficar")
            return
        
        history_dict = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history_dict['accuracy'], label='Train')
        axes[0, 0].plot(history_dict['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history_dict['loss'], label='Train')
        axes[0, 1].plot(history_dict['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history_dict:
            axes[1, 0].plot(history_dict['precision'], label='Train')
            axes[1, 0].plot(history_dict['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history_dict:
            axes[1, 1].plot(history_dict['recall'], label='Train')
            axes[1, 1].plot(history_dict['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    print("Módulo de entrenamiento cargado")
