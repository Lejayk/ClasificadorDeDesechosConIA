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


class EpochTrackerCallback(keras.callbacks.Callback):
    """Callback para exponer la época actual al modelo adversarial."""

    def __init__(self, adv_model):
        super().__init__()
        self.adv_model = adv_model

    def on_epoch_begin(self, epoch, logs=None):
        self.adv_model.current_epoch = int(epoch)


class AdversarialTrainingModel(keras.Model):
    """Wrapper Keras para entrenamiento adversarial FGSM básico."""

    def __init__(self, base_model: keras.Model, epsilon: float, adv_ratio: float, adv_start_epoch: int = 0):
        super().__init__()
        self.base_model = base_model
        self.epsilon = float(epsilon)
        self.adv_ratio = float(np.clip(adv_ratio, 0.0, 1.0))
        self.adv_start_epoch = int(max(0, adv_start_epoch))
        self.current_epoch = 0

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data

        use_adversarial = self.epsilon > 0 and self.adv_ratio > 0 and self.current_epoch >= self.adv_start_epoch

        if use_adversarial:
            with tf.GradientTape() as adv_tape:
                adv_tape.watch(x)
                clean_pred = self(x, training=True)
                clean_loss = self.compiled_loss(y, clean_pred, regularization_losses=self.losses)

            input_grads = adv_tape.gradient(clean_loss, x)
            x_adv = x + self.epsilon * tf.sign(input_grads)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

            batch_size = tf.shape(x)[0]
            mask = tf.cast(
                tf.random.uniform((batch_size, 1, 1, 1), minval=0.0, maxval=1.0) < self.adv_ratio,
                dtype=x.dtype
            )
            x_train = (mask * x_adv) + ((1.0 - mask) * x)
        else:
            x_train = x

        with tf.GradientTape() as tape:
            y_pred = self(x_train, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss

        return results


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
                monitor='val_loss',
                mode='min',
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
              epochs: int = 70,
              model_name: str = "waste_classifier",
              fgsm_epsilon: float = 0.0,
              adv_ratio: float = 0.5,
              adv_start_epoch: int = 5) -> Dict[str, Any]:
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
        print(f"FGSM epsilon: {fgsm_epsilon}")
        print(f"FGSM ratio adversarial: {adv_ratio}")
        print(f"FGSM inicio en época: {adv_start_epoch}")
        print("="*60 + "\n")
        
        # Crear callbacks
        callbacks = self.create_callbacks(model_name)

        training_model = self.model.model
        adv_model = None

        if fgsm_epsilon > 0 and adv_ratio > 0:
            optimizer_config = self.model.model.optimizer.get_config()
            optimizer_class = self.model.model.optimizer.__class__
            optimizer = optimizer_class.from_config(optimizer_config)

            adv_model = AdversarialTrainingModel(
                base_model=self.model.model,
                epsilon=fgsm_epsilon,
                adv_ratio=adv_ratio,
                adv_start_epoch=adv_start_epoch
            )

            adv_model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=[
                    keras.metrics.CategoricalAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )

            callbacks = [EpochTrackerCallback(adv_model)] + callbacks
            training_model = adv_model
            print("Entrenamiento adversarial FGSM activado")
        else:
            print("Entrenamiento estándar (sin FGSM)")
        
        # Entrenar modelo
        self.history = training_model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        if adv_model is not None:
            self.model.model.set_weights(adv_model.base_model.get_weights())
        
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
