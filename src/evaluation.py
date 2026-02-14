"""
Módulo de evaluación del modelo de clasificación de residuos.
Calcula métricas y genera visualizaciones de rendimiento.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras


class ModelEvaluator:
    """
    Clase para evaluar el modelo de clasificación de residuos.
    """
    
    def __init__(self, model_path: str, class_mapping_path: str):
        """
        Inicializa el evaluador.
        
        Args:
            model_path: Ruta al modelo guardado
            class_mapping_path: Ruta al archivo de mapeo de clases
        """
        self.model = keras.models.load_model(model_path)
        
        # Cargar mapeo de clases
        with open(class_mapping_path, 'r') as f:
            self.index_to_class = json.load(f)
        
        # Convertir claves de string a int
        self.index_to_class = {int(k): v for k, v in self.index_to_class.items()}
        self.class_to_index = {v: k for k, v in self.index_to_class.items()}
    
    def evaluate(self, test_generator) -> Dict[str, float]:
        """
        Evalúa el modelo en datos de test.
        
        Args:
            test_generator: Generador de datos de test
            
        Returns:
            Diccionario con métricas
        """
        print("\n" + "="*60)
        print("EVALUANDO MODELO")
        print("="*60)
        
        # Evaluar
        results = self.model.evaluate(test_generator, verbose=1)
        
        # Crear diccionario de métricas
        metrics = {}
        metric_names = self.model.metrics_names
        for name, value in zip(metric_names, results):
            metrics[name] = float(value)
        
        # Imprimir resultados
        print("\nRESULTADOS:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print("="*60 + "\n")
        
        return metrics
    
    def predict_and_analyze(self, test_generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones y las compara con las etiquetas verdaderas.
        
        Args:
            test_generator: Generador de datos de test
            
        Returns:
            Tupla (predicciones, etiquetas_verdaderas)
        """
        # Obtener predicciones
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Obtener etiquetas verdaderas
        true_classes = test_generator.classes
        
        return predicted_classes, true_classes
    
    def generate_classification_report(self,
                                      test_generator,
                                      save_path: str = None) -> str:
        """
        Genera reporte de clasificación detallado.
        
        Args:
            test_generator: Generador de datos de test
            save_path: Ruta para guardar el reporte (opcional)
            
        Returns:
            Reporte como string
        """
        predicted_classes, true_classes = self.predict_and_analyze(test_generator)
        
        # Obtener nombres de clases
        class_names = [self.index_to_class[i] for i in sorted(self.index_to_class.keys())]
        
        # Generar reporte
        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=class_names,
            digits=4
        )
        
        print("\n" + "="*60)
        print("REPORTE DE CLASIFICACIÓN")
        print("="*60)
        print(report)
        print("="*60 + "\n")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Reporte guardado en: {save_path}")
        
        return report
    
    def plot_confusion_matrix(self,
                             test_generator,
                             save_path: str = None,
                             figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Grafica la matriz de confusión.
        
        Args:
            test_generator: Generador de datos de test
            save_path: Ruta para guardar la gráfica (opcional)
            figsize: Tamaño de la figura
        """
        predicted_classes, true_classes = self.predict_and_analyze(test_generator)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Obtener nombres de clases
        class_names = [self.index_to_class[i] for i in sorted(self.index_to_class.keys())]
        
        # Crear figura
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada en: {save_path}")
        else:
            plt.show()
    
    def analyze_per_class_accuracy(self, test_generator) -> Dict[str, float]:
        """
        Calcula accuracy por clase.
        
        Args:
            test_generator: Generador de datos de test
            
        Returns:
            Diccionario con accuracy por clase
        """
        predicted_classes, true_classes = self.predict_and_analyze(test_generator)
        
        # Calcular accuracy por clase
        per_class_accuracy = {}
        
        for class_idx in sorted(self.index_to_class.keys()):
            class_name = self.index_to_class[class_idx]
            
            # Filtrar muestras de esta clase
            mask = true_classes == class_idx
            if mask.sum() == 0:
                continue
            
            # Calcular accuracy
            correct = (predicted_classes[mask] == true_classes[mask]).sum()
            total = mask.sum()
            accuracy = correct / total
            
            per_class_accuracy[class_name] = float(accuracy)
        
        # Imprimir resultados
        print("\n" + "="*60)
        print("ACCURACY POR CLASE")
        print("="*60)
        for class_name, acc in sorted(per_class_accuracy.items()):
            print(f"  {class_name.capitalize()}: {acc:.4f}")
        print("="*60 + "\n")
        
        return per_class_accuracy
    
    def plot_per_class_accuracy(self,
                               test_generator,
                               save_path: str = None,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Grafica accuracy por clase.
        
        Args:
            test_generator: Generador de datos de test
            save_path: Ruta para guardar la gráfica (opcional)
            figsize: Tamaño de la figura
        """
        per_class_accuracy = self.analyze_per_class_accuracy(test_generator)
        
        # Crear gráfica de barras
        classes = list(per_class_accuracy.keys())
        accuracies = list(per_class_accuracy.values())
        
        plt.figure(figsize=figsize)
        bars = plt.bar(classes, accuracies, color='steelblue', alpha=0.8)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.title('Accuracy por Clase de Residuo')
        plt.xlabel('Clase')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1.1])
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    print("Módulo de evaluación cargado")
