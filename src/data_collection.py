"""
Módulo para la recopilación de datos de imágenes de residuos.
Proporciona utilidades para descargar y organizar datasets de residuos.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple
import shutil

class DataCollector:
    """
    Clase para recopilar y organizar datos de imágenes de residuos.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Inicializa el recolector de datos.
        
        Args:
            data_dir: Directorio donde se guardarán los datos
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Categorías de residuos
        self.categories = [
            'glass',
            'paper',
            'cardboard',
            'plastic',
            'metal',
            'trash'
        ]
    
    def create_category_directories(self) -> None:
        """
        Crea directorios para cada categoría de residuo.
        """
        for category in self.categories:
            category_path = self.data_dir / category
            category_path.mkdir(parents=True, exist_ok=True)
            print(f"Directorio creado: {category_path}")
    
    def get_categories(self) -> List[str]:
        """
        Retorna la lista de categorías de residuos.
        
        Returns:
            Lista de categorías
        """
        return self.categories
    
    def organize_dataset(self, source_dir: str) -> None:
        """
        Organiza imágenes desde un directorio fuente en categorías.
        
        Args:
            source_dir: Directorio fuente con imágenes
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ValueError(f"El directorio {source_dir} no existe")
        
        self.create_category_directories()
        
        print(f"Organizando dataset desde {source_dir}...")
        print("Nota: Las imágenes deben estar organizadas por categoría en el directorio fuente")
    
    def validate_dataset(self) -> Tuple[int, dict]:
        """
        Valida el dataset contando imágenes por categoría.
        
        Returns:
            Tupla con (total_imágenes, dict_categorías)
        """
        total_images = 0
        category_counts = {}
        
        for category in self.categories:
            category_path = self.data_dir / category
            if category_path.exists():
                images = list(category_path.glob('*.jpg')) + \
                        list(category_path.glob('*.png')) + \
                        list(category_path.glob('*.jpeg'))
                count = len(images)
                category_counts[category] = count
                total_images += count
        
        return total_images, category_counts
    
    def print_dataset_summary(self) -> None:
        """
        Imprime un resumen del dataset recopilado.
        """
        total, counts = self.validate_dataset()
        
        print("\n" + "="*50)
        print("RESUMEN DEL DATASET")
        print("="*50)
        print(f"\nTotal de imágenes: {total}\n")
        print("Imágenes por categoría:")
        for category, count in counts.items():
            print(f"  - {category.capitalize()}: {count}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    collector = DataCollector()
    collector.create_category_directories()
    collector.print_dataset_summary()
