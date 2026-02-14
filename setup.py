#!/usr/bin/env python3
"""
Script de configuración inicial del sistema de clasificación de residuos.
Verifica requisitos y prepara el entorno.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True


def check_dependencies():
    """Verifica si las dependencias están instaladas."""
    try:
        import tensorflow
        print(f"✓ TensorFlow {tensorflow.__version__} instalado")
    except ImportError:
        print("⚠ TensorFlow no instalado")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} instalado")
    except ImportError:
        print("⚠ OpenCV no instalado")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__} instalado")
    except ImportError:
        print("⚠ NumPy no instalado")
        return False
    
    return True


def create_directory_structure():
    """Crea la estructura de directorios necesaria."""
    directories = [
        'data/raw/plastico',
        'data/raw/papel',
        'data/raw/vidrio',
        'data/raw/organico',
        'data/raw/metal',
        'data/raw/carton',
        'data/processed',
        'models',
        'models/logs',
        'notebooks'
    ]
    
    print("\nCreando estructura de directorios...")
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    return True


def install_dependencies():
    """Instala las dependencias desde requirements.txt."""
    print("\n¿Deseas instalar las dependencias ahora? (s/n): ", end='')
    response = input().strip().lower()
    
    if response == 's':
        print("\nInstalando dependencias...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✓ Dependencias instaladas exitosamente")
            return True
        except subprocess.CalledProcessError:
            print("❌ Error al instalar dependencias")
            return False
    else:
        print("\nPuedes instalar las dependencias manualmente con:")
        print("  pip install -r requirements.txt")
        return True


def display_next_steps():
    """Muestra los siguientes pasos al usuario."""
    print("\n" + "="*70)
    print("CONFIGURACIÓN COMPLETADA")
    print("="*70)
    print("\n📚 Próximos pasos:")
    print("\n1. Recopila imágenes de residuos para cada categoría:")
    print("   - Plástico, Papel, Vidrio, Orgánico, Metal, Cartón")
    print("   - Mínimo 100-200 imágenes por categoría")
    print("   - Recomendado: 500+ imágenes por categoría")
    
    print("\n2. Organiza las imágenes en la estructura de directorios:")
    print("   data/raw/[categoria]/imagen.jpg")
    
    print("\n3. Entrena el modelo:")
    print("   python train_model.py --data-dir data/raw --epochs 50")
    
    print("\n4. Clasifica imágenes:")
    print("   python predict.py --image ruta/a/imagen.jpg")
    
    print("\n📖 Documentación disponible en:")
    print("   - docs/GUIA_USUARIO.md")
    print("   - docs/DOCUMENTACION_TECNICA.md")
    print("   - notebooks/demo.ipynb")
    
    print("\n" + "="*70 + "\n")


def main():
    """Función principal."""
    print("\n" + "="*70)
    print("CONFIGURACIÓN DEL SISTEMA DE CLASIFICACIÓN DE RESIDUOS CON IA")
    print("="*70 + "\n")
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Verificar dependencias
    print("\nVerificando dependencias...")
    dependencies_ok = check_dependencies()
    
    if not dependencies_ok:
        print("\n⚠ Algunas dependencias no están instaladas")
        install_dependencies()
    else:
        print("\n✓ Todas las dependencias están instaladas")
    
    # Crear estructura de directorios
    create_directory_structure()
    
    # Mostrar siguientes pasos
    display_next_steps()


if __name__ == "__main__":
    main()
