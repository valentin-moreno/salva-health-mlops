"""
Configuración centralizada del proyecto.

Este módulo define las rutas principales utilizadas por el pipeline de
MLOps. Centralizar las rutas evita valores hardcodeados en el código y
facilita el mantenimiento y la reproducibilidad del proyecto.

Autor: Valentín Moreno Vásquez
Proyecto: Prueba Técnica MLOps - Salva Health
"""

from pathlib import Path

# RUTA RAÍZ DEL PROYECTO

# Obtiene automáticamente la carpeta raíz del proyecto
# (salva-health-mlops/), sin importar desde dónde se ejecute el código.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# RUTAS DE DATOS

# Directorio donde se almacenan los datos crudos.
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

# Directorio donde se almacenan los datos procesados.
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# RUTAS DE ARTEFACTOS

# Directorio para almacenar modelos entrenados.
MODELS_PATH = PROJECT_ROOT / "models"

# Directorio para artefactos generados durante el pipeline.
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"

# Directorio utilizado por MLflow para registrar experimentos.
MLRUNS_PATH = PROJECT_ROOT / "mlruns"

# Directorio para documentación del proyecto.
DOCS_PATH = PROJECT_ROOT / "docs"

# Directorio para pruebas unitarias.
TESTS_PATH = PROJECT_ROOT / "tests"

# CONFIGURACIÓN DE ENTRENAMIENTO

# Semilla para garantizar reproducibilidad.
RANDOM_STATE = 42

# Proporción del conjunto de prueba.
TEST_SIZE = 0.20
