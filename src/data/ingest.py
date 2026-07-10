"""
Módulo de ingesta de datos.

Este módulo es responsable de cargar el dataset clínico y las señales ECG
desde los archivos de entrada. No realiza transformaciones ni limpieza;
únicamente centraliza la lectura de datos para mantener el pipeline
modular y reproducible.

Autor: Valentín Moreno Vásquez
Proyecto: Prueba Técnica MLOps - Salva Health
"""

from pathlib import Path

import pandas as pd

from src.utils.config import RAW_DATA_PATH, SIGNALS_PATH

def load_patients() -> pd.DataFrame:
    """
    Carga el archivo pacientes.csv.

    Returns

    pd.DataFrame
        DataFrame con la información clínica de los pacientes.
    """

    patients_path = RAW_DATA_PATH / "pacientes.csv"

    return pd.read_csv(patients_path)
