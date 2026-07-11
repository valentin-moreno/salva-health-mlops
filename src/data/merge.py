"""
Unión de datasets procesados.

Este módulo combina la información clínica de los pacientes con las
características extraídas de las señales ECG para construir el dataset
final de entrenamiento.

Autor: Valentín Moreno Vásquez
Proyecto: Prueba Técnica MLOps - Salva Health
"""


import pandas as pd

from src.utils.config import PROCESSED_DATA_PATH


def merge_datasets() -> pd.DataFrame:
    """
    Une el dataset clínico con las características del ECG.

    Returns
    -------
    pd.DataFrame
        Dataset final listo para entrenamiento.
    """

    patients_path = PROCESSED_DATA_PATH / "patients.parquet"
    features_path = PROCESSED_DATA_PATH / "features.parquet"

    patients = pd.read_parquet(patients_path)
    features = pd.read_parquet(features_path)

    dataset = patients.merge(
        features,
        on="id_paciente",
        how="inner",
        validate="one_to_one",
    )

    return dataset
