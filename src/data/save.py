"""
Funciones para guardar datasets procesados.

Este módulo centraliza el almacenamiento de los datasets generados
durante el pipeline en formato Parquet.

Autor: Valentín Moreno Vásquez
Proyecto: Prueba Técnica MLOps - Salva Health
"""

import pandas as pd

from src.utils.config import PROCESSED_DATA_PATH


def save_parquet(
    df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Guarda un DataFrame en formato Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que se desea almacenar.

    filename : str
        Nombre del archivo Parquet.
    """

    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = PROCESSED_DATA_PATH / filename

    df.to_parquet(
        output_path,
        index=False,
    )

    print(f"Archivo guardado en: {output_path}")
