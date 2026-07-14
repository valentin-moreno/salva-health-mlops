"""
Separación del dataset para entrenamiento.

Este módulo prepara el dataset final para Machine Learning.

Responsabilidades:
- Cargar el dataset final.
- Separar variables predictoras y objetivo.
- Eliminar columnas que no aportan al modelo.
- Dividir los datos en entrenamiento y prueba.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path
from src.data.ingest import load_processed_dataset
from src.utils.config import RANDOM_STATE
from src.utils.config import TEST_SIZE


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara las variables para entrenamiento.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset final.

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.

    y : pd.Series
        Variable objetivo.
    """

    X = df.drop(
        columns=[
            "id_paciente",
            "fecha_registro",
            "derivacion_ecg",
            "frecuencia_muestreo_hz",
            "etiqueta",
        ]
    )

    y = df["etiqueta"]

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Divide el dataset en entrenamiento y prueba.

    Parameters
    ----------
    X : pd.DataFrame
        Variables predictoras.

    y : pd.Series
        Variable objetivo.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def prepare_train_test(
    dataset_path: str | None = None,
):
    """
    Ejecuta el pipeline completo de preparación
    para entrenamiento.

    Parameters
    ----------
    dataset_path : str, optional
        Ruta alternativa del dataset. Si no se especifica,
        se utiliza dataset_final.parquet del directorio
        de datos procesados.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    if dataset_path is None:

        dataset = load_processed_dataset("dataset_final.parquet")

    else:

        dataset = pd.read_parquet(Path(dataset_path))

    X, y = prepare_features(dataset)

    return split_dataset(X, y)
