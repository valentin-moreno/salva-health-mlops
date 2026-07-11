"""
Entrenamiento genérico de modelos de Machine Learning.

Este módulo permite entrenar diferentes algoritmos de clasificación
manteniendo un único flujo reproducible.

Responsabilidades:
- Recibir un modelo configurado.
- Entrenarlo con los datos disponibles.
- Retornar el modelo entrenado.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import pandas as pd


def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Entrena un modelo de clasificación.

    Parameters
    ----------
    model:
        Modelo compatible con scikit-learn.

    X_train:
        Variables predictoras de entrenamiento.

    y_train:
        Variable objetivo.

    Returns
    -------
    Modelo entrenado.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model