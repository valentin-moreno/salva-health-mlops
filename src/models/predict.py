"""
Inferencia utilizando el modelo entrenado.

Responsabilidades:
- Generar predicciones a partir de un modelo entrenado.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import pandas as pd


def predict(
    model,
    X: pd.DataFrame,
):
    """
    Genera predicciones utilizando un modelo entrenado.

    Parameters
    ----------
    model
        Modelo previamente cargado.

    X : pd.DataFrame
        Variables predictoras.

    Returns
    -------
    tuple
        Predicciones y probabilidades.
    """

    predictions = model.predict(
        X,
    )

    probabilities = model.predict_proba(
        X,
    )

    return (
        predictions,
        probabilities,
    )
