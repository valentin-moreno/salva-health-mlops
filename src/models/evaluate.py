"""
Evaluación del modelo de Machine Learning.

Este módulo calcula las principales métricas de desempeño
para un modelo de clasificación binaria.

Responsabilidades:
- Generar predicciones.
- Calcular métricas de clasificación.
- Calcular matriz de confusión.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

from typing import Dict

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Evalúa el desempeño del modelo.

    Parameters
    ----------
    model
        Modelo previamente entrenado.

    X_test : pd.DataFrame
        Variables predictoras del conjunto de prueba.

    y_test : pd.Series
        Etiquetas reales.

    Returns
    -------
    dict
        Diccionario con las métricas calculadas.
    """

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "precision": precision_score(
            y_test,
            y_pred,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_prob,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ),
    }

    return metrics