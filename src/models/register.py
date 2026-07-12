"""
Catálogo de modelos utilizados en los experimentos.

Cada modelo representa una variante del pipeline
de entrenamiento y evaluación.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models():
    """
    Devuelve los modelos candidatos para el proceso
    de comparación.

    Returns
    -------
    dict
        Diccionario de modelos.
    """

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            random_state=42,
        ),

        "xgboost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        ),
    }

    return models