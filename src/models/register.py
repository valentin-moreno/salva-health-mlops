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

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
        ),

        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    return models