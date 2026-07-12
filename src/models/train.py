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

"""
Entrenamiento de modelos.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Entrena un modelo de clasificación.

    Logistic Regression se utiliza como baseline.

    Random Forest y XGBoost se optimizan mediante
    RandomizedSearchCV.

    Parameters
    ----------
    model
        Modelo compatible con scikit-learn.

    X_train
        Variables predictoras.

    y_train
        Variable objetivo.

    Returns
    -------
    Modelo entrenado.
    """

    # Baseline
    if isinstance(
        model,
        LogisticRegression,
    ):

        model.fit(
            X_train,
            y_train,
        )

        return model

    # Random Forest
    if isinstance(
        model,
        RandomForestClassifier,
    ):

        param_distributions = {

            "n_estimators": [
                100,
                200,
                300,
                500,
            ],

            "max_depth": [
                4,
                6,
                8,
                10,
                None,
            ],

            "min_samples_split": [
                2,
                5,
                10,
            ],

            "min_samples_leaf": [
                1,
                2,
                4,
            ],

            "max_features": [
                "sqrt",
                "log2",
            ],
        }

    # XGBoost
    elif isinstance(
        model,
        XGBClassifier,
    ):

        param_distributions = {

            "n_estimators": [
                100,
                200,
                300,
            ],

            "max_depth": [
                3,
                4,
                5,
                6,
                8,
            ],

            "learning_rate": [
                0.01,
                0.05,
                0.1,
                0.2,
            ],

            "subsample": [
                0.8,
                0.9,
                1.0,
            ],

            "colsample_bytree": [
                0.8,
                0.9,
                1.0,
            ],
        }

    else:

        model.fit(
            X_train,
            y_train,
        )

        return model

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_distributions,

        n_iter=20,

        scoring="roc_auc",

        cv=5,

        random_state=42,

        n_jobs=-1,

        verbose=1,

    )

    search.fit(

        X_train,

        y_train,

    )

    print()

    print("MEJORES HIPERPARÁMETROS")

    for key, value in search.best_params_.items():

        print(f"{key}: {value}")

    print()

    print(
        f"Mejor ROC AUC (CV): {search.best_score_:.4f}"
    )

    return search.best_estimator_