"""
Utilidades para seguimiento de experimentos con MLflow.

Registra parámetros, métricas y modelos entrenados.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost


def log_experiment(
    model,
    model_name: str,
    metrics: dict,
):
    """
    Registra un experimento en MLflow.
    """

    with mlflow.start_run(run_name=model_name):

        # Información general
        mlflow.set_tags(
            {
                "author": "Valentin Moreno Vásquez",
                "project": "Salva Health MLOps",
                "stage": "training",
                "framework": model_name,
            }
        )

        # Nombre del modelo
        mlflow.log_param(
            "model",
            model_name,
        )

        # Hiperparámetros
        mlflow.log_params(
            model.get_params()
        )

        # Métricas
        for metric_name, value in metrics.items():

            if metric_name != "confusion_matrix":

                mlflow.log_metric(
                    metric_name,
                    float(value),
                )

        # Modelo
        if model_name.lower() == "xgboost":

            mlflow.xgboost.log_model(
                model,
                name="model",
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model",
            )
