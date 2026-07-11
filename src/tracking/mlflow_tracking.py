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

    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_param(
            "model",
            model_name,
        )

        # Registrar métricas
        for metric_name, value in metrics.items():

            if metric_name != "confusion_matrix":

                mlflow.log_metric(
                    metric_name,
                    float(value),
                )


        # Guardado específico según modelo
        if model_name == "xgboost":

            mlflow.xgboost.log_model(
                model,
                name="model",
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model",
            )
