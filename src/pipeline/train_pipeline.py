"""
Pipeline completo de entrenamiento.

Ejecuta de forma reproducible el proceso de entrenamiento,
evaluación, registro de experimentos, selección del mejor
modelo y almacenamiento para despliegue.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import mlflow
import pandas as pd

from src.data.split import prepare_train_test
from src.models.register import get_models
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.select_best import select_best_model
from src.models.save_model import save_model
from src.tracking.mlflow_tracking import log_experiment


def run_training_pipeline() -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Returns
    -------
    pd.DataFrame
        Resumen de métricas obtenidas por cada modelo.
    """

    print("PIPELINE DE ENTRENAMIENTO")

    mlflow.set_experiment(
        "Salva_Health_Model_Comparison"
    )

    X_train, X_test, y_train, y_test = prepare_train_test()

    models = get_models()

    results = []

    trained_models = {}

    for name, model in models.items():

        print()

        print(f"Entrenando: {name}")

        trained_model = train_model(
            model,
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            trained_model,
            X_test,
            y_test,
        )

        log_experiment(
            trained_model,
            name,
            metrics,
        )

        trained_models[name] = trained_model

        results.append(
            {
                "modelo": name,
                **metrics,
            }
        )

    results = pd.DataFrame(results)

    best_model, best_model_name, best_metrics = select_best_model(
        results,
        trained_models,
    )

    save_model(best_model)

    print()

    print("PIPELINE FINALIZADO")

    print(f"Modelo seleccionado : {best_model_name}")
    print(f"ROC AUC             : {best_metrics['roc_auc']:.4f}")

    print()

    print(" Modelo almacenado correctamente.")

    return results