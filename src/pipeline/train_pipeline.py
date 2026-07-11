"""
Pipeline completo de entrenamiento.

Ejecuta de forma reproducible el proceso de entrenamiento,
evaluación y registro de experimentos.

Responsabilidades:
- Preparar los datos.
- Entrenar múltiples modelos.
- Evaluarlos.
- Registrar experimentos en MLflow.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import mlflow
import pandas as pd

from src.data.split import prepare_train_test
from src.models.register import get_models
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.tracking.mlflow_tracking import log_experiment


def run_training_pipeline() -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Returns
    -------
    tuple[pd.DataFrame, dict]

    - DataFrame con las métricas de todos los modelos.
    - Diccionario con los modelos entrenados.
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

    print()
    print("✓ Pipeline ejecutado correctamente.")

    return (
        pd.DataFrame(results),
        trained_models,
    )