"""
Selección del mejor modelo entrenado.

Responsabilidades:
- Comparar el desempeño de los modelos entrenados.
- Seleccionar el mejor modelo según una métrica.
- Retornar el modelo ganador y sus métricas.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import pandas as pd


def select_best_model(
    results: pd.DataFrame,
    trained_models: dict,
    metric: str = "roc_auc",
) -> tuple:
    """
    Selecciona el mejor modelo entrenado.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame con las métricas de cada modelo.

    trained_models : dict
        Diccionario con los modelos entrenados.

    metric : str, default="roc_auc"
        Métrica utilizada para seleccionar el mejor modelo.

    Returns
    -------
    tuple
        (
            best_model,
            best_model_name,
            best_metrics
        )
    """

    if metric not in results.columns:
        raise ValueError(
            f"La métrica '{metric}' no existe."
        )

    best_row = results.loc[
        results[metric].idxmax()
    ]

    best_model_name = best_row["modelo"]

    best_model = trained_models[
        best_model_name
    ]

    print()
    print("MEJOR MODELO")

    print(f"Modelo seleccionado : {best_model_name}")
    print(f"{metric}            : {best_row[metric]:.4f}")

    return (
        best_model,
        best_model_name,
        best_row.to_dict(),
    )