"""
Carga del modelo entrenado.

Responsabilidades:
- Cargar el mejor modelo entrenado para inferencia.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""


from pathlib import Path

import joblib


MODEL_PATH = Path(
    "artifacts/models/best_model.joblib"
)


def load_model():
    """
    Carga el mejor modelo entrenado.

    Returns
    -------
    object
        Modelo entrenado.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró el modelo en: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH,
    )

    print()

    print("MODELO CARGADO")

    print(f"Ruta: {MODEL_PATH}")

    return model
