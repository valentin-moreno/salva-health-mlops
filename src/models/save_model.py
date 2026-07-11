"""
Persistencia del modelo seleccionado.

Este módulo almacena el mejor modelo entrenado para
su posterior uso en inferencia y despliegue.

Responsabilidades:
- Guardar el mejor modelo entrenado.
- Crear automáticamente el directorio de almacenamiento.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

from pathlib import Path

import joblib

MODEL_DIRECTORY = Path("artifacts/models")

def save_model(model) -> Path:
    """
    Guarda el mejor modelo entrenado.

    Parameters
    ----------
    model
        Modelo entrenado.

    Returns
    -------
    Path
        Ruta donde se almacenó el modelo.
    """

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIRECTORY
        / "best_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print()
    print("MODELO GUARDADO")

    print(f"Ruta: {model_path}")

    return model_path