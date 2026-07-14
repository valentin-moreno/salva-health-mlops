"""
Pruebas del pipeline de entrenamiento.

Responsabilidades:
- Verificar que el pipeline completo de entrenamiento
  pueda ejecutarse utilizando un dataset reducido.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""


from pathlib import Path

from src.pipeline.train_pipeline import run_training_pipeline


def test_training_pipeline() -> None:
    """
    Verifica que el pipeline de entrenamiento
    finalice correctamente utilizando un dataset
    de prueba.

    Returns
    -------
    None
    """

    dataset_path = (
        Path("tests")
        / "data"
        / "dataset_final.parquet"
    )

    results = run_training_pipeline(
        dataset_path=str(dataset_path),
    )

    assert results is not None

    assert len(results) == 3

    assert "roc_auc" in results.columns
