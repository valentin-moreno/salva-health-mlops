"""
Pipeline reproducible de entrenamiento.

Responsabilidades:
- Ejecutar el flujo completo desde la ingesta de datos hasta el registro
  del mejor modelo.
- Construir automáticamente el dataset final.
- Entrenar y registrar el mejor modelo mediante MLflow.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

from src.data.ingest import load_patients
from src.data.validation import validate_dataset
from src.data.preprocessing import preprocess_dataset
from src.data.save import save_parquet
from src.data.merge import merge_datasets

from src.features.extract_features import build_feature_dataset

from src.pipeline.train_pipeline import run_training_pipeline


def run_pipeline():
    """
    Ejecuta el pipeline completo de Machine Learning.

    Returns
    -------
    pd.DataFrame
        Resumen de métricas de los modelos entrenados.
    """

  
    print("SALVA HEALTH MLOPS PIPELINE")

    # 1. Ingesta

    print("\n[1/6] Ingestando datos clínicos...")

    patients = load_patients()

    # 2. Validación

    print("[2/6] Validando datos clínicos...")

    validate_dataset(patients)

    # 3. Preprocesamiento

    print("[3/6] Preprocesando datos clínicos...")

    patients = preprocess_dataset(patients)

    save_parquet(
        patients,
        "patients.parquet",
    )

    # 4. Extracción de características

    print("[4/6] Extrayendo características ECG...")

    features = build_feature_dataset()

    save_parquet(
        features,
        "features.parquet",
    )
    # 5. Construcción del dataset final

    print("[5/6] Construyendo dataset final...")

    dataset = merge_datasets()

    save_parquet(
        dataset,
        "dataset_final.parquet",
    )

    # 6. Entrenamiento

    print("[6/6] Entrenando modelos y registrando experimentos...")

    results = run_training_pipeline()

    print("PIPELINE FINALIZADO CORRECTAMENTE")

    return results


if __name__ == "__main__":
    run_pipeline()