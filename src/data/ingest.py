"""
Módulo de ingesta de datos.

Responsabilidades:
- Conectarse a Azure Blob Storage.
- Leer los archivos almacenados.
- Entregar los datos al resto del pipeline.

Este módulo NO realiza validaciones ni transformaciones.
"""

from io import BytesIO

import pandas as pd

from src.utils.config import PROCESSED_DATA_PATH

from azure.storage.blob import BlobServiceClient, ContainerClient

from src.utils.secrets import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_CONTAINER_NAME,
)


def _get_container_client() -> ContainerClient:
    """
    Crea la conexión con el contenedor de Azure Blob Storage.

    Returns
    -------
    ContainerClient
        Cliente del contenedor.
    """

    blob_service = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    return blob_service.get_container_client(AZURE_CONTAINER_NAME)


def load_patients() -> pd.DataFrame:
    """
    Carga el archivo pacientes.csv desde Azure Blob Storage.

    Returns
    -------
    pd.DataFrame
        Información clínica de los pacientes.
    """

    container = _get_container_client()

    blob = container.get_blob_client("pacientes.csv")

    data = blob.download_blob().readall()

    return pd.read_csv(BytesIO(data))


def list_signals() -> list[str]:
    """
    Obtiene la lista de señales ECG disponibles.

    Returns
    -------
    list[str]
        IDs de pacientes con señal ECG.
    """

    container = _get_container_client()

    signals = []

    for blob in container.list_blobs(name_starts_with="senales/"):

        if blob.name.endswith(".csv"):

            patient_id = blob.name.split("/")[-1].replace(".csv", "")

            signals.append(patient_id)

    return sorted(signals)


def load_signal(patient_id: str) -> pd.DataFrame:
    """
    Carga la señal ECG de un paciente.

    Parameters
    ----------
    patient_id : str
        Identificador del paciente.

    Returns
    -------
    pd.DataFrame
        Señal ECG del paciente.
    """

    container = _get_container_client()

    blob = container.get_blob_client(f"senales/{patient_id}.csv")

    data = blob.download_blob().readall()

    return pd.read_csv(BytesIO(data))


def load_processed_dataset(
    filename: str,
) -> pd.DataFrame:
    """
    Carga un dataset procesado almacenado en formato Parquet.

    Parameters
    ----------
    filename : str
        Nombre del archivo Parquet ubicado en data/processed.

    Returns
    -------
    pd.DataFrame
        Dataset procesado.
    """

    filepath = PROCESSED_DATA_PATH / filename

    return pd.read_parquet(filepath)
