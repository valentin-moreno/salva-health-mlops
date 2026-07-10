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
from azure.storage.blob import BlobServiceClient

from src.utils.secrets import (AZURE_STORAGE_CONNECTION_STRING,AZURE_CONTAINER_NAME,)


def load_patients() -> pd.DataFrame:
    """
    Carga el archivo pacientes.csv desde Azure Blob Storage.

    Returns
    -------
    pd.DataFrame
        Información clínica de los pacientes.
    """

    blob_service = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    container = blob_service.get_container_client(
        AZURE_CONTAINER_NAME
    )

    blob = container.get_blob_client("pacientes.csv")

    data = blob.download_blob().readall()

    patients = pd.read_csv(BytesIO(data))

    return patients
