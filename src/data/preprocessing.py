"""
Preprocesamiento de variables clínicas.

Este módulo transforma los datos clínicos obtenidos desde Azure Blob
Storage a un formato compatible con los modelos de Machine Learning.

Responsabilidades:
- Limpieza de fechas.
- Tratamiento de valores faltantes.
- Codificación de variables categóricas.
- Preparación de variable objetivo.
- Eliminación de duplicados.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""


import pandas as pd


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza diferentes formatos de fecha en fecha_registro.

    Maneja formatos como:
    - YYYY-MM-DD          -> 2023-10-04
    - DD/MM/YYYY          -> 04/10/2023
    - DD-MM-YYYY          -> 04-10-2023
    - DD Month YYYY       -> 15 abril 2025
    - DD Mon YYYY         -> 15 Apr 2025

    Las fechas no interpretables se convierten en NaT y se reportan.
    """


    df = df.copy()

    # Convertir todo a string para manejar formatos mezclados
    df["fecha_registro_original"] = df["fecha_registro"].astype(str)

    # Primera conversión flexible
    df["fecha_registro"] = pd.to_datetime(
        df["fecha_registro_original"],
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )

    # Reportar fechas que no pudieron convertirse
    invalid_dates = df["fecha_registro"].isna().sum()

    if invalid_dates > 0:
        print(
            f"Advertencia: {invalid_dates} fechas no pudieron convertirse"
        )

        print(
            df.loc[
                df["fecha_registro"].isna(),
                "fecha_registro_original"
            ].head(10)
        )

    # Eliminar columna auxiliar
    df.drop(
        columns=["fecha_registro_original"],
        inplace=True,
    )

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en variables clínicas.

    Estrategia:
    - Variables numéricas: mediana.
    - Variables categóricas: moda.
    """

    df = df.copy()

    numeric_columns = [
        "edad_paciente",
        "peso_kg",
        "altura_cm",
        "frecuencia_cardiaca_media_bpm",
    ]

    categorical_columns = [
        "sexo",
    ]

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    for column in categorical_columns:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    return df


def encode_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables categóricas a valores numéricos.
    """

    df = df.copy()

    df["sexo"] = df["sexo"].map(
        {
            "M": 1,
            "F": 0,
        }
    )

    df["etiqueta"] = df["etiqueta"].map(
        {
            "Normal": 0,
            "Anormal": 1,
        }
    )

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros completamente duplicados.

    No elimina pacientes con el mismo ID si contienen
    información clínica diferente.
    """
    df = df.drop_duplicates(subset="id_paciente")

    return df

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset validado.

    Returns
    -------
    pd.DataFrame
        Dataset preparado para entrenamiento.
    """

    df = clean_dates(df)

    df = handle_missing_values(df)

    df = encode_variables(df)

    df = remove_duplicates(df)

    return df
