"""
Validación de datos clínicos.

Este módulo contiene reglas de calidad para verificar que los datos
recibidos desde Azure Blob Storage cumplen las condiciones esperadas
antes de entrar al pipeline de procesamiento y entrenamiento.

Elaborado por: Valentin Moreno Vasquez
Proyecto: Salva Health MLOps

"""

import pandas as pd

# Columnas esperadas del dataset clínico
REQUIRED_COLUMNS = [
    "id_paciente",
    "edad_paciente",
    "sexo",
    "peso_kg",
    "altura_cm",
    "fecha_registro",
    "frecuencia_cardiaca_media_bpm",
    "derivacion_ecg",
    "frecuencia_muestreo_hz",
    "etiqueta",
]


# Etiquetas permitidas por el problema de clasificación
VALID_LABELS = [
    "Normal",
    "Anormal",
]


def validate_columns(df: pd.DataFrame) -> None:
    """
    Verifica que el dataset contenga todas las columnas requeridas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset clínico de pacientes.

    Raises
    ------
    ValueError
        Si existen columnas faltantes.
    """

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Columnas faltantes en dataset: {missing_columns}")


def validate_missing_values(df: pd.DataFrame) -> None:
    """
    Detecta valores faltantes en el dataset.

    Los valores faltantes no detienen el pipeline porque serán tratados
    posteriormente durante la etapa de preprocessing.
    """

    missing = df.isnull().sum()

    columns_with_missing = missing[missing > 0]

    if not columns_with_missing.empty:
        print("Advertencia: se encontraron valores nulos:")
        print(columns_with_missing)


def validate_labels(df: pd.DataFrame) -> None:
    """
    Verifica que las etiquetas de clasificación sean válidas.

    Raises
    ------
    ValueError
        Si existen etiquetas diferentes a Normal o Anormal.
    """

    labels = set(df["etiqueta"].dropna().unique())

    invalid_labels = labels - set(VALID_LABELS)

    if invalid_labels:
        raise ValueError(f"Etiquetas inválidas encontradas: {invalid_labels}")


def validate_ecg_configuration(df: pd.DataFrame) -> None:
    """
    Valida parámetros técnicos asociados al ECG.

    Se espera:
    - Derivación II.
    - Frecuencia de muestreo de 250 Hz.
    """

    if not all(df["derivacion_ecg"] == "II"):
        raise ValueError("Se encontraron derivaciones ECG diferentes a II")

    if not all(df["frecuencia_muestreo_hz"] == 250):
        raise ValueError("La frecuencia de muestreo debe ser 250 Hz")


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Ejecuta todas las validaciones principales del dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset clínico.

    Returns
    -------
    bool
        True si todas las validaciones críticas fueron exitosas.
    """

    validate_columns(df)
    validate_missing_values(df)
    validate_labels(df)
    validate_ecg_configuration(df)

    return True


# VALIDACION DE SEÑALES ECG

# Columnas esperadas de cada señal ECG
REQUIRED_SIGNAL_COLUMNS = [
    "t_seg",
    "ecg_mV",
]

# Longitud esperada de cada señal
EXPECTED_SIGNAL_LENGTH = 2500


# VALIDACION DE COLUMNAS DE SEÑALES ECG
def validate_signal_columns(signal: pd.DataFrame) -> None:
    """
    Verifica que la señal tenga las columnas esperadas.
    """

    missing_columns = set(REQUIRED_SIGNAL_COLUMNS) - set(signal.columns)

    if missing_columns:
        raise ValueError(f"Columnas faltantes en la señal: {missing_columns}")


def validate_signal_missing_values(
    signal: pd.DataFrame,
) -> None:
    """
    Verifica que la señal no contenga valores nulos.
    """

    if signal.isnull().values.any():
        raise ValueError("La señal contiene valores nulos.")


def validate_signal_length(
    signal: pd.DataFrame,
) -> None:
    """
    Verifica que la señal tenga la longitud esperada.
    """

    if len(signal) != EXPECTED_SIGNAL_LENGTH:
        raise ValueError(
            f"La señal contiene {len(signal)} muestras "
            f"y se esperaban {EXPECTED_SIGNAL_LENGTH}."
        )


def validate_signal_dtypes(
    signal: pd.DataFrame,
) -> None:
    """
    Verifica que las columnas sean numéricas.
    """

    if not pd.api.types.is_numeric_dtype(signal["t_seg"]):
        raise ValueError("La columna t_seg debe ser numérica.")

    if not pd.api.types.is_numeric_dtype(signal["ecg_mV"]):
        raise ValueError("La columna ecg_mV debe ser numérica.")


def validate_signal(
    signal: pd.DataFrame,
) -> bool:
    """
    Ejecuta todas las validaciones de una señal ECG.
    """

    validate_signal_columns(signal)
    validate_signal_missing_values(signal)
    validate_signal_length(signal)
    validate_signal_dtypes(signal)

    return True
