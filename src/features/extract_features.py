"""
Extracción de características de señales ECG.

Responsabilidades:
- Calcular características estadísticas.
- Calcular características de energía.
- Generar un vector de características por señal.
- Construir el dataset completo de características.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import numpy as np
import pandas as pd

from src.data.ingest import list_signals, load_signal
from src.data.validation import validate_signal


def extract_statistical_features(
    signal: pd.DataFrame,
) -> dict:
    """
    Extrae características estadísticas de una señal ECG.
    """

    ecg = signal["ecg_mV"]

    return {
        "ecg_mean": ecg.mean(),
        "ecg_median": ecg.median(),
        "ecg_std": ecg.std(),
        "ecg_variance": ecg.var(),
        "ecg_minimum": ecg.min(),
        "ecg_maximum": ecg.max(),
        "ecg_range": ecg.max() - ecg.min(),
        "ecg_q25": ecg.quantile(0.25),
        "ecg_q75": ecg.quantile(0.75),
        "ecg_iqr": ecg.quantile(0.75) - ecg.quantile(0.25),
    }


def extract_energy_features(
    signal: pd.DataFrame,
) -> dict:
    """
    Extrae características relacionadas con la energía de la señal.
    """

    ecg = signal["ecg_mV"].to_numpy()

    energy = np.sum(ecg ** 2)
    rms = np.sqrt(np.mean(ecg ** 2))
    power = energy / len(ecg)

    return {
        "ecg_energy": energy,
        "ecg_rms": rms,
        "ecg_power": power,
    }


def extract_features(
    signal: pd.DataFrame,
) -> pd.Series:
    """
    Extrae todas las características de una señal ECG.
    """

    features = {}

    features.update(
        extract_statistical_features(signal)
    )

    features.update(
        extract_energy_features(signal)
    )

    return pd.Series(features)


def build_feature_dataset() -> pd.DataFrame:
    """
    Construye el dataset completo de características ECG.

    Returns
    -------
    pd.DataFrame
        Dataset con una fila por paciente.
    """

    feature_rows = []

    for patient_id in list_signals():

        signal = load_signal(patient_id)

        validate_signal(signal)

        features = extract_features(signal)

        features["id_paciente"] = patient_id

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)