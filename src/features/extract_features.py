"""
Extracción de características de señales ECG.

Responsabilidades:
- Calcular características estadísticas.
- Calcular características de energía.
- Generar un vector de características por señal.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import numpy as np
import pandas as pd


def extract_statistical_features(
    signal: pd.DataFrame,
) -> dict:
    """
    Extrae características estadísticas de una señal ECG.

    Parameters
    ----------
    signal : pd.DataFrame
        Señal ECG.

    Returns
    -------
    dict
        Características estadísticas.
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

    Parameters
    ----------
    signal : pd.DataFrame
        Señal ECG.

    Returns
    -------
    dict
        Características de energía.
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

    Parameters
    ----------
    signal : pd.DataFrame
        Señal ECG.

    Returns
    -------
    pd.Series
        Vector de características.
    """

    features = {}

    features.update(
        extract_statistical_features(signal)
    )

    features.update(
        extract_energy_features(signal)
    )

    return pd.Series(features)