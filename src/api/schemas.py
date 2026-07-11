"""
Esquemas de entrada para la API.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

from pydantic import BaseModel


class PatientFeatures(BaseModel):

    edad_paciente: float
    sexo: int
    peso_kg: float
    altura_cm: float
    frecuencia_cardiaca_media_bpm: float
    imc: float
    ecg_mean: float
    ecg_median: float
    ecg_std: float
    ecg_variance: float
    ecg_minimum: float
    ecg_maximum: float
    ecg_range: float
    ecg_q25: float
    ecg_q75: float
    ecg_iqr: float
    ecg_energy: float
    ecg_rms: float
    ecg_power: float
