"""
Endpoints de inferencia.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

import pandas as pd

from fastapi import APIRouter

from src.models.load_model import load_model
from src.models.predict import predict
from src.api.schemas import PatientFeatures


router = APIRouter()

model = load_model()


@router.post("/predict")
def predict_patient(data: PatientFeatures):

    X = pd.DataFrame(
        [data.model_dump()]
    )

    prediction, probability = predict(
        model,
        X,
    )

    return {
        "prediction": int(prediction[0]),
        "probability": float(
            probability[0][1]
        ),
    }