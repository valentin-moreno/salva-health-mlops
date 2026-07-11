"""
Aplicación principal.

Autor: Valentin Moreno Vásquez
Proyecto: Salva Health MLOps
"""

from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="Salva Health API",
    version="1.0.0",
)

app.include_router(router)