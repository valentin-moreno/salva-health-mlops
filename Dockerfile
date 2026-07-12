FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc y forzar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Crear usuario seguro para producción
RUN useradd -u 8888 appuser && chown -R appuser /app

# Instalar dependencias del sistema si XGBoost o Scikit-learn lo requieren (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código y los artefactos necesarios
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Cambiar al usuario no privilegiado
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
