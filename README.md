# Pipeline MLOps de Salva Health

Pipeline de Machine Learning Operations de extremo a extremo para la predicción de riesgo clínico usando información del paciente y características derivadas de ECG.

Este proyecto implementa un ciclo de vida completo de ML:

* Ingesta de datos desde almacenamiento en la nube
* Validación y preprocesamiento de datos
* Ingeniería de características de ECG
* Entrenamiento reproducible de modelos
* Seguimiento de experimentos con MLflow
* Selección automática de modelo
* Servicio de inferencia REST API
* Containerización con Docker
* Integración y entrega continua
* Despliegue en Azure Container Apps

El objetivo es demostrar un flujo de trabajo MLOps orientado a producción donde un modelo de machine learning pueda ser entrenado, versionado, desplegado y consumido de manera confiable.

---

# 1. Arquitectura del sistema

```text
                         Azure Storage
                              |
                              |
                       Ingesta de Datos
                              |
                              v
                  Capa de Validación de Datos
                              |
                              v
                       Preprocesamiento
                              |
                              v
                Extracción de Características ECG
                              |
                              v
                    Pipeline de Entrenamiento
                              |
              +---------------+---------------+
              |                               |
              v                               v
      Seguimiento de Experimentos      Evaluación del Modelo
             MLflow                            |
                                              |
                                              v
                                  Selección del Mejor Modelo
                                              |
                                              v
                                  artefacto model.joblib
                                              |
                                              v
                                    Servicio FastAPI
                                              |
                                              v
                                  Azure Container Apps
                                              |
                                              v
                                Endpoint de Predicción
```

---

# 2. Stack tecnológico

## Machine Learning

* Python
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* MLflow

## Ingeniería de datos

* Azure Blob Storage
* Datasets en formato Parquet
* Pipeline de procesamiento tipo ETL

## API y despliegue

* FastAPI
* Uvicorn
* Docker
* Azure Container Registry
* Azure Container Apps

## Automatización

* GitHub Actions
* Pytest

---

# 3. Estructura del repositorio

```text
salva-health-mlops/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   │
│   ├── data/
│   │   ├── ingest.py
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   └── save.py
│   │
│   ├── features/
│   │   └── extract_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── register.py
│   │   └── select_best.py
│   │
│   ├── pipeline/
│   │   └── run_pipeline.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── tests/
│
├── artifacts/
│   └── models/
│       └── best_model.joblib
│
├── mlruns/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 4. Pipeline de datos

El pipeline parte de información clínica y señales de ECG en bruto.

El proceso incluye:

## Ingesta de datos

Los datos se consumen desde Azure Blob Storage.

El pipeline descarga los datasets requeridos y los prepara para su procesamiento.

---

## Validación de datos

Validaciones implementadas:

* Validación de columnas requeridas
* Detección de valores faltantes
* Verificaciones de consistencia de datos
* Normalización del formato de fechas

Ejemplo:

```text
Registros iniciales:
515

Pacientes únicos:
500

IDs de pacientes duplicados detectados:
15
```

---

## Preprocesamiento

La etapa de preprocesamiento:

* Limpia valores inconsistentes
* Estandariza formatos
* Maneja información faltante
* Genera datasets limpios

Salida:

```text
data/processed/
```

---

# 5. Ingeniería de características

Las señales de ECG se transforman en características numéricas.

Características generadas:

## Características estadísticas

* Media
* Mediana
* Desviación estándar
* Varianza
* Mínimo
* Máximo
* Rango
* Cuartiles
* Rango intercuartílico

## Características de energía

* Energía de la señal
* RMS
* Potencia

Estas características se usan como entradas del modelo.

---

# 6. Entrenamiento del modelo y seguimiento de experimentos

El proceso completo de entrenamiento se puede reproducir con un solo comando:

```bash
python -m src.pipeline.run_pipeline
```

El pipeline realiza:

1. Carga de datos
2. Validación
3. Preprocesamiento
4. Generación de características
5. División train/test
6. Entrenamiento del modelo
7. Registro del experimento
8. Comparación de modelos
9. Selección del mejor modelo
10. Generación del artefacto del modelo

---

# 7. Comparación y selección de modelos

Se evalúan múltiples modelos y se rastrean con MLflow.

Cada experimento almacena:

* Parámetros del modelo
* Métricas
* Información de entrenamiento
* Artefactos generados

El modelo seleccionado se promueve como:

```text
artifacts/models/best_model.joblib
```

Este archivo representa el modelo autorizado para despliegue.

---

# 8. Decisión sobre el escalado de características

Durante el desarrollo del modelo se evaluó la normalización de características.

El modelo final está basado en árboles, por lo tanto no se aplicó escalado.

Los algoritmos de árboles dividen los datos usando umbrales y generalmente son invariantes a la magnitud de las características.

Se mantiene la misma representación durante:

* Entrenamiento
* Validación
* Inferencia en producción

Por lo tanto, la API recibe los valores clínicos originales.

Ejemplo:

```json
{
  "edad_paciente":58,
  "peso_kg":74.5,
  "ecg_energy":84.6
}
```

---

# 9. Estrategia de versionamiento

## Código fuente

El desarrollo se realizó usando ramas de Git:

```text
main

 |
 └── feature/<nombre-desarrollador>
```

Los cambios se integran mediante Pull Requests.

---

## Versionamiento de datasets

Los datasets se organizan por ciclo de vida:

```text
data/

raw/
    archivos originales

processed/
    datasets validados
    características generadas
```

Los artefactos procesados se almacenan en formatos reproducibles:

* Archivos Parquet
* Salidas explícitas del pipeline

Para sistemas de producción más grandes, se recomendaría DVC.

---

## Versionamiento de modelos

Los modelos se rastrean mediante:

* Experimentos de MLflow
* Metadatos de las ejecuciones (runs)
* Artefactos generados

Modelo de producción:

```text
artifacts/models/best_model.joblib
```

---

# 10. Servicio de inferencia API

El modelo entrenado se expone a través de FastAPI.

## Health Check

Endpoint:

```
GET /health
```

Respuesta:

```json
{
  "status": "healthy",
  "service": "salva-health-api"
}
```

---

## Endpoint de predicción

Endpoint:

```
POST /predict
```

Ejemplo de solicitud:

```json
{
  "edad_paciente": 58,
  "sexo": 1,
  "peso_kg": 74.5,
  "altura_cm": 175.0,
  "frecuencia_cardiaca_media_bpm": 72,
  "imc": 24.3,
  "ecg_mean": 0.0015,
  "ecg_median": 0.0012,
  "ecg_std": 0.1840,
  "ecg_variance": 0.0339,
  "ecg_minimum": -0.95,
  "ecg_maximum": 1.03,
  "ecg_range": 1.98,
  "ecg_q25": -0.06,
  "ecg_q75": 0.07,
  "ecg_iqr": 0.13,
  "ecg_energy": 84.6,
  "ecg_rms": 0.1840,
  "ecg_power": 0.0339
}
```

Respuesta:

```json
{
  "prediction":1,
  "probability":0.967
}
```

---

## Manejo de errores

La API valida las solicitudes entrantes usando Pydantic.

Las solicitudes inválidas retornan:

```text
HTTP 422 Unprocessable Entity
```

con información detallada de la validación.

---

# 11. Ejecución local

## Clonar el repositorio

```bash
git clone https://github.com//salva-health-mlops.git
cd salva-health-mlops
```

---

## Crear entorno

```bash
python -m venv .venv
```

Activar:

Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar el pipeline del proyecto

```bash
python -m src.pipeline.run_pipeline
```

---

## Visualizar experimentos con MLflow

Una vez finalizado el entrenamiento, inicia la interfaz de MLflow con:

```bash
mlflow ui
```

Por defecto, la interfaz estará disponible en:

```text
http://localhost:5000
```

Allí podrás consultar los experimentos, métricas, parámetros y artefactos registrados durante el entrenamiento.

---


## Ejecutar la API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentación Swagger:

```
http://localhost:8000/docs
```

---

# 12. Despliegue con Docker

Construir la imagen:

```bash
docker build -t salva-health-api .
```

Ejecutar el contenedor:

```bash
docker run -p 8000:8000 salva-health-api
```

Probar:

```bash
curl http://localhost:8000/health
```

---

# 13. Pipeline de CI/CD

## Integración continua

GitHub Actions realiza automáticamente:

* Instalación de dependencias
* Pruebas automatizadas
* Validación del pipeline

Disparado por:

* Push
* Pull Request

---

## Entrega continua

La extensión implementada realiza:

```text
GitHub Actions

      |

Docker Build

      |

Azure Container Registry

      |

Azure Container Apps
```

La imagen Docker se publica como:

```text
salvahealthacr.azurecr.io/salva-health-api:latest
```

---

# 14. Despliegue en Azure

La aplicación corre en Azure Container Apps.

Configuración:

```text
CPU:
0.5 vCPU

Memoria:
1 GiB

Puerto del contenedor:
8000

Ingress:
Externo
```

Endpoint de producción:

```text
https://salva-health-api.calmmeadow-06a04c7b.eastus2.azurecontainerapps.io
```

---

# 15. Propuesta de monitoreo

Monitoreo de producción recomendado:

## Métricas técnicas

* Latencia de la API
* Tasa de error HTTP
* Utilización de CPU
* Consumo de memoria
* Disponibilidad del contenedor

## Métricas de ML

* Drift en la distribución de características
* Cambios en la distribución de predicciones
* Degradación del rendimiento del modelo
* Tasa de falsos positivos / falsos negativos

Acciones:

```text
Drift detectado

        |

Revisión del dataset

        |

Pipeline de reentrenamiento del modelo
```

---

# 16. Mejoras futuras

Mejoras potenciales:

* Agregar DVC para versionamiento de datasets a nivel empresarial
* Agregar promoción automatizada del registro de modelos
* Implementar monitoreo con Azure Application Insights
* Agregar monitoreo del rendimiento del modelo con etiquetas de producción
* Optimización de hiperparámetros usando Optuna
* Implementar flujo de reentrenamiento automatizado

---

# 17. Uso de herramientas de IA

Durante el desarrollo, se usaron asistentes de IA como herramientas de soporte técnico para:

* Revisión de arquitectura
* Mejora de documentación
* Asistencia en debugging
* Guía de despliegue en la nube
* Validación de buenas prácticas

Todas las decisiones de implementación fueron verificadas mediante ejecución, pruebas y validación en el entorno desplegado.

---

# 18. Estado final

Implementado:

- Ingesta de datos en la nube
- Pipeline de validación de datos
- Ingeniería de características de ECG
- Flujo de trabajo de entrenamiento reproducible
- Seguimiento de experimentos con MLflow
- Mecanismo de selección de modelo
- API de inferencia con FastAPI
- Containerización con Docker
- CI/CD con GitHub Actions
- Azure Container Registry
- Despliegue en Azure Container Apps
- Endpoint de predicción en producción

El proyecto demuestra un ciclo de vida MLOps completo, desde la ingesta de datos hasta la inferencia de modelos en la nube.
