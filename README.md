# Practica 2 - Repo 1: Modelado

Repositorio para la optimizacion de hiperparametros, calibracion e incertidumbre de modelos de deteccion de impago.

## Estructura del proyecto

```
practica2_repo1/
├── data/
│   ├── df_train_small.csv
│   ├── df_test_small.csv
│   └── variables_withExperts.xlsx
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── base_preprocessing.py
│   ├── filtering/
│   │   ├── __init__.py
│   │   └── base_filtering.py
│   └── models/
│       ├── __init__.py
│       └── model_wrapper.py
├── models/
│   ├── preprocessor.pkl (pre-fitteado)
│   ├── filter.pkl (pre-fitteado)
│   └── practica2_model.pkl (generado por el notebook)
├── practica2_notebook.ipynb (ejecutado con outputs)
├── pyproject.toml
├── .gitignore
└── README.md
```

## Requisitos

- Python >= 3.11
- uv (gestor de paquetes)

## Prerequisitos

**IMPORTANTE**: Antes de ejecutar el notebook, asegúrate de tener los artefactos pre-fitteados en la carpeta `models/`:

- `preprocessor.pkl` (objeto BasePreprocess ya fitteado en clase)
- `filter.pkl` (objeto BaseFiltering ya fitteado en clase)

⚠️ **Estos artefactos NO deben re-generarse en esta práctica**. El notebook los carga directamente con `joblib.load()` sin hacer `fit()`, según las instrucciones de la práctica.

## Instalacion

1. Instalar uv si no lo tienes:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clonar el repositorio y navegar al directorio:
```bash
cd practica2_repo1
```

3. Instalar dependencias:
```bash
uv sync
```

4. Verificar que existen los artefactos pre-fitteados:
```bash
ls -lh models/preprocessor.pkl models/filter.pkl
```

## Ejecucion

Para ejecutar el notebook de modelado:

```bash
uv run jupyter notebook practica2_notebook.ipynb
```

O usando JupyterLab:

```bash
uv run jupyter lab
```

## Contenido del notebook

El notebook `practica2_notebook.ipynb` implementa el pipeline completo de modelado:

1. **Optimizacion con Optuna**: Optimizacion de hiperparametros usando Log Loss como metrica objetivo, comparando modelos balanceados vs no balanceados.

2. **Calibracion**: Diagnostico y calibracion de probabilidades del modelo ganador.

3. **Medida de incertidumbre**: Implementacion de intervalos de prediccion para cuantificar la incertidumbre y politica de derivacion a agente humano.

4. **Persistencia**: Generacion del artefacto final `practica2_model.pkl` para su uso en la API.

## Artefactos

### Pre-existentes (prerequisitos):
- `models/preprocessor.pkl` (6.5 KB): Pipeline de preprocesamiento fitteado
- `models/filter.pkl` (2.7 MB): Pipeline de filtrado de features fitteado

### Generados por el notebook:
- `models/practica2_model.pkl` (4.0 MB): Modelo final que incluye:
  - Preprocessor
  - Filter
  - Modelo optimizado (LightGBM o XGBoost)
  - Calibrador (si aplicable)
  - MAPIE classifier (conformal prediction)
  - Metadata con métricas y configuración

## Notas

- El notebook debe ejecutarse completamente de principio a fin
- Los artefactos generados seran consumidos por el Repo 2 (API)
- Todos los experimentos usan semilla aleatoria fija para reproducibilidad
