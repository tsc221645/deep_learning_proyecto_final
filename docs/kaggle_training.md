# Fases 7 a 10 en Kaggle

Las fases 7 a 10 combinan **código y ejecuciones**. Este repositorio incluye el
entrenador LSTM, el registro de candidatos, la evaluación final y el simulador
económico. Los resultados siguen pendientes hasta ejecutar las corridas.

## 1. Preparar el repositorio

Publica estos cambios en GitHub antes de abrir Kaggle. El Notebook debe usar el
commit que contiene `train_lstm.py`, `evaluate_final.py` y `policy.py`. Registra
el hash del commit en el informe. No subas `data/raw/*.txt` ni `*.npz` a Git.

## 2. Crear Notebook

En Kaggle, abre **Code > New Notebook**. En **Settings > Session options** activa
**Internet** y selecciona **Accelerator > GPU** si está disponible. La GPU es
conveniente para la LSTM, aunque FD001 también puede procesarse en CPU. Guarda
una versión con **Save Version > Save & Run All**; los archivos creados en
`/kaggle/working` aparecerán en **Output**. La cuota y disponibilidad de GPU
pueden variar.

El Notebook puede ser privado. No guardes claves ni tokens en celdas.

## 3. Preparar datos y entorno

En una celda de código:

```python
!git clone https://github.com/tsc221645/deep_learning_proyecto_final.git /kaggle/working/predictguard
%cd /kaggle/working/predictguard
%env PYTHONPATH=/kaggle/working/predictguard/src
!git rev-parse HEAD
!python -c "import torch,numpy,pandas,sklearn,yaml,matplotlib; print('CUDA:',torch.cuda.is_available(),'PyTorch:',torch.__version__)"
```

El `pyproject.toml` fija Python 3.13 y versiones locales. Kaggle puede usar
otras versiones: por eso se usa `PYTHONPATH=src` en vez de instalar el proyecto
con todas sus versiones fijadas. Si falta alguna librería, instálala en el
Notebook y registra su versión.

El repositorio versiona el manifiesto de datos pero no los archivos originales.
Para descargar el ZIP oficial de NASA en el entorno de Kaggle:

```python
from pathlib import Path
Path("data/raw/data_manifest.json").unlink(missing_ok=True)
!python -m predictguard.data.download --config configs/config.yaml
!python -m predictguard.data.validate --config configs/config.yaml
!python -m predictguard.data.preprocess --config configs/config.yaml
```

Si NASA no responde, adjunta a **Input** un dataset de Kaggle que contenga
`train_FD001.txt`, `test_FD001.txt` y `RUL_FD001.txt` (idealmente los archivos
originales que ya validaste). Copia esos archivos desde `/kaggle/input/...`
a `data/raw/`, ejecuta el validador y verifica que los SHA-256 coincidan con
`artifacts/processed/fd001/preprocessing_manifest.json` del experimento MLP
original. En ese caso omite la descarga. Elige un dataset privado si subes tu
copia de los archivos. **No uses el test oficial para seleccionar modelos.**

## 4. Entrenar y comparar en validación

Para medir MLP y LSTM en el mismo hardware y sobre el mismo preprocesamiento:

```python
!python -m predictguard.train --config configs/config.yaml
!python -m predictguard.train_lstm --config configs/config.yaml --run-id lstm
```

La LSTM inicial usa la misma ventana, features, partición y pérdida que el MLP.
El entrenador ejecuta las semillas 42, 43 y 44, early stopping por MAE de
validación y guarda el mejor checkpoint de cada una. Los resultados van a
`artifacts/metrics/` y `artifacts/experiments/results.csv`.

Para una iteración controlada de la fase 8, cambia **una familia** de parámetros
por corrida. Por ejemplo:

```python
!python -m predictguard.train_lstm --config configs/config.yaml --run-id lstm_h32 --hidden-size 32
!python -m predictguard.train_lstm --config configs/config.yaml --run-id lstm_lr3e4 --learning-rate 0.0003
```

Revisa solo validación. Selecciona un `run-id` y anota su MAE, dispersión entre
semillas, sesgo y tiempo. La meta de 10 % es aspiracional. Si no se cumple,
reporta el resultado observado. Los cambios de ventana o features requieren
otro preprocesamiento y otro MLP comparable; no mezcles resultados de pipelines
distintos en la tabla principal.

## 5. Política de costos y test final

Pon el `run-id` elegido en `configs/business.yaml` bajo `lstm_run`. Los costos
son ejemplos hipotéticos en quetzales; ajústalos y explica el motivo en el
informe. Calcula costos con los motores completos de validación:

```python
!python -m predictguard.policy --config configs/business.yaml
```

Esto genera `artifacts/business/cost_curve.csv`, `sensitivity.csv`,
`engine_outcomes.csv` y `summary.json`. El costo mínimo y los umbrales se
calculan sobre los mismos motores de validación; descríbelos como una estimación
exploratoria, no como ahorro real ni como evaluación económica independiente.

Después de cerrar las decisiones, congela los hashes de modelos y
preprocesamiento. Reemplaza `lstm` por el `run-id` elegido:

```python
!python -m predictguard.evaluate_final freeze --lstm-run lstm
!python -m predictguard.explain
!python -m predictguard.evaluate_final evaluate
```

`evaluate` lee `test.npz` únicamente tras el congelamiento. Produce una
predicción por cada uno de los 100 motores de test en su último ciclo observado,
métricas y bootstrap por motor. No repitas la evaluación para ajustar parámetros.
`explain` permuta trayectorias de features en validación; mide sensibilidad
predictiva, no causalidad física.

## 6. Descargar resultados

Elige **Save Version > Save & Run All**. En **Output**, descarga como mínimo:

- `artifacts/models/mlp_best.pt` y `artifacts/models/<run-id>_best.pt`;
- `artifacts/metrics/*training_summary.json`, `*validation_predictions.csv`,
  `test_predictions.csv` y `test_metrics.json`;
- `artifacts/experiments/results.csv` y `selected.json`;
- `artifacts/business/` completo;
- `artifacts/processed/fd001/preprocessing_manifest.json` y `scaler.joblib`;
- log del Notebook, versiones de librerías y hash del commit.

Conserva también una copia del Notebook versionado. Si el proceso se corta,
puedes reutilizar salidas de una versión como **Input** de otro Notebook, pero
debes mantener el mismo commit y documentar las versiones de los artefactos.

## Límites metodológicos

- `RUL_FD001.txt` solo etiqueta el último punto observado de cada motor de
  test; la simulación secuencial se hace sobre validación completa.
- Las ventanas solapadas no son observaciones independientes. El intervalo de
  test remuestrea motores.
- La política supone que una intervención hecha con al menos `lead_time_cycles`
  de margen evita la falla. Ese supuesto no proviene de NASA.
- El escenario usa ciclos simulados, nunca horas o fechas de mantenimiento.
