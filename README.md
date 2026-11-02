# PredictGuard: Predicción de vida útil restante de motores aeronáuticos mediante Deep Learning y optimización económica del mantenimiento

## Propuesta
Desarrollar una plataforma web de mantenimiento predictivo que utilice modelos de Deep Learning para estimar la vida útil restante (Remaining Useful Life, RUL) de motores aeronáuticos a partir de datos históricos de sensores. El sistema empleará arquitecturas de redes neuronales, como MLP y LSTM, para analizar patrones de degradación y predecir cuántos ciclos de funcionamiento le quedan a un motor antes de presentar una falla. Además, incorporará un dashboard interactivo para visualizar el estado de los motores, comparar predicciones y simular decisiones de mantenimiento, cuantificando el impacto económico de las mejoras en la precisión de los modelos.

## Justificación
El mantenimiento no planificado de maquinaria puede generar costos elevados debido a interrupciones operativas, reparaciones y reemplazos de componentes. Los métodos tradicionales de mantenimiento preventivo, basados en intervalos fijos, pueden provocar intervenciones prematuras o no anticipar determinadas fallas. Mediante Deep Learning es posible analizar patrones temporales complejos en las mediciones de sensores y estimar el desgaste progresivo de los equipos. Este proyecto busca demostrar cómo una mejora en la precisión de las predicciones puede traducirse en decisiones de mantenimiento más eficientes, reduciendo costos por fallas inesperadas y aprovechando mejor la vida útil de los componentes. Asimismo, permitirá evaluar si el beneficio económico de utilizar modelos más complejos justifica sus costos computacionales.

## Fuente de datos
NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation): conjunto de datos público desarrollado por la NASA que contiene registros simulados de degradación de motores turbofán, incluyendo 21 variables de sensores, condiciones operativas y ciclos de funcionamiento. Se utilizará principalmente el subconjunto FD001, que contiene historiales de 100 motores para entrenamiento y 100 para evaluación, junto con valores de referencia de vida útil restante. Estos datos permiten entrenar y evaluar modelos de regresión para predecir el RUL.

Fuente oficial: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## Problema de investigación
¿En qué medida una mejora en la precisión de la predicción de vida útil restante de motores aeronáuticos simulados permite reducir el costo esperado de las decisiones de mantenimiento?

## Objetivo general
Desarrollar e implementar un sistema inteligente basado en redes neuronales que estime la vida útil restante de motores turbofán a partir de datos históricos de sensores y evalúe el impacto económico de diferentes arquitecturas de Deep Learning sobre las decisiones de mantenimiento.

## Objetivos específicos medibles
+ **Procesar y analizar los datos**: preparar el 100 % de los registros del subconjunto FD001 de NASA C-MAPSS, incluyendo normalización, análisis exploratorio y generación de secuencias temporales para el entrenamiento y evaluación de los modelos.

+ **Implementar y comparar modelos**: entrenar al menos dos arquitecturas de Deep Learning (MLP y LSTM), evaluando su desempeño mediante MAE, RMSE y NASA Score sobre conjuntos de evaluación independientes.

+ **Optimizar las predicciones**: buscar una reducción de al menos el 10 % en el MAE del modelo LSTM respecto al MLP, mediante ajustes de arquitectura, hiperparámetros y estrategias de entrenamiento.

+ **Cuantificar el impacto económic**o: desarrollar un modelo de costos que considere mantenimiento preventivo, fallas no anticipadas y reemplazos prematuros, con el objetivo de evaluar si la mejora en las predicciones permite reducir al menos un 10 % los costos simulados frente al modelo base, bajo los mismos supuestos operativos.

+ **Desarrollar una aplicación funcional**: implementar y desplegar una plataforma web que permita seleccionar motores, visualizar sus mediciones, estimar su vida útil restante y comparar los resultados de ambos modelos, procurando un tiempo de respuesta inferior a 2 segundos por predicción.

+ **Evaluar la viabilidad del sistema**: medir el tiempo de entrenamiento, la latencia de inferencia y el costo computacional de cada modelo, determinando si la mejora técnica justifica los recursos adicionales requeridos.

## Hipótesis
Una arquitectura temporal LSTM puede reducir el error de estimación de RUL respecto a una red neuronal densa, generando una disminución del costo esperado de mantenimiento bajo una política de intervención definida.

## Resultados esperados
Una plataforma web funcional, con modelos entrenados y desplegados, que permita analizar motores, estimar su vida útil restante y cuantificar el beneficio de utilizar una arquitectura de mayor precisión, considerando también sus costos operativos.

## Entorno local

Se utiliza el entorno Conda `dm1` con Python 3.13.2. En Anaconda Prompt:

```powershell
conda activate dm1
.\scripts\setup.ps1
.\scripts\test.ps1
```

El proyecto usa el wheel CUDA 12.6 de PyTorch. La GPU local se detecta correctamente. FD001 es suficientemente pequeño para trabajar y entrenar aquí; Kaggle no hace falta por ahora.

Los archivos locales de C-MAPSS se mantienen fuera de Git. La configuración inicial está en `configs/config.yaml`.

## Fase 2: datos NASA C-MAPSS FD001

Descarga y valida los archivos originales en `data/raw/`. Si ya están descargados, la descarga no los sobrescribe.

```powershell
conda activate dm1
.\scripts\download.ps1
.\scripts\validate.ps1
```

La validación genera `data/raw/data_manifest.json` con los hashes SHA-256, tamaños, conteos y procedencia. Los datos se mantienen fuera de Git; el manifiesto sí se versiona.

## Fase 3: análisis exploratorio

Genera las tablas, figuras y el reporte de EDA sin modificar los datos crudos:

```powershell
conda activate dm1
.\scripts\eda.ps1
```

El notebook `notebooks/01_eda.ipynb` permite revisar y volver a generar los resultados. El setup instala las dependencias de Jupyter en `dm1`.

## Fase 4: particiones y preprocesamiento

Separa motores completos de train para validación, ajusta el escalador solo con motores de ajuste, genera ventanas y conserva el test oficial reservado:

```powershell
conda activate dm1
.\scripts\preprocess.ps1
```

Los tensores y el escalador quedan en `artifacts/processed/fd001/`. Los IDs, hashes y la configuración quedan en el manifiesto. El reporte está en `docs/preprocessing_report.md`.

## Fase 5: protocolo experimental y baseline

```powershell
conda activate dm1
.\scripts\baseline.ps1
```

El protocolo queda en `docs/experimental_protocol.md`. El baseline usa la mediana de las etiquetas de entrenamiento y reporta métricas en validación. No carga ni evalúa el test oficial.

## Fase 6: entrenamiento del MLP

```powershell
conda activate dm1
.\scripts\train_mlp.ps1
```

Entrena tres semillas y selecciona el checkpoint con menor MAE de validación. Guarda historial, predicciones, gráficas y resultados en `artifacts/`. No carga el test oficial.

## Fases 7–10: LSTM, comparación, test y valor

Las corridas de Kaggle y sus artefactos ya se integraron. La
[guía de Kaggle](docs/kaggle_training.md) describe cómo reproducirlas. El test
oficial se evaluó después de congelar la selección en
`artifacts/experiments/selected.json`. Consulta el
[resumen de resultados de las fases 7 a 10](docs/results_phases_7_10.md).
