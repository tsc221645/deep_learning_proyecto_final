# Hallazgos EDA: NASA C-MAPSS FD001

## Alcance

Se analizaron los archivos originales en train y test. No se modificaron ni
eliminaron registros. Los datos son simulados; cada fila representa un ciclo,
no una fecha u hora real. FD001 contiene una condición operativa y un modo de
degradación (HPC), según el `readme.txt` y el catálogo del conjunto.

## Integridad y duración de trayectorias

- Train: 100 motores, 20,631 filas.
  Duración mediana 199 ciclos, rango
  128-362.
- Test: 100 motores, 13,096 filas.
  Duración mediana 133.5 ciclos, rango
  31-303.
- No se detectaron faltantes ni infinitos en train/test.
- **train**: 0 pares (motor, ciclo) duplicados; 0 saltos no crecientes; 0 saltos no consecutivos; ciclos estrictamente crecientes en orden de archivo: True.
- **test**: 0 pares (motor, ciclo) duplicados; 0 saltos no crecientes; 0 saltos no consecutivos; ciclos estrictamente crecientes en orden de archivo: True.

## Sensores

- Constantes en train: `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`.
- Candidatos casi constantes: `sensor_6`.
  El criterio exploratorio es que un único valor represente al menos
  98% de las observaciones; no es una decisión
  automática para excluir variables.
- Pares con correlación de Pearson absoluta >= 0.95:
  sensor_9 y sensor_14 (0.963).
- Sensores que se muestran en trayectorias representativas:
  `sensor_11`, `sensor_12`, `sensor_4`, `sensor_7`.
  La selección es visual y no constituye una selección final de features.
- Sensores con mayor cambio entre los tercios inicial y final de la vida:
  `sensor_17`, `sensor_11`, `sensor_4`, `sensor_12`, `sensor_7`, `sensor_15`.
- Motores representativos seleccionados por cuantiles de duración:
  93, 80, 79, 47, 56.

## RUL y ventanas

- Se calculó RUL de entrenamiento por fila como
  `máximo ciclo observado del motor - ciclo`.
- Test se describe exclusivamente con los 100 RUL oficiales del último ciclo
  observado por motor. No se asignaron etiquetas a los ciclos parciales de test.
- Tamaño de ventana usado únicamente para el conteo diagnóstico:
  30 ciclos. Train produce
  17,731 ventanas completas; cada motor produce
  entre 99 y
  333. De los extremos de ventana,
  3,100 (17.5%) tienen RUL
  <= 30 ciclos. Esto evidencia desbalance de etiquetas según su cercanía al fallo.
- Las etiquetas de test son más escasas que las de train y resumen un único
  ciclo terminal por motor. Ambas distribuciones se grafican por separado y
  no se deben comparar como si fueran muestras equivalentes.

## Comparación descriptiva train/test

Se compararon sensores sin usar el RUL de test. Las medias de test con mayor
desplazamiento, expresadas en desviaciones estándar estimadas solo en train,
son: `sensor_16: 2.00`, `sensor_5: -0.67`, `sensor_11: -0.47`, `sensor_4: -0.47`, `sensor_12: 0.45`. Esta visualización es descriptiva y no decide features,
normalización ni ajustes del modelo.

## Valores atípicos

El criterio de cribado fue el rango 1.5 IQR aplicado a cada sensor de train.
Los sensores con mayor proporción de observaciones marcadas fueron:
- `sensor_9`: 1686 registros (8.17%) fuera de 1.5 IQR.
- `sensor_14`: 1543 registros (7.48%) fuera de 1.5 IQR.
- `sensor_6`: 406 registros (1.97%) fuera de 1.5 IQR.
- `sensor_8`: 320 registros (1.55%) fuera de 1.5 IQR.
- `sensor_11`: 167 registros (0.81%) fuera de 1.5 IQR.

Estos valores se conservan. Antes de filtrarlos se deben inspeccionar sus
trayectorias: pueden ser señales válidas de degradación y su eliminación
podría quitar información predictiva.

## Decisiones para la siguiente fase

1. Conservar los archivos crudos intactos y no corregir valores sin una anomalía
   verificable.
2. Evaluar los sensores constantes y casi constantes usando **solo el
   subconjunto de ajuste** en la fase de preprocesamiento. La lista anterior es
   exploratoria, no una decisión entrenada con validación/test.
3. Ajustar cualquier escalador solo con motores de ajuste.
4. Mantener el test oficial reservado. Sus etiquetas corresponden únicamente al
   último ciclo observado de cada motor.
5. Tratar las diferencias entre motores como trayectorias separadas y nunca
   presentar ciclos como tiempo calendario.

## Archivos generados

- `artifacts/figures/eda_engine_durations.png`
- `artifacts/figures/eda_sensor_trajectories.png`
- `artifacts/figures/eda_sensor_variance.png`
- `artifacts/figures/eda_sensor_correlations.png`
- `artifacts/figures/eda_train_rul_distribution.png`
- `artifacts/figures/eda_train_test_sensor_distributions.png`
- `artifacts/figures/eda_lifecycle_segments.png`
- Tablas CSV en `artifacts/metrics/`.
