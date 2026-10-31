# Preprocesamiento: NASA C-MAPSS FD001

## Partición por motor

- Semilla: 42.
- Motores para ajuste: 80.
- Motores para validación: 20.
- Intersección train/validación: vacía.
- El conjunto oficial de 100 motores de test permanece reservado.
- IDs congelados: `artifacts/processed/fd001/train_unit_ids.json` y
  `artifacts/processed/fd001/val_unit_ids.json`.

## Etiquetas y escalado

- Train y validación usan `RUL = max_cycle_del_motor - cycle`. Las etiquetas
  corresponden al ciclo final de cada ventana y son no negativas.
- Test usa exclusivamente el RUL oficial del último ciclo observado por motor.
  No se inventan etiquetas para los ciclos parciales de test.
- `StandardScaler` se ajustó con 16,561 filas de los
  motores de ajuste. No se usaron datos de validación ni test para ajustar el
  escalador.
- `unit_id` se conserva como metadato de las ventanas y no es una feature.
- Orden de features: `cycle`, `setting_1`, `setting_2`, `setting_3`, `sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`.
- Sensores constantes excluidos según la varianza de ajuste
  (umbral <= 1e-12):
  `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`.
- Sensores no constantes, aunque casi planos, se retienen para evitar descartes
  basados en todo el conjunto antes de experimentar.

## Ventanas

- Tamaño: 30 ciclos. Paso train/validación: 1.
- Train: 14,241 ventanas de
  80 motores.
- Validación: 3,490 ventanas de
  20 motores.
- Test: 100 ventanas, una por cada uno de los
  100 motores. Cada ventana termina en el último ciclo
  disponible de ese motor.
- Train/validación generan ventanas completas. Para trayectorias más cortas que
  la ventana, test repite a la izquierda la primera observación disponible y
  guarda la longitud real en `lengths`. No se usan observaciones futuras.
- Cada archivo `.npz` contiene `X` con forma
  `(muestras, ventana, features)`, `X_flat` para MLP, `y`, `unit_ids`,
  `end_cycles` y `lengths`.

## Artefactos

- `artifacts/processed/fd001/train.npz`
- `artifacts/processed/fd001/validation.npz`
- `artifacts/processed/fd001/test.npz`
- `artifacts/processed/fd001/scaler.joblib`
- `artifacts/processed/fd001/preprocessing_manifest.json`

Los hashes de los archivos fuente, parámetros del escalador, columnas y
partición quedan registrados en el manifiesto de preprocesamiento.
