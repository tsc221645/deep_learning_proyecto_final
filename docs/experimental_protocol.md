# Protocolo experimental: NASA C-MAPSS FD001

## Objetivo e hipótesis

Se evalúa si un modelo temporal puede mejorar la estimación de RUL respecto a
un MLP y si los cambios predictivos justifican su costo computacional bajo
supuestos explícitos. La hipótesis es una pregunta de investigación, no un
resultado prometido.

## Datos y particiones

- Se usa FD001: train oficial para desarrollo, test oficial reservado.
- Los 100 motores de train se dividen por `unit_id` con semilla `42`: 80 para
  ajuste y 20 para validación. Los IDs congelados están en
  `artifacts/processed/fd001/train_unit_ids.json` y `val_unit_ids.json`.
- MLP, LSTM y baseline usan los mismos motores, features, ventana, escalador y
  etiquetas sin truncar.
- El `StandardScaler` ya se ajustó exclusivamente con registros de los motores
  de ajuste. No se vuelve a ajustar al evaluar ni se usa test para tomar
  decisiones.
- Train/validación se reportan sobre todas las ventanas completas. Las ventanas
  se solapan y **no** son observaciones independientes; cualquier intervalo de
  confianza futuro debe remuestrear motores, no ventanas.
- Para validación se usa `RUL = último ciclo completo del motor - ciclo final
  de ventana`. Para test, una predicción por motor en el último ciclo observado
  se comparará con `RUL_FD001.txt` solo después de congelar las decisiones del
  modelo.
- No se calculan etiquetas por ciclo parcial de test, no se predice sobre la
  continuación real y no se reportan ciclos como horas o fechas.

## Métricas

MAE en ciclos es la métrica primaria. Se reportan también RMSE, sesgo firmado
(`predicción - real`), percentiles 50 y 90 del error absoluto y NASA PHM08 Score.
Valores menores de MAE, RMSE y Score son mejores. Un sesgo positivo significa
sobreestimación de la vida restante.

Para cada ejemplo `i`, se define `d_i = RUL_predicho_i - RUL_real_i`. La
convención PHM08 asimétrica usada es:

```text
s_i = exp(-d_i / 13) - 1, si d_i < 0  (subestimación)
s_i = exp( d_i / 10) - 1, si d_i >= 0 (sobreestimación)
Score_sum = sum(s_i)
Score_mean = mean(s_i)
```

La predicción perfecta tiene score cero. Con igual magnitud de error, esta
convención penaliza más la sobreestimación. Se guardan suma y promedio, porque
la suma depende de la cantidad de motores/ventanas. La implementación rechaza
datos inválidos y errores que desbordan el rango numérico en lugar de producir
un score engañoso.

Referencia de datos y benchmark: Saxena, A., Goebel, K., Simon, D., y Eklund,
N. (2008). “Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation”. PHM08, DOI:
https://doi.org/10.1109/PHM.2008.4711414.

Referencia de métricas de pronóstico: Saxena, A., Celaya, J., Balaban, E.,
Goebel, K., Saha, B., Saha, S., y Schwabacher, M. (2008). “Metrics for
Evaluating Performance of Prognostic Techniques”. PHM08, DOI:
https://doi.org/10.1109/PHM.2008.4711436.

## Línea base

`predictguard.baseline` predice en cada ventana la mediana de las etiquetas de
las ventanas de ajuste, sin aprender de validación. Informa métricas de
entrenamiento como diagnóstico y métricas de validación para comparación. El
script solo abre `train.npz`, `validation.npz` y el manifiesto de
preprocesamiento; no abre `test.npz` ni `RUL_FD001.txt`.

Las métricas de validación de la línea base son promedios por ventana. Las
ventanas solapadas se correlacionan; no se deben interpretar como si hubiera
3,490 motores independientes.

## Entrenamiento comparable

Configuración inicial congelada en `configs/config.yaml`:

- Semillas: `42`, `43` y `44`. Se fijarán Python, NumPy, PyTorch y CUDA cuando
  corresponda y se registrarán versiones, dispositivo, semilla y configuración
  por ejecución. Se reportarán las variaciones entre semillas; no se promete
  determinismo bit a bit entre hardware/versiones.
- Máximo: 100 épocas; batch: 64 ventanas.
- Optimizador: Adam, `learning_rate=0.001`.
- Pérdida: Huber en ciclos, `delta=1.0`.
- Early stopping: evaluar MAE de validación por época, paciencia 10, cambio
  mínimo 0 ciclos, y restaurar el checkpoint con menor MAE. Un empate conserva
  el primer checkpoint.
- No se usa PHM08 Score ni ninguna métrica de test para escoger checkpoints.
- La comparación mantiene partición, ventana, escalado, features y presupuesto
  de entrenamiento equivalentes entre MLP y LSTM.

## Registro de experimentos y costos

`artifacts/experiments/results.csv` es el registro tabular común. Cada
ejecución futura debe conservar su ID y semilla, arquitectura/features/ventana,
learning rate, épocas, métricas train/validación, duración, latencias p50/p95 y
estado. No sobrescribir ejecuciones con otro ID. La línea base reemplaza
únicamente su propia fila al repetirse.

Las latencias se medirán en inferencia de batch uno sobre la misma ventana,
hardware y política de calentamiento; se registran dispositivo, número de
repeticiones y memoria/tamaño cuando estén disponibles. La línea base no
entrena ni calcula una latencia de modelo y deja esos campos vacíos.

## Test oficial

No ejecutar selección, calibración ni comparación de modelos en test durante
las fases de desarrollo. Después de cerrar arquitectura, hiperparámetros,
features, transformador, ventana y código de métricas, evaluar una única vez
los 100 últimos ciclos observados con sus 100 etiquetas oficiales. Registrar
esa evaluación por separado y reportar cualquier re-evaluación.
