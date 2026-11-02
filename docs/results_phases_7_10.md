# Resultados de las fases 7 a 10 — FD001

Estos resultados se generaron en Kaggle con el mismo preprocesamiento y la
partición congelada de 80 motores de ajuste y 20 de validación. Los artefactos
originales están en `artifacts/`; `selected.json` registra los hashes de los
modelos y el manifiesto usados en la evaluación final.

## Fases 7 y 8: entrenamiento y selección en validación

| Modelo / corrida | MAE por semilla 42 / 43 / 44 (ciclos) | Mejor MAE |
|---|---:|---:|
| MLP | 17.03 / 16.99 / 17.43 | 16.99 |
| LSTM inicial (`lstm`) | 18.15 / 18.26 / 18.13 | 18.13 |
| LSTM 32 unidades (`lstm_h32`) | 19.34 / 19.55 / 19.51 | 19.34 |
| LSTM learning rate 0.0003 (`lstm_lr3e4`) | 19.34 / 19.70 / 18.75 | 18.75 |

Se congelaron el MLP y la LSTM inicial para la comparación final. La LSTM no
redujo el MAE de validación frente al MLP. Las dos iteraciones de hiperparámetros
no mejoraron la LSTM inicial. Las corridas compararon las mismas features,
ventana de 30 ciclos, motores y función de pérdida.

## Fase 9: test oficial terminal

La evaluación se hizo **una vez después de congelar la selección**, con una
predicción por cada uno de los 100 motores del test oficial en su último ciclo
observado. No se utilizaron esas etiquetas para ajustar modelos ni umbrales.

| Métrica | MLP | LSTM |
|---|---:|---:|
| MAE (ciclos) | 16.09 | 14.70 |
| RMSE (ciclos) | 24.07 | 20.74 |
| NASA PHM08 Score, suma | 30,837.07 | 1,814.59 |
| Sesgo, predicho − real (ciclos) | +7.31 | +6.46 |

La mejora observada de MAE de LSTM frente a MLP en test fue **8.6 %**, inferior
a la meta aspiracional del 10 %. El bootstrap pareado por motor dio un intervalo
del 95 % de **−4.85 a +1.67 ciclos** para `MAE_LSTM − MAE_MLP`; incluye cero.
Por tanto, estos 100 motores no aportan evidencia suficiente para afirmar una
ventaja robusta de MAE. Ambos sesgos positivos indican tendencia a sobreestimar
la vida restante.

En el caso más extremo del MLP, el motor 78 tenía RUL real de 107 ciclos y se
predijeron 205.99. Para la LSTM, el mayor error absoluto fue el motor 67:
RUL real de 77 y predicción de 144.42. Estos errores de sobreestimación son
relevantes para decisiones de intervención. Las predicciones por motor y los
residuales están en `artifacts/metrics/test_predictions.csv`.

La sensibilidad por permutación se calculó en validación. Para el MLP, las
mayores variaciones de MAE correspondieron a `sensor_8` y `cycle`; para la LSTM,
a `sensor_11`, `sensor_9` y `sensor_14`. Es sensibilidad predictiva frente a
perturbaciones, **no causalidad física**.

## Fase 10: política económica simulada

Se evaluaron trayectorias completas de los 20 motores de validación. El test
oficial truncado no permite simular decisiones ciclo por ciclo. Los costos son
**hipotéticos**, en quetzales, definidos en `configs/business.yaml` y explicados
en `docs/business_assumptions.md`.

| Política | Umbral elegido | Costo medio Q/motor | Ahorro frente a calendario |
|---|---:|---:|---:|
| Calendario | edad 125 | 17,850.00 | — |
| MLP | RUL predicho 15 | 11,491.60 | 35.62 % |
| LSTM | RUL predicho 15 | 11,333.06 | 36.51 % |

En este escenario, la LSTM tiene un costo medio **Q158.53 por motor** menor que
el MLP pese a su mayor MAE de validación. Las tres políticas seleccionadas
registraron cero fallas en estos 20 motores. Los umbrales y sus costos se
calcularon en los **mismos motores de validación**; los porcentajes son una
estimación exploratoria potencialmente optimista, no ahorro operativo real.
`artifacts/business/sensitivity.csv` contiene los escenarios de ±20 % y ±50 %
para intervención, falla y vida descartada.

## Limitaciones

- FD001 representa motores simulados, ciclos y un solo modo de degradación.
- El test oficial aporta etiquetas terminales, no historias completas de fallo.
- El intervalo bootstrap se calculó por motor; las ventanas solapadas de
  validación no son observaciones independientes.
- Los costos y la regla de que una intervención oportuna evita la falla son
  supuestos del simulador.
- La selección de umbrales y su evaluación económica comparten validación;
  confirmar el ahorro requeriría otra muestra completa de motores.
