# Supuestos del simulador económico

El simulador de `predictguard.policy` utiliza los 20 motores completos de
validación de FD001. Los costos de `configs/business.yaml` son **hipotéticos**,
expresados en quetzales por evento, ciclo o inferencia. No representan costos
publicados por NASA ni por una aerolínea.

## Decisiones

- Calendario: intervención en el ciclo de edad configurado como umbral.
- MLP y LSTM: desde la primera ventana completa, se evalúa una predicción por
  ciclo y se interviene en el primer ciclo donde `RUL_predicho <= umbral`.
- Una intervención se considera oportuna solo si faltan al menos
  `lead_time_cycles` hasta la falla simulada. En ese caso evita la falla y se
  cobra la intervención más el costo de vida útil descartada.
- Una señal tardía, o la ausencia de señal, lleva al costo de falla. No se cobra
  también la intervención ni vida descartada.
- Cada predicción evaluada cobra el valor hipotético configurado en
  `inference_q_by_policy`. El calendario no infiere. Estos importes son
  supuestos contables; la latencia de cada modelo se mide aparte.
- El resultado se calcula por motor y luego se promedia; los umbrales se
  seleccionan por costo medio en validación.

La selección de umbrales y la comparación económica usan los mismos motores de
validación. Por ello el ahorro estimado es **exploratorio** y puede ser
optimista. El test oficial, que solo tiene una etiqueta terminal por motor, no
se usa para simular trayectorias completas ni para elegir umbrales.

El barrido de sensibilidad varía cada costo de intervención, falla y ciclo
descartado por separado en −50 %, −20 %, +20 % y +50 %. Los resultados se
guardan en `artifacts/business/`; un cambio de costos exige regenerarlos.
