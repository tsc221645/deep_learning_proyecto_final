# Guía de la interfaz de PredictGuard

## Propósito y límites

La aplicación presenta estimaciones de vida útil restante (RUL) para motores
simulados NASA C-MAPSS FD001. Un ciclo es una unidad de operación del conjunto,
no una hora, una fecha ni una probabilidad de falla. La interfaz es un prototipo
académico y no debe usarse para decidir mantenimiento real.

La navegación tiene cuatro pantallas: **Flota**, **Motor**, **Modelos** y
**Simulador**. La API y los modelos deben estar disponibles para cargar los datos.

## 1. Flota

Resume los 100 motores del conjunto oficial de test, cada uno en su último
ciclo observado.

- **Motores observados**: cantidad de trayectorias terminales mostradas.
- **Bajo el umbral**: número de motores para los que al menos uno de los
  modelos estima un RUL menor o igual al umbral de atención editable. Inicia
  en **15 ciclos** y solo cambia el indicador de la flota. No altera las
  predicciones ni los umbrales del simulador.
- **MAE test**: error absoluto medio de las predicciones terminales de cada
  modelo, comparadas con las etiquetas oficiales de los 100 motores.
- La tabla permite buscar un ID, ordenar por menor RUL y abrir el detalle.

### Qué significa el umbral de la flota

Si el umbral se fija en 15, una fila aparece como **Bajo el umbral** cuando
`min(RUL_MLP, RUL_LSTM) <= 15`. Es una regla visual para llamar la atención,
no una recomendación de intervenir. No significa que la falla ocurrirá en 15
ciclos ni que exista una confianza del 100 %.

## 2. Motor

Combina la señal histórica disponible, ambas predicciones y la vista
conceptual del turbofan para el mismo motor seleccionado.

- **Ciclos observados**: último ciclo disponible en la trayectoria truncada.
- **RUL predicho MLP y LSTM**: ciclos restantes estimados en esa observación.
- **Señal observada**: trayectoria del sensor elegido, desde el primer ciclo
  disponible hasta el último, con valores crudos del sensor. El selector cambia
  la señal y su valor reciente en la explicación, pero la predicción sigue
  usando las 19 señales del modelo.
- **Explorador 3D**: arrastrar para rotar, usar la rueda para acercar, elegir
  corte interno o exterior, restablecer la cámara y seleccionar visualmente
  ventilador, compresor, combustión, turbina, escape o carcasa.
- **Flujo del modelo**: ilustra la ventana de 30 ciclos por 19 señales, las
  capas densas de 128 y 64 unidades del MLP, y el RUL predicho. También muestra
  la estimación LSTM, que procesa la secuencia temporal sin aplanarla.

La forma del motor es conceptual y no representa un diseño exacto de NASA.
C-MAPSS no especifica la ubicación física de cada canal de sensor. La interfaz
no coloca sensores sobre el motor ni atribuye a un sensor un componente real.

El conjunto oficial solo proporciona una etiqueta RUL terminal por motor. Las
estimaciones del detalle no muestran una etiqueta real como si existiera para
todo ciclo.

## 3. Modelos

Compara los modelos congelados usando una predicción terminal para cada uno de
los 100 motores de test.

| Resultado mostrado | Interpretación |
|---|---|
| MAE | Error absoluto medio en ciclos. Menor es mejor. |
| RMSE | Penaliza más los errores grandes. Menor es mejor. |
| NASA PHM Score | Penalización asimétrica acumulada. Menor es mejor. |
| Sesgo | Promedio de `predicción - valor real`. Positivo indica sobreestimación; negativo, subestimación. |
| MAE de validación | Referencia del conjunto de validación empleado durante desarrollo. No es el resultado de test. |

### Evaluación final reportada

| Métrica | MLP | LSTM |
|---|---:|---:|
| MAE test | 16.09 ciclos | 14.70 ciclos |
| RMSE test | 24.07 ciclos | 20.74 ciclos |
| NASA PHM08 Score, suma | 30,837.07 | 1,814.59 |
| Sesgo | +7.31 ciclos | +6.46 ciclos |
| MAE validación de la corrida seleccionada | 16.99 ciclos | 18.13 ciclos |

La diferencia de MAE observada en test fue favorable a LSTM, pero el intervalo
bootstrap pareado por motor del 95 % para `MAE_LSTM - MAE_MLP` fue de
**−4.85 a +1.67 ciclos** e incluye cero. No permite afirmar una ventaja
estadísticamente robusta de MAE con esta muestra. El sesgo positivo de ambos
modelos advierte que pueden sobreestimar el RUL.

La comparación de test se ejecutó después de congelar la selección de modelos.
No se deben ajustar hiperparámetros ni umbrales a partir de estos resultados.

## 4. Simulador

Compara políticas de intervención en las 20 trayectorias completas de
validación. No usa el test terminal, porque sus historiales completos de falla
no están disponibles.

| Campo | Valor inicial | Efecto |
|---|---:|---|
| Edad de calendario | 125 ciclos | Interviene al alcanzar esa edad. |
| Umbral MLP | 15 ciclos | Interviene en el primer ciclo con `RUL_MLP <= umbral`. |
| Umbral LSTM | 15 ciclos | Interviene en el primer ciclo con `RUL_LSTM <= umbral`. |
| Margen de acción | 5 ciclos | La señal se considera oportuna si quedan al menos esos ciclos antes de la falla simulada. |
| Intervención | Q10,000 | Costo hipotético por intervención. |
| Falla | Q50,000 | Costo hipotético cuando no hay aviso oportuno. |
| Vida útil descartada | Q100/ciclo | Costo supuesto por ciclo restante que se descarta al intervenir. |
| Inferencia MLP / LSTM | Q0.01 / Q0.05 | Costo hipotético por predicción evaluada; el calendario no realiza inferencias. |

El umbral del simulador es independiente del umbral visual de Flota. Bajar el
umbral de RUL retrasa la intervención; subirlo tiende a adelantarla. Si la
intervención se dispara con al menos el margen configurado antes de la falla,
el simulador cobra intervención y vida descartada y considera evitada la falla.
Sin señal, o con una señal tardía, cobra el costo de falla. El detalle de esta
regla está en [business_assumptions.md](./business_assumptions.md).

### Escenario base reportado

| Política | Umbral | Costo medio | Fallas en validación | Ahorro estimado frente a calendario |
|---|---:|---:|---:|---:|
| Calendario | 125 ciclos de edad | Q17,850.00/motor | 0 de 20 | Base |
| MLP | 15 ciclos de RUL | Q11,491.60/motor | 0 de 20 | 35.62 % |
| LSTM | 15 ciclos de RUL | Q11,333.06/motor | 0 de 20 | 36.51 % |

Estos montos y porcentajes son hipotéticos. Los umbrales se compararon en los
mismos motores de validación que se usaron para reportar los costos, por lo que
el ahorro puede ser optimista. No representa ahorro real de una aerolínea.
Cambiar valores en el formulario calcula un nuevo escenario y no reentrena los
modelos.

## Lectura recomendada

1. En **Flota**, localizar un motor y revisar si sus estimaciones cruzan el
   umbral visual.
2. Abrir **Motor** para inspeccionar sus señales, comparar las dos estimaciones
   y entender de forma conceptual el flujo de inferencia.
3. En **Modelos**, comparar métricas agregadas y sus limitaciones, no solo el
   MAE.
4. En **Simulador**, cambiar supuestos y observar cómo modifican las decisiones
   y los costos hipotéticos.

## Estilo visual

La interfaz usa Roboto local, navegación y paneles con una paleta azul espacial,
acentos turquesa y ámbar y superficies consistentes. El esquema toma como
referencia visual las imágenes de turbofan proporcionadas, sin copiarlas ni
presentarlas como datos técnicos del benchmark.
