# Guía de la interfaz de PredictGuard

Esta guía tiene dos partes: **A. Versión resumida** (una o dos hojas, para
llevar impresa y repasar antes de presentar) y **B. Versión extendida** (cada
pantalla, cada elemento y cómo responder preguntas).

---

# A. VERSIÓN RESUMIDA

## En una frase

PredictGuard estima cuántos **ciclos de operación le quedan a un motor
turbofan simulado** (RUL, vida útil restante) con dos redes neuronales, **MLP**
y **LSTM**, y muestra qué tan confiable es esa estimación y cuánto costaría
decidir con ella. Los datos son simulados (NASA C-MAPSS FD001). Es un
prototipo académico, **no sirve para mantenimiento real**.

## Tres ideas que no se deben olvidar

1. **RUL = ciclos, no horas ni fechas ni probabilidad de falla.**
2. **Los datos y los costos son simulados.** Los quetzales son supuestos; el
   ahorro es una estimación dentro de un escenario, no un ahorro real.
3. **Ningún modelo es claramente mejor.** La LSTM tuvo menor error en test,
   pero la diferencia no es estadísticamente concluyente, y ambos tienden a
   **sobreestimar** la vida restante.

## El recorrido: cuatro pantallas, en orden

| # | Pantalla | Pregunta que responde | Qué mostrar |
|---|---|---|---|
| 1 | **Flota** | ¿Qué motores atender primero? | 100 motores ordenados por RUL; contadores por estado; umbral editable; histograma. |
| 2 | **Motor** | ¿Qué le pasa a este motor y dónde? | Ciclo de vida estimado, modelo 3D, variables por componente, gráfico del sensor, cómo lee cada modelo. |
| 3 | **Modelos** | ¿Cuánto confiar en MLP y LSTM? | Métricas de test, intervalo de confianza, curvas de entrenamiento. |
| 4 | **Simulador** | ¿Cuánto cuesta decidir con cada política? | Costo medio por motor: calendario vs. MLP vs. LSTM. |

## Guion sugerido (3 a 4 minutos)

1. **Flota (45 s).** "Estos son 100 motores en su último ciclo observado.
   Con umbral de 15 ciclos, 12 quedan bajo el umbral." Cambia el umbral y
   muestra que los contadores se mueven. Aclara: el umbral solo ordena la
   atención en esta pantalla.
2. **Motor (90 s).** Abre el más urgente. Señala la barra de ciclo de vida
   (observado + RUL estimado). En el 3D: gira, usa *Corte* y *Despiece*, haz
   clic en *HPC*, elige la variable T30 y muestra que se enfoca su estación.
3. **Modelos (45 s).** MAE 14.70 (LSTM) vs. 16.09 (MLP), pero el intervalo
   incluye cero: no se puede afirmar que uno gane.
4. **Simulador (45 s).** Calendario Q17,850 vs. ≈ Q11,400 con los modelos.
   Cambia el costo de falla y muestra que se recalcula solo. Cierra con la
   advertencia: costos hipotéticos, umbrales elegidos en los mismos motores.

## Números clave (escenario reportado)

| | MLP | LSTM |
|---|---:|---:|
| MAE test (ciclos) | 16.09 | 14.70 |
| RMSE test (ciclos) | 24.07 | 20.74 |
| Score NASA PHM08 | 30,837 | 1,815 |
| Sesgo (ciclos) | +7.31 | +6.46 |
| MAE validación (ciclos) | 16.99 | 18.13 |

Diferencia de MAE (LSTM − MLP), IC bootstrap 95 %: **−4.85 a +1.67** (incluye 0).

Simulador base (20 motores de validación): calendario Q17,850 · MLP Q11,491.60
(−35.6 %) · LSTM Q11,333.06 (−36.5 %) · 0 fallas en las tres.

## Preguntas que seguro hacen

- **¿Un ciclo son horas?** No. Es una unidad de operación del conjunto de datos.
- **¿El motor 3D es real?** No: es un turbofan conceptual de dos ejes, el tipo
  que simula C-MAPSS, no un diseño exacto de NASA.
- **¿Los sensores están en esos lugares?** Cada variable se asocia a la
  *estación de medición* que describe la documentación de C-MAPSS (por ejemplo,
  T30 = temperatura a la salida del compresor de alta presión), no a un punto
  físico exacto.
- **¿Ahorra 36 %?** Solo dentro del escenario simulado, y puede ser optimista.
- **¿Por qué dos modelos?** El MLP ve la ventana como un vector; la LSTM lee la
  secuencia en orden. Se compara si el orden temporal aporta.

---

# B. VERSIÓN EXTENDIDA

## B.0 Propósito y límites

La aplicación presenta estimaciones de vida útil restante (RUL) para motores
simulados NASA C-MAPSS FD001. Un **ciclo** es una unidad de operación del
conjunto de datos: no es una hora, una fecha ni una probabilidad de falla. La
interfaz es un prototipo académico y no debe usarse para decidir mantenimiento
real. La API y los modelos deben estar disponibles para cargar los datos.

**Datos que usa cada pantalla**

| Pantalla | Datos | Etiqueta real disponible |
|---|---|---|
| Flota y Motor | 100 motores de **test**, truncados en un ciclo | Solo el RUL del último ciclo (no se muestra) |
| Modelos | Métricas congeladas en test (1 observación terminal por motor) | Sí, oficial |
| Simulador | 20 trayectorias **completas** de validación | Sí, porque se conoce la falla |

El conjunto oficial de test solo entrega un RUL real por motor (el del último
ciclo observado). Por eso la interfaz **no dibuja RUL reales para ciclos
anteriores** ni inventa etiquetas.

## B.1 Navegación y elementos comunes

- **Barra lateral "Recorrido":** los cuatro pasos numerados. Sugiere el orden de
  lectura (qué atender → diagnosticar → confiar → decidir).
- **Barra superior:** ruta actual (por ejemplo, *Motor / Motor #017*) y estado
  de los modelos cargados.
- **URL con estado:** `#/fleet`, `#/engine/17`, `#/lab`, `#/simulator`. Se puede
  recargar o compartir un motor concreto, y el botón "atrás" del navegador
  funciona.
- **Estados del motor (colores):**

| Estado | Regla (con umbral *T*) | Color |
|---|---|---|
| **Bajo umbral** | `min(RUL MLP, RUL LSTM) ≤ T` | Naranja |
| **Vigilar** | `T < min(...) ≤ 2T` | Ámbar |
| **Sobre umbral** | `min(...) > 2T` | Turquesa |

Se usa el **menor** de los dos RUL a propósito: es la lectura más prudente.
Es una regla visual para ordenar la atención, no una recomendación de
intervenir. No significa que la falla ocurrirá en *T* ciclos.

## B.2 Pantalla 1 · Flota

**Pregunta:** ¿qué motores atender primero?

**Tarjetas superiores**

- **Bajo el umbral / Vigilar / Sobre el umbral:** cantidad de motores en cada
  estado. Son botones: un clic filtra la tabla por ese estado, otro clic quita
  el filtro. Con el umbral inicial de 15 ciclos: 12 / 15 / 73.
- **RUL medio de la flota:** promedio del RUL mínimo estimado. Debajo se
  recuerda el MAE de test de cada modelo como referencia de error.

**Distribución del RUL estimado (histograma)**

- Cuenta motores por rango de 20 ciclos usando el menor RUL entre MLP y LSTM.
- Las barras toman el color del estado; la línea punteada es el **umbral**.
- **Umbral de atención** (campo editable, inicia en 15): cambia contadores,
  colores, histograma, filtros y el estado mostrado en la pantalla Motor. **No
  altera las predicciones ni el umbral del simulador.**

**Tabla "Motores observados"**

| Columna | Significado |
|---|---|
| Motor | ID del motor de test (1 a 100). |
| Ciclos observados | Último ciclo disponible de su trayectoria truncada. |
| RUL estimado (mín.) | Barra y valor del menor RUL entre los dos modelos. |
| MLP / LSTM | RUL estimado por cada modelo, en ciclos. |
| Δ modelos | Diferencia absoluta entre ambos. En **ámbar** si supera 15 ciclos: los modelos discrepan y conviene revisar con más cuidado. |
| Estado | Bajo umbral, Vigilar o Sobre umbral. |

- Encabezados clicables ordenan por esa columna (clic de nuevo invierte el
  orden). Por defecto: menor RUL primero.
- Filtros por estado, búsqueda por ID y **clic en la fila** (o Enter) para abrir
  el diagnóstico del motor.

**Cómo explicarla:** "Es el tablero de triage: ordena la flota por vida
restante estimada y señala dónde los dos modelos no se ponen de acuerdo."

## B.3 Pantalla 2 · Motor

**Pregunta:** ¿qué tan cerca está este motor de su fin de vida, qué dicen sus
señales y dónde se miden?

**Encabezado.** Título con el estado del motor. Selector de motor y botones
**← Más urgente / Menos urgente →**, que recorren la flota por RUL estimado
(sirven para revisar uno por uno sin volver a la tabla). Al entrar por primera
vez se carga el motor más urgente para que la pantalla nunca esté vacía.

**Resumen (4 tarjetas)**

- **Ciclos observados:** vida ya recorrida en el conjunto de datos.
- **RUL · MLP / RUL · LSTM:** ciclos restantes que estima cada modelo.
- **Prioridad en la flota:** posición del motor al ordenar por RUL (1 = menor).

**Ciclo de vida estimado.** Una barra por modelo:

- Azul: ciclos **observados**.
- Verde rayado: **RUL estimado** por delante.
- Franja naranja al final: los últimos *T* ciclos (la **zona bajo el umbral**).
- A la derecha: *vida total ≈ observados + RUL*.
- Debajo, una **lectura automática** dice si la estimación más baja está bajo,
  cerca o muy por encima del umbral, y si los modelos coinciden (≤ 15 ciclos de
  diferencia) o discrepan. Recuerda que ambos tienden a sobreestimar.

**Dónde ocurre: el turbofan (modelo 3D)**

El modelo es un **turbofan de dos ejes de alta derivación** con sus componentes:

| Componente | Qué hace | Eje |
|---|---|---|
| Góndola | Carcasa exterior; guía el aire y contiene el flujo de derivación. | — |
| Ventilador (fan) | Acelera el aire; la mayor parte pasa por fuera del núcleo y da la mayor parte del empuje. | Baja presión (LP) |
| Ducto de derivación | Canal entre góndola y núcleo; las guías enderezan el flujo. | — |
| LPC (compresor de baja) | Etapas "booster" que comprimen el aire que entra al núcleo. | LP |
| HPC (compresor de alta) | Muchas etapas; elevan presión y temperatura antes de la combustión. | Alta presión (HP) |
| Cámara de combustión | Mezcla combustible con aire comprimido; gases a la mayor temperatura. | — |
| HPT (turbina de alta) | Mueve el HPC; necesita aire de enfriamiento. | HP |
| LPT (turbina de baja) | Mueve el ventilador y el LPC. | LP |
| Tobera y cono | Aceleran los gases del núcleo para dar el resto del empuje. | — |

Controles:

- **Corte / Exterior:** el corte abre el motor a la mitad para ver el interior.
- **Despiece:** separa los módulos a lo largo del eje.
- **Flujo de aire:** partículas azules (aire frío) que se vuelven naranjas tras
  la combustión.
- **Rotación:** los dos ejes giran a distinta velocidad (HP más rápido que LP).
- **↺ Vista:** restablece la cámara. Arrastrar gira; la rueda acerca.
- **Fila de componentes** (Góndola … Tobera) y **clic directo** en el 3D: eligen
  un componente, que se resalta y muestra un anillo en su estación.

Panel inferior del 3D: descripción del componente y **variables C-MAPSS medidas
en esa estación**, con su último valor y el cambio desde el ciclo 1. Hacer clic
en una variable la elige para el gráfico. Elegir una variable en el selector
también enfoca su componente: **gráfico y 3D están enlazados**.

| Variable | Código | Estación (componente) |
|---|---|---|
| sensor_2 | T24 · temperatura, salida LPC | LPC |
| sensor_3 | T30 · temperatura, salida HPC | HPC |
| sensor_4 | T50 · temperatura, salida LPT | LPT |
| sensor_6 | P15 · presión total, ducto de derivación | Derivación |
| sensor_7 | P30 · presión total, salida HPC | HPC |
| sensor_8 | Nf · velocidad física del ventilador | Ventilador |
| sensor_9 | Nc · velocidad física del núcleo | HPC (eje del núcleo) |
| sensor_11 | Ps30 · presión estática, salida HPC | HPC |
| sensor_12 | phi · combustible / Ps30 | Combustión |
| sensor_13 | NRf · velocidad corregida del ventilador | Ventilador |
| sensor_14 | NRc · velocidad corregida del núcleo | HPC (eje del núcleo) |
| sensor_15 | BPR · relación de derivación | Derivación |
| sensor_17 | htBleed · entalpía del aire sangrado | HPC |
| sensor_20 | W31 · refrigerante sangrado para la HPT | HPT |
| sensor_21 | W32 · refrigerante sangrado para la LPT | LPT |

> Aviso: la asociación es por **sección del motor**, tomada de la tabla de
> variables de Saxena et al. (2008). No indica la posición física exacta de un
> sensor y no se debe leer como diagnóstico de un componente dañado.

**Gráfico de la señal.** Valor crudo del sensor elegido en cada ciclo observado,
con valor inicial, final, cambio acumulado y rango. Pasar el cursor muestra el
ciclo y el valor exactos. El selector cambia la señal mostrada, pero **la
predicción siempre usa las 19 variables** (ciclo, 3 ajustes operativos y 15
sensores), no solo la elegida.

**De las señales a la predicción.** Dos tarjetas:

- **MLP:** ventana de 30 ciclos × 19 variables → se aplana (570 entradas) →
  capas densas de 128 y 64 neuronas con ReLU → RUL. No distingue el orden de
  los ciclos.
- **LSTM:** secuencia de 30 pasos × 19 variables → LSTM de 2 capas y 64
  unidades → estado final → RUL. Lee los ciclos en orden y conserva memoria de
  la tendencia reciente.

**Cómo explicarla:** "Primero cuánta vida estiman los modelos; luego el motor
para ubicar de dónde vienen las señales; por último cómo calculan la respuesta."

## B.4 Pantalla 3 · Modelos

**Pregunta:** ¿cuánto confiar en cada modelo?

**Tarjetas de modelo (MLP y LSTM).** Versión del checkpoint (huella SHA-256
abreviada), ventana de 30 ciclos y barras comparativas de cada métrica. El
modelo con menor MAE lleva la etiqueta **Menor MAE**. Cada barra es relativa al
mayor de los dos valores de esa métrica.

| Métrica | Interpretación |
|---|---|
| MAE test | Error absoluto medio en ciclos. Menor es mejor. |
| RMSE test | Como el MAE, pero penaliza más los errores grandes. Menor es mejor. |
| Score NASA (PHM08) | Penalización asimétrica acumulada: sobreestimar la vida restante cuesta más que subestimarla. Menor es mejor. |
| Sesgo | Promedio de `predicción − valor real`. Positivo = sobreestima (optimista); negativo = subestima. |
| MAE validación | Referencia del conjunto de validación usado en desarrollo. **No es el resultado de test.** |

**¿La diferencia es real?** Gráfico del intervalo bootstrap del 95 %
(remuestreo por motor) de `MAE_LSTM − MAE_MLP`: **−4.85 a +1.67 ciclos**. Como
incluye el cero, no se puede afirmar una ventaja estadísticamente robusta de
ningún modelo con esta muestra.

**Figuras**

- *Predicción y residuales en test:* cada punto es un motor frente a su
  etiqueta oficial.
- *Curvas de entrenamiento* de MLP y LSTM: pérdida de entrenamiento y de
  validación por época.

**Mensajes honestos para decir en voz alta**

- La comparación de test se hizo **después de congelar** los modelos: el test no
  eligió hiperparámetros, umbrales ni checkpoints.
- En validación el MLP fue ligeramente mejor (16.99 vs. 18.13); en test, la LSTM.
- El Score NASA de la LSTM es mucho menor (1,815 vs. 30,837), pero esa métrica
  es sensible a pocos errores grandes de sobreestimación; no es evidencia de
  superioridad por sí sola.
- Ambos modelos pueden **sobreestimar** la vida restante (sesgo positivo).

## B.5 Pantalla 4 · Simulador

**Pregunta:** ¿cuánto costaría decidir con cada política?

Compara tres políticas de intervención en las **20 trayectorias completas de
validación**. No usa el test, porque ahí no se conoce la falla completa.

| Política | Cuándo interviene |
|---|---|
| **Calendario** | Al alcanzar cierta edad (sin predecir nada). |
| **MLP** | En el primer ciclo en que `RUL_MLP ≤ umbral`. |
| **LSTM** | En el primer ciclo en que `RUL_LSTM ≤ umbral`. |

**Escenario (panel izquierdo)**

| Campo | Valor base | Efecto |
|---|---:|---|
| Calendario · edad | 125 ciclos | Edad a la que interviene el calendario. |
| MLP / LSTM · umbral de RUL | 15 ciclos | Bajar el umbral retrasa la intervención; subirlo la adelanta. |
| Margen de acción | 5 ciclos | La señal es oportuna si quedan al menos esos ciclos antes de la falla. |
| Intervención | Q10,000 | Costo hipotético por intervención. |
| Falla | Q50,000 | Costo hipotético si no hay aviso oportuno. |
| Vida descartada | Q100/ciclo | Costo por cada ciclo restante que se desperdicia al intervenir. |
| Inferencia MLP / LSTM | Q0.01 / Q0.05 | Costo por predicción; el calendario no hace inferencias. |

- Los resultados se **recalculan solos** poco después de cambiar un valor (el
  texto de estado indica "Cambios pendientes" o "Resultados al día").
  "Calcular ahora" fuerza el cálculo.
- **Restablecer escenario base** devuelve todos los valores a los reportados.
- Cambiar valores **no reentrena** los modelos.
- El umbral del simulador es **independiente** del umbral de la pantalla Flota.

**Resultado (panel derecho).** Una barra por política con el costo medio por
motor, el umbral usado, las fallas (en naranja si hay alguna), el ahorro frente
al calendario y la etiqueta **Menor costo**.

| Política | Umbral | Costo medio | Fallas | Ahorro vs. calendario |
|---|---:|---:|---:|---:|
| Calendario | 125 ciclos de edad | Q17,850.00 | 0 de 20 | Base |
| MLP | 15 ciclos de RUL | Q11,491.60 | 0 de 20 | 35.62 % |
| LSTM | 15 ciclos de RUL | Q11,333.06 | 0 de 20 | 36.51 % |

**Regla de costo:** si la intervención se dispara con al menos el margen de
acción antes de la falla simulada, se cobra intervención + vida descartada y se
considera evitada la falla. Sin señal, o con señal tardía, se cobra el costo de
falla. Detalle en [business_assumptions.md](./business_assumptions.md).

**Curva de costo del escenario base.** Costo medio por motor según el umbral de
intervención para cada política. Un umbral bajo arriesga avisar tarde (fallas);
uno alto interviene pronto y descarta vida útil.

**Advertencia obligatoria:** los umbrales se eligieron y se evaluaron en los
**mismos** motores de validación, por lo que el ahorro puede ser **optimista**.
Los montos son supuestos, no datos de NASA ni ahorros reales de una aerolínea.

## B.6 Preguntas frecuentes (respuestas cortas)

| Pregunta | Respuesta |
|---|---|
| ¿Qué es el RUL? | Vida útil restante, en **ciclos de operación** del conjunto de datos. |
| ¿Por qué el estado usa el menor de los dos RUL? | Es la lectura más prudente: si cualquiera de los modelos ve poca vida, el motor se marca. |
| ¿Qué pasa si los modelos discrepan? | Se señala en ámbar (Δ > 15 ciclos) y la lectura automática pide cautela. |
| ¿El umbral de Flota cambia las predicciones? | No. Solo ordena y colorea la pantalla. |
| ¿Por qué no se ve el RUL real? | El conjunto oficial solo da uno por motor (el terminal); no se muestran etiquetas inventadas. |
| ¿Qué modelo es mejor? | La LSTM tuvo menor error en test, pero el intervalo incluye cero y en validación ganó el MLP. No hay ventaja concluyente. |
| ¿Los sensores están donde dice el 3D? | Se asocian a la estación de medición documentada, no a una posición exacta. |
| ¿Por qué el simulador usa validación? | Porque tiene trayectorias completas hasta la falla; el test no. |
| ¿El 36 % de ahorro es real? | No. Es una estimación hipotética y posiblemente optimista dentro del escenario simulado. |
| ¿Sirve para mantenimiento real? | No. Es un prototipo académico con datos simulados. |

## B.7 Lectura recomendada (flujo de uso)

1. **Flota:** localizar un motor y revisar su estado; ajustar el umbral si hace
   falta.
2. **Motor:** leer el ciclo de vida estimado, explorar el 3D y la variable que
   interesa, y comprobar si los modelos coinciden.
3. **Modelos:** entender el error y el límite estadístico de la comparación.
4. **Simulador:** cambiar supuestos y ver cómo mueven las decisiones y los
   costos hipotéticos.

## B.8 Estilo visual

Interfaz oscura con paleta azul espacial y acentos turquesa y ámbar, tipografía
Roboto local, y un único archivo de estilos (`frontend/src/theme.css`). El 3D se
genera en el navegador con Three.js; ninguna imagen del motor se presenta como
dato técnico del benchmark.
