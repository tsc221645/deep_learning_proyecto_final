# Evaluación del asistente de mantenimiento (LLM)

Reproducir: `python scripts/eval_assistant.py` (requiere `ANTHROPIC_API_KEY`). Salidas en `artifacts/assistant_eval/`
(`cases.json`, `results.json`, `summary.json`, `per_case.csv`). Costo de una corrida completa: ~US$0.48.

## Diseño
- **32 casos** en 6 categorías: diagnóstico de motor (10), flota (5), simulador de costos (5), métricas (4),
  conceptos (5) y robustez (3: fuera de tema, inyección de prompt y motor inexistente).
- **Respuestas de referencia calculadas, no escritas a mano**: las cifras esperadas salen de las mismas funciones
  determinísticas de la API sobre los modelos congelados (verificadas contra `/compare` en `tests/test_assistant.py`).
  Los conceptos se validan con grupos de palabras clave. **Pendiente: validación humana de los casos por el grupo.**
- **Configuraciones:** *agente* = Claude Sonnet 5.5 con herramientas; *baseline* = el mismo modelo y prompt, sin
  herramientas (no puede consultar los modelos).
- **Métricas:** `pass` (herramientas correctas + todos los hechos + sin números no respaldados), `facts` (cifras y
  conceptos esperados presentes), `hallucination` (respuesta con algún número "fuerte" que no aparece en el contexto ni
  en las salidas de las herramientas; heurística conservadora), `tool_accuracy`, latencia, tokens y costo estimado.
- **Parámetros de generación:** `max_tokens=1500`. `temperature` y `top_p` no se fijan porque Sonnet 5.5 rechaza
  (400) valores distintos del predeterminado.
- **Costo estimado:** tarifas públicas de Sonnet 5.5 (US$2 entrada / US$10 salida por millón de tokens; lectura de
  caché US$0.20; escritura de caché 1.25x entrada).

## Resultados (1 corrida, 32 casos por configuración)
| Métrica | Agente (con herramientas) | Baseline (sin herramientas) |
|---|---|---|
| Casos aprobados | **29/32 (90.6 %)** | 6/32 (18.8 %) |
| Hechos correctos | **32/32 (100 %)** | 6/32 (18.8 %) |
| Respuestas con números no respaldados | 3/32 (9.4 %) | 11/32 (34.4 %) |
| Elección correcta de herramientas | 31/32 (96.9 %) | n/a |
| Latencia media / p95 | 5.5 s / 8.6 s | 5.9 s / 7.3 s |
| Tokens de contexto por consulta (media) | 4,824 | 1,347 |
| Tokens de salida (media) | 547 | 687 |
| Llamadas a la API por consulta | 1.78 | 1.00 |
| Costo estimado por consulta | US$0.0078 | US$0.0073 |

| Categoría | Agente (aprobados) | Baseline (aprobados) |
|---|---|---|
| Diagnóstico de motor | 10/10 | 0/10 |
| Flota | 5/5 | 0/5 |
| Simulador | 4/5 | 0/5 |
| Métricas | 3/4 | 0/4 |
| Conceptos | 5/5 | 5/5 |
| Robustez | 2/3 | 1/3 |

**Lectura:** en preguntas sobre datos (diagnóstico, flota, simulador, métricas) el baseline no acierta ninguna, porque
no tiene acceso a los modelos; el agente acierta todas las cifras. En conceptos ambos empatan: las herramientas no
aportan nada allí. El costo y la latencia son similares porque el costo lo domina la generación de la respuesta.

## Los 3 casos que no pasan (revisión manual)
Los tres son limitaciones del evaluador automático, no cifras erróneas del asistente:
1. **#20 (simulador):** sugiere "probar un calendario más tardío (por ejemplo, 140)": un valor propuesto, no un dato.
2. **#21 (métricas):** "intervalo bootstrap 95 %": es una etiqueta, no una cifra de resultado.
3. **#32 (motor #999):** responde correctamente que solo hay motores 1 a 100 **sin llamar a la herramienta** (el
   modelo dedujo que el número era inválido). Cuenta como fallo de herramienta y marca "100" como número no respaldado.

El resultado **automático se reporta tal cual (90.6 %)**. No se ajustó el evaluador después de ver los resultados,
salvo corregir dos errores de extracción de números (signos y montos con "Q") detectados en una prueba piloto.

## Limitaciones de la evaluación
- Una sola corrida por configuración, sin repeticiones: la varianza del LLM no está medida.
- 32 casos; las referencias numéricas son automáticas, las de conceptos son palabras clave (no un juez humano ni un
  LLM-as-judge validado).
- La heurística de alucinación solo detecta números; no detecta afirmaciones cualitativas falsas.
- La métrica de latencia incluye red y puede variar el día de la feria.
