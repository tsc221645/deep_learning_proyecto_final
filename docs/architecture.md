# Arquitectura de PredictGuard

PredictGuard combina dos capas: modelos predictivos entrenados (MLP y LSTM) que
estiman la vida útil restante (RUL, en ciclos) de motores turbofan simulados, y
un **asistente de mantenimiento con LLM** (Claude) que consulta esos modelos
mediante herramientas y explica los resultados en lenguaje natural.

## Pipeline completo

```mermaid
flowchart LR
    subgraph ENT["1. Entrada"]
        U["Usuario en la web (Vue 3)<br/>pregunta + motor + umbral"]
        DS["NASA C-MAPSS FD001<br/>21 sensores + 3 ajustes operativos"]
    end

    subgraph PRE["2. Preprocesamiento (offline)"]
        P1["Split por motor 80/20"]
        P2["Quitar sensores constantes<br/>StandardScaler con estadísticas de train"]
        P3["Ventanas de 30 ciclos"]
        P1 --> P2 --> P3
    end

    subgraph DL["3. Modelos predictivos (congelados)"]
        M1["MLP 128→64"]
        M2["LSTM 2×64"]
        SIM["Simulador de costos<br/>calendario vs MLP vs LSTM"]
    end

    subgraph NA["Tokenización, embeddings y RAG"]
        N1["No aplica: los datos son numéricos<br/>y se consultan por herramientas.<br/>La tokenización la hace la API de Claude"]
    end

    subgraph API["4. Backend FastAPI · /assistant/chat"]
        V["Validación de la solicitud<br/>máx. 4000 caracteres, 12 mensajes<br/>límite de 30 consultas/min"]
        C{"¿Respuesta<br/>en caché?"}
        CTX["Contexto: motor y umbral de la pantalla<br/>+ prompt de sistema (reglas y glosario)"]
    end

    subgraph LLM["5. LLM con herramientas (agente)"]
        L["Claude Sonnet 5.5<br/>streaming, hasta 6 rondas"]
        T1["get_fleet_overview"]
        T2["get_engine_diagnosis"]
        T3["simulate_maintenance"]
        T4["get_model_metrics"]
        L <-->|"tool_use / tool_result"| T1
        L <--> T2
        L <--> T3
        L <--> T4
    end

    subgraph POST["6. Postprocesamiento"]
        S["Eventos NDJSON:<br/>text, tool, notice, done<br/>(tokens, latencia, costo)"]
        K["Guardar en caché"]
        FB["Respaldo sin LLM:<br/>resumen determinista si falla<br/>la API o falta la llave"]
    end

    subgraph UI["7. Interfaz"]
        CH["Chat con markdown seguro<br/>chips de herramientas<br/>origen: Claude / caché / sin LLM"]
    end

    DS --> P1
    P3 --> M1
    P3 --> M2
    M1 --> SIM
    M2 --> SIM

    U --> V --> C
    C -->|"sí"| S
    C -->|"no"| CTX --> L
    T1 --> M1
    T1 --> M2
    T2 --> M1
    T2 --> M2
    T3 --> SIM
    T4 --> M1
    T4 --> M2
    L --> S --> K
    L -. "error" .-> FB --> S
    S --> CH

    style NA fill:#eee,stroke:#999,stroke-dasharray: 5 5,color:#666
```

## Componentes

| Etapa | Implementación |
|---|---|
| Interfaz | Vue 3 + Vite, servida por Nginx (`frontend/`). Pestañas Flota, Motor, Modelos, Simulador y Asistente. |
| Backend | FastAPI (`api/main.py`). Endpoints de predicción, comparación, simulación y evaluación, más `/assistant/chat` y `/assistant/status`. |
| Modelos | MLP y LSTM congelados; la API verifica los hashes de `selected.json` antes de cargarlos. |
| Asistente | `api/assistant.py`: prompt de sistema, cuatro herramientas de solo lectura, bucle de herramientas, caché y respaldo. |
| Evaluación del LLM | `scripts/eval_assistant.py` y `docs/assistant_evaluation.md` (agente vs. baseline sin herramientas). |

## Decisiones de diseño

- **El LLM no predice.** Cada cifra que muestra sale de una herramienta que lee
  los modelos congelados, el simulador o las métricas de test. El prompt le
  prohíbe inventar números.
- **Herramientas de solo lectura.** Reciben parámetros acotados (enteros y
  costos validados por `SimulationRequest`), por lo que una instrucción
  maliciosa en la pregunta no puede modificar datos ni ejecutar acciones.
- **Sin RAG.** El asistente consulta datos estructurados mediante herramientas;
  no hay corpus de documentos que recuperar, por eso no hay embeddings, chunking
  ni base vectorial.
- **Resiliencia.** Caché de respuestas en memoria, reintentos del SDK y un
  resumen determinista, claramente marcado como "sin LLM", cuando no hay
  conexión o llave.
- **Observabilidad.** Cada respuesta registra tokens, llamadas a la API,
  latencia y costo estimado (evento `done` y logs de la API).
- **Parámetros de generación.** `max_tokens=1500`. `temperature` y `top_p` no
  se fijan porque Sonnet 5.5 rechaza valores distintos del predeterminado.

## Datos y carga

La API verifica los hashes de `selected.json` antes de cargar los checkpoints.
El escalador y el orden de features se comparan con el manifiesto. Los datos de
test sirven para mostrar motores y predecir su RUL terminal; el simulador usa
predicciones de validación de motores completos. Nginx distribuye la interfaz y
redirige `/api/` al backend (con el almacenamiento en búfer desactivado para
`/api/assistant/` para permitir el streaming). El backend carga todos los
recursos una vez al arrancar y los conserva en memoria para las solicitudes.

## Configuración del asistente

La llave se lee de `ANTHROPIC_API_KEY` (variable de entorno o `.env`, que no se
sube a git) y el modelo de `ANTHROPIC_MODEL`. `docker-compose.yml` pasa ambas al
contenedor de la API.
