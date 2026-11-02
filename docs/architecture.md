# Arquitectura de PredictGuard

```mermaid
flowchart LR
    U[Usuario] --> W[Vue 3 + Nginx]
    W -->|/api| A[FastAPI]
    A --> M[MLP y LSTM congelados]
    A --> S[Scaler y manifiesto]
    A --> D[FD001 test para demo]
    A --> V[Predicciones de validación]
    V --> C[Simulador de costos]
```

La API verifica los hashes de `selected.json` antes de cargar los checkpoints.
El escalador y el orden de features se comparan con el manifiesto. Los datos de
test sirven para mostrar motores y predecir su RUL terminal; el simulador usa
predicciones de validación de motores completos. Nginx distribuye la interfaz
y redirige `/api/` al backend. El backend carga todos los recursos una vez al
arrancar y los conserva en memoria para las solicitudes.
