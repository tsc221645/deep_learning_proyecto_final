# Fases 11 y 12: aplicación y despliegue

## Arranque desde un clon limpio

El repositorio incluye los checkpoints congelados, el manifiesto, el escalador,
los datos FD001 de demostración, las predicciones de validación y las figuras.
No hace falta abrir un Notebook ni entrenar modelos para mostrar la aplicación.

```powershell
docker compose up --build
```

- Web: `http://localhost:8080`
- Documentación interactiva de API: `http://localhost:8000/docs`
- Salud: `http://localhost:8000/health`

`docker-compose.yml` inicia el frontend cuando la API está sana. El frontend
sirve Vue con Nginx y envía `/api/*` al servicio FastAPI. Los modelos y el
escalador se cargan al iniciar la API; el navegador nunca recibe checkpoints.
Los puertos se pueden cambiar con `PREDICTGUARD_WEB_PORT` y
`PREDICTGUARD_API_PORT` en un archivo `.env` basado en `.env.example`.

## Flujo de demostración

1. Abre **Flota**, cambia el umbral y selecciona un motor de test.
2. En **Motor**, observa una señal y las predicciones MLP y LSTM del último
   ciclo observado. No se muestra una etiqueta real por ciclo histórico.
3. En **Modelos**, compara MAE, RMSE, score NASA, sesgo y curvas congeladas.
4. En **Simulador**, cambia costos o umbrales y calcula el costo medio sobre
   los 20 motores completos de validación. Todos los importes son hipotéticos.

## Contratos de API

`POST /predict` acepta `{"model":"mlp","unit_id":1}` o
`{"model":"lstm","unit_id":1,"cycle":30}`. También acepta `window`, una
lista de 1 a 30 filas con **exactamente** las features ordenadas según el
manifiesto. El servidor aplica el mismo escalador que se usó al entrenar;
`unit_id` identifica el motor y nunca entra al modelo. `POST /compare` acepta
la misma fuente y devuelve ambos modelos. El esquema completo está en `/docs`.

El API rechaza motores/ciclos inexistentes, ventanas con features faltantes,
ciclos no consecutivos y cuerpos mayores que 1 MB. Los logs JSON registran
ruta, estado, duración y tamaño, sin guardar sensores ni payloads.

## Evidencia de pruebas

- `python -m pytest -q`: incluye contrato de API, consistencia de inferencia,
  datos, costos y tests de fases anteriores.
- `./scripts/benchmark_api.ps1`: 10 solicitudes de calentamiento y 100
  mediciones de `POST /compare` con batch uno. El resultado queda en
  `artifacts/metrics/api_performance.json` e identifica hardware y versiones.
- `docker compose config --quiet`: valida el archivo Compose.
- `docker compose up --build`: verifica el arranque real de web y API.

La medición con `TestClient` ejecuta solicitudes dentro del proceso y no
incluye red, proxy ni carga simultánea. La meta p95 <2 s se refiere al
hardware y protocolo declarados; para un despliegue público debe repetirse
la prueba de rendimiento por HTTP bajo la concurrencia esperada.

En la verificación local del 2 de noviembre de 2026, 100 solicitudes de
`POST /compare` dieron p95 de **18 ms** con `TestClient`, **105 ms** mediante
HTTP local hacia Uvicorn y **56 ms** a través de Nginx y Docker Compose, en
Windows 11, Intel Family 6 Model 165 y CPU. Son mediciones secuenciales de
batch uno; el entorno y el método hacen que las cifras no sean directamente
comparables. En reposo, los contenedores usaban aproximadamente 252 MiB (API)
y 10 MiB (web), según `docker stats --no-stream`.

La suite completa terminó con 39 pruebas aprobadas. Se construyeron las dos
imágenes y `docker compose up -d --wait` dejó ambos servicios sanos. Se
comprobaron por HTTP la página, salud, modelos, motores, historial, predicción,
comparación, simulación y resumen de evaluación. También se recorrió en el
navegador el flujo Flota → Motor → Modelos → Simulador; al cambiar el costo de
vida descartada de Q100 a Q200 por ciclo, los resultados simulados cambiaron.

## Monitoreo propuesto

Los logs actuales permiten vigilar errores HTTP, duración de las solicitudes
y versión del modelo. Cuando existan datos futuros etiquetados, se debería
comparar el MAE y el sesgo con el test congelado. Para detectar cambios de
entrada, comparar distribuciones de sensores contra las estadísticas de
entrenamiento del manifiesto, preferiblemente por motor y con umbrales
predefinidos. Un cambio detectado debe revisarse antes de reentrenar; no usar
las etiquetas del test oficial como flujo de monitoreo.

## Límites

FD001 es simulado y usa ciclos, no horas ni fechas. Los costos de intervención
son escenarios hipotéticos. La web es un prototipo académico, no un sistema de
mantenimiento aeronáutico operativo. El despliegue público requeriría controles
adicionales de autenticación y capacidad.
