<script setup>
import { computed, onMounted, reactive, ref } from 'vue'

const tabs = [
  { id: 'fleet', label: 'Flota', icon: '◫' },
  { id: 'engine', label: 'Motor', icon: '◉' },
  { id: 'lab', label: 'Modelos', icon: '◈' },
  { id: 'simulator', label: 'Simulador', icon: '▥' },
]
const activeTab = ref('fleet')
const engines = ref([])
const selectedEngine = ref(null)
const history = ref(null)
const comparison = ref(null)
const models = ref([])
const evaluation = ref(null)
const simulation = ref(null)
const loading = ref(true)
const actionLoading = ref(false)
const error = ref('')
const search = ref('')
const sensor = ref('sensor_11')
const maintenanceThreshold = ref(15)
const form = reactive({
  calendar_age: 125,
  mlp_threshold: 15,
  lstm_threshold: 15,
  lead_time_cycles: 5,
  costs: { intervention_q: 10000, failure_q: 50000, unused_cycle_q: 100,
           mlp_inference_q: 0.01, lstm_inference_q: 0.05 },
})

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `Error ${response.status}`)
  }
  return response.json()
}

async function loadEngine(unitId) {
  error.value = ''
  selectedEngine.value = unitId
  try {
    const [detail, compared] = await Promise.all([
      api(`/engines/${unitId}/history`),
      api('/compare', { method: 'POST', body: JSON.stringify({ unit_id: unitId }) }),
    ])
    history.value = detail
    comparison.value = compared
    sensor.value = detail.features.includes(sensor.value) ? sensor.value : detail.features.find(x => x.startsWith('sensor_'))
    activeTab.value = 'engine'
  } catch (problem) {
    error.value = problem.message
  }
}

async function runSimulation() {
  actionLoading.value = true
  error.value = ''
  try {
    simulation.value = await api('/simulate-maintenance', {
      method: 'POST', body: JSON.stringify(form),
    })
  } catch (problem) {
    error.value = problem.message
  } finally {
    actionLoading.value = false
  }
}

onMounted(async () => {
  try {
    const [fleet, modelList, summary] = await Promise.all([
      api('/engines'), api('/models'), api('/evaluation/summary'),
    ])
    engines.value = fleet
    models.value = modelList
    evaluation.value = summary
    const base = summary.business_validation
    if (base?.costs_q) {
      form.costs.intervention_q = base.costs_q.intervention_q
      form.costs.failure_q = base.costs_q.failure_q
      form.costs.unused_cycle_q = base.costs_q.unused_cycle_q
      form.costs.mlp_inference_q = base.inference_q_by_policy?.mlp ?? 0.01
      form.costs.lstm_inference_q = base.inference_q_by_policy?.lstm ?? 0.05
      form.lead_time_cycles = base.lead_time_cycles
    }
    await runSimulation()
  } catch (problem) {
    error.value = problem.message
  } finally {
    loading.value = false
  }
})

const visibleEngines = computed(() => engines.value.filter(engine =>
  String(engine.unit_id).includes(search.value.trim())).sort((a, b) =>
    Math.min(a.predictions.mlp, a.predictions.lstm) - Math.min(b.predictions.mlp, b.predictions.lstm)))
const atRisk = computed(() => engines.value.filter(engine =>
  Math.min(engine.predictions.mlp, engine.predictions.lstm) <= maintenanceThreshold.value).length)
const selectedPolicy = computed(() => simulation.value?.policies ?? [])
const maxCost = computed(() => Math.max(1, ...selectedPolicy.value.map(item => item.mean_cost_q_per_engine)))
const chart = computed(() => {
  const records = history.value?.records ?? []
  if (records.length < 2 || !sensor.value) return { points: '', min: 0, max: 0 }
  const values = records.map(row => Number(row[sensor.value]))
  const min = Math.min(...values), max = Math.max(...values)
  const span = max - min || 1
  const points = values.map((value, index) =>
    `${24 + index * 752 / (values.length - 1)},${226 - (value - min) * 178 / span}`).join(' ')
  return { points, min, max }
})

const fmt = value => Number(value ?? 0).toLocaleString('es-GT', { maximumFractionDigits: 2 })
const money = value => `Q${fmt(value)}`
const modelName = name => name === 'mlp' ? 'MLP' : 'LSTM'
const figureUrl = name => `/api/evaluation/figures/${name}`
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><span class="brand-mark">✦</span><div><strong>PredictGuard</strong><small>Laboratorio FD001</small></div></div>
      <div class="side-label">EXPLORAR</div>
      <nav aria-label="Vistas principales">
        <button v-for="tab in tabs" :key="tab.id" :class="['nav-item', { active: activeTab === tab.id }]" @click="activeTab = tab.id">
          <span class="nav-icon">{{ tab.icon }}</span>{{ tab.label }}
        </button>
      </nav>
      <div class="side-note"><div class="pulse"></div><div><strong>Datos simulados</strong><span>NASA C-MAPSS · FD001</span></div></div>
      <p class="sidebar-foot">Prototipo académico.<br>No usar para mantenimiento real.</p>
    </aside>

    <div class="main-wrap">
      <header class="topbar"><div class="topbar-title">PREDICCIÓN DE VIDA ÚTIL RESTANTE</div><div class="topbar-right"><span class="status-dot"></span> Modelos cargados <span class="top-divider"></span> RUL en ciclos</div></header>
      <main>
        <div v-if="error" class="error-banner" role="alert">{{ error }} <button @click="error = ''">Cerrar</button></div>
        <div v-if="loading" class="loading-panel">Cargando motores y modelos…</div>

        <template v-else-if="activeTab === 'fleet'">
          <div class="page-heading"><div><div class="eyebrow">VISTA GENERAL</div><h1>Estado de la flota</h1><p>Explora 100 motores simulados fuera del conjunto de entrenamiento.</p></div><span class="date-pill">FD001 · Test oficial</span></div>
          <div class="stat-grid">
            <div class="stat-card"><span>Motores observados</span><strong>{{ engines.length }}</strong><small>trayectorias truncadas</small></div>
            <div class="stat-card accent"><span>Bajo el umbral</span><strong>{{ atRisk }}</strong><small>según cualquiera de los modelos</small></div>
            <div class="stat-card"><span>MAE test · MLP</span><strong>{{ fmt(evaluation?.test?.metrics?.mlp?.mae_cycles) }}</strong><small>ciclos · observación terminal</small></div>
            <div class="stat-card"><span>MAE test · LSTM</span><strong>{{ fmt(evaluation?.test?.metrics?.lstm?.mae_cycles) }}</strong><small>ciclos · observación terminal</small></div>
          </div>
          <section class="panel"><div class="panel-head"><div><h2>Motores de demostración</h2><p>Predicciones en el último ciclo observado. Los ciclos no representan fechas.</p></div><div class="control-row"><label class="inline-control">Umbral <input v-model.number="maintenanceThreshold" type="number" min="0" max="300" /> ciclos</label><input v-model="search" class="search" type="search" placeholder="Buscar motor…" aria-label="Buscar motor" /></div></div>
            <div class="table-scroll"><table><thead><tr><th>Motor</th><th>Último ciclo</th><th>MLP · RUL</th><th>LSTM · RUL</th><th>Indicador</th><th></th></tr></thead><tbody><tr v-for="engine in visibleEngines" :key="engine.unit_id"><td><strong>#{{ String(engine.unit_id).padStart(3, '0') }}</strong></td><td>{{ engine.last_observed_cycle }}</td><td>{{ fmt(engine.predictions.mlp) }} ciclos</td><td>{{ fmt(engine.predictions.lstm) }} ciclos</td><td><span :class="['badge', Math.min(engine.predictions.mlp, engine.predictions.lstm) <= maintenanceThreshold ? 'badge-warn' : 'badge-ok']">{{ Math.min(engine.predictions.mlp, engine.predictions.lstm) <= maintenanceThreshold ? 'Bajo umbral' : 'Sobre umbral' }}</span></td><td><button class="text-button" @click="loadEngine(engine.unit_id)">Ver motor →</button></td></tr></tbody></table></div>
          </section>
        </template>

        <template v-else-if="activeTab === 'engine'">
          <div class="page-heading"><div><div class="eyebrow">DETALLE DEL MOTOR</div><h1>Motor #{{ String(selectedEngine ?? 1).padStart(3, '0') }}</h1><p>Mediciones observadas y predicción en el último ciclo disponible.</p></div><select class="engine-select" :value="selectedEngine ?? ''" @change="loadEngine(Number($event.target.value))"><option disabled value="">Seleccionar motor</option><option v-for="engine in engines" :key="engine.unit_id" :value="engine.unit_id">Motor #{{ String(engine.unit_id).padStart(3, '0') }}</option></select></div>
          <div v-if="!history" class="empty-state">Selecciona un motor en la vista de flota.</div>
          <template v-else>
            <div class="engine-top"><div class="engine-fact"><span>Ciclos observados</span><strong>{{ history.last_observed_cycle }}</strong></div><div class="engine-fact"><span>RUL predicho · MLP</span><strong>{{ fmt(comparison?.predictions?.mlp?.rul_predicted) }}</strong><small>ciclos</small></div><div class="engine-fact"><span>RUL predicho · LSTM</span><strong>{{ fmt(comparison?.predictions?.lstm?.rul_predicted) }}</strong><small>ciclos</small></div></div>
            <section class="panel chart-panel"><div class="panel-head"><div><h2>Señal observada</h2><p>Trayectoria disponible del motor seleccionado</p></div><label class="inline-control">Sensor <select v-model="sensor"><option v-for="feature in history.features.filter(item => item.startsWith('sensor_'))" :key="feature" :value="feature">{{ feature }}</option></select></label></div><div class="chart-wrap"><div class="chart-y"><span>{{ fmt(chart.max) }}</span><span>{{ fmt(chart.min) }}</span></div><svg viewBox="0 0 800 250" preserveAspectRatio="none" role="img" :aria-label="`Trayectoria de ${sensor}`"><line x1="24" y1="48" x2="776" y2="48" class="grid-line" /><line x1="24" y1="137" x2="776" y2="137" class="grid-line" /><line x1="24" y1="226" x2="776" y2="226" class="grid-line" /><polyline :points="chart.points" class="sensor-line" /></svg></div><div class="chart-axis"><span>Ciclo 1</span><span>Ciclo {{ history.last_observed_cycle }}</span></div></section>
            <div class="info-strip"><strong>Etiqueta real:</strong> el conjunto oficial solo proporciona RUL real para el último ciclo observado de cada motor. No se muestran etiquetas inventadas para ciclos anteriores.</div>
          </template>
        </template>

        <template v-else-if="activeTab === 'lab'">
          <div class="page-heading"><div><div class="eyebrow">LABORATORIO DE MODELOS</div><h1>Comparar predicciones</h1><p>Resultados congelados sobre 100 motores de test, una observación terminal por motor.</p></div><span class="date-pill">Evaluación final</span></div>
          <div class="model-grid"><article v-for="model in models" :key="model.name" class="model-card"><div class="model-head"><span class="model-symbol">{{ model.name === 'mlp' ? 'M' : 'L' }}</span><div><h2>{{ modelName(model.name) }}</h2><small>Versión {{ model.version }}</small></div></div><div class="model-metric"><span>MAE en test</span><strong>{{ fmt(evaluation?.test?.metrics?.[model.name]?.mae_cycles) }} <em>ciclos</em></strong></div><div class="model-details"><div><span>RMSE</span><strong>{{ fmt(evaluation?.test?.metrics?.[model.name]?.rmse_cycles) }}</strong></div><div><span>Score NASA</span><strong>{{ fmt(evaluation?.test?.metrics?.[model.name]?.nasa_phm_score_sum) }}</strong></div><div><span>Sesgo</span><strong>{{ fmt(evaluation?.test?.metrics?.[model.name]?.bias_cycles) }}</strong></div><div><span>MAE validación</span><strong>{{ fmt(model.validation_mae_cycles) }}</strong></div></div></article></div>
          <div class="figure-grid"><section class="panel figure-panel"><h2>Comparación en test</h2><img :src="figureUrl('test_comparison.png')" alt="Predicción y residuales de MLP y LSTM en test" /></section><section class="panel figure-panel"><h2>Entrenamiento MLP</h2><img :src="figureUrl('mlp_learning_curves.png')" alt="Curvas de aprendizaje MLP" /></section><section class="panel figure-panel"><h2>Entrenamiento LSTM</h2><img :src="figureUrl('lstm_learning_curves.png')" alt="Curvas de aprendizaje LSTM" /></section></div>
          <div class="info-strip">La LSTM tuvo menor MAE en test, pero el intervalo bootstrap de la diferencia incluye cero. Las predicciones pueden sobreestimar la vida restante.</div>
        </template>

        <template v-else-if="activeTab === 'simulator'">
          <div class="page-heading"><div><div class="eyebrow">ESCENARIOS HIPOTÉTICOS</div><h1>Simulador de mantenimiento</h1><p>Compara calendario, MLP y LSTM sobre 20 trayectorias completas de validación.</p></div><span class="date-pill">Costos en quetzales</span></div>
          <div class="sim-layout"><section class="panel settings-panel"><h2>Configurar escenario</h2><p>Ajusta costos y umbrales para explorar decisiones simuladas.</p><div class="field-grid"><label>Costo de intervención (Q)<input v-model.number="form.costs.intervention_q" type="number" min="0" /></label><label>Costo de falla (Q)<input v-model.number="form.costs.failure_q" type="number" min="0" /></label><label>Vida descartada (Q/ciclo)<input v-model.number="form.costs.unused_cycle_q" type="number" min="0" /></label><label>Margen de acción (ciclos)<input v-model.number="form.lead_time_cycles" type="number" min="1" /></label><label>Calendario · edad<input v-model.number="form.calendar_age" type="number" min="1" /></label><label>MLP · umbral RUL<input v-model.number="form.mlp_threshold" type="number" min="1" /></label><label>LSTM · umbral RUL<input v-model.number="form.lstm_threshold" type="number" min="1" /></label><label>Inferencia LSTM (Q)<input v-model.number="form.costs.lstm_inference_q" type="number" min="0" step="0.01" /></label></div><button class="primary-button" :disabled="actionLoading" @click="runSimulation">{{ actionLoading ? 'Calculando…' : 'Calcular costos' }}</button></section>
            <section class="panel results-panel"><div class="panel-head"><div><h2>Costo medio por motor</h2><p>Mismo conjunto y costos base para las tres políticas</p></div></div><div v-if="simulation" class="bars"><div v-for="item in selectedPolicy" :key="item.policy" class="bar-row"><div class="bar-head"><strong>{{ item.policy === 'calendar' ? 'Calendario' : modelName(item.policy) }}</strong><span>{{ money(item.mean_cost_q_per_engine) }}</span></div><div class="bar-track"><div :class="['bar-fill', item.policy]" :style="{ width: `${100 * item.mean_cost_q_per_engine / maxCost}%` }"></div></div><div class="bar-foot">Umbral {{ item.threshold_cycles }} ciclos · {{ item.failures }} fallas · ahorro {{ fmt(item.savings_vs_calendar_pct) }} %</div></div></div><div v-else class="empty-state">Configura un escenario para ver resultados.</div><div class="info-strip compact">Estimación exploratoria sobre validación. Los costos son supuestos, no datos de NASA ni ahorros reales.</div></section></div>
          <section class="panel figure-panel full-figure"><h2>Curva de valor del escenario base</h2><img :src="figureUrl('business_cost_curve.png')" alt="Costo medio por motor según umbral de intervención" /></section>
        </template>
      </main>
    </div>
  </div>
</template>
