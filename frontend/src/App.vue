<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import FleetView from './views/FleetView.vue'
import EngineView from './views/EngineView.vue'
import ModelsView from './views/ModelsView.vue'
import SimulatorView from './views/SimulatorView.vue'
import { minRul, pad } from './lib/engine.js'

const tabs = [
  { id: 'fleet', label: 'Flota', hint: 'Qué motores atender' },
  { id: 'engine', label: 'Motor', hint: 'Diagnóstico y 3D' },
  { id: 'lab', label: 'Modelos', hint: 'Cuánto confiar' },
  { id: 'simulator', label: 'Simulador', hint: 'Cuánto cuesta decidir' },
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
const simBusy = ref(false)
const simDirty = ref(false)
const error = ref('')
const sensor = ref('sensor_11')
const threshold = ref(15)

const BASE_FORM = () => ({
  calendar_age: 125, mlp_threshold: 15, lstm_threshold: 15, lead_time_cycles: 5,
  costs: { intervention_q: 10000, failure_q: 50000, unused_cycle_q: 100, mlp_inference_q: 0.01, lstm_inference_q: 0.05 },
})
const form = reactive(BASE_FORM())
let baseForm = BASE_FORM()

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `Error ${response.status}`)
  }
  return response.json()
}

// ---- enrutado por hash: #/fleet, #/engine/17, #/lab, #/simulator ------------
const queue = computed(() => [...engines.value].sort((a, b) => minRul(a.predictions) - minRul(b.predictions)).map(e => e.unit_id))

function go(tab, unitId) {
  const target = tab === 'engine' && (unitId ?? selectedEngine.value) ? `#/engine/${unitId ?? selectedEngine.value}` : `#/${tab}`
  if (location.hash !== target) location.hash = target
  else applyRoute()
}

let loadToken = 0
async function loadEngine(unitId) {
  const token = ++loadToken
  selectedEngine.value = unitId
  history.value = null
  try {
    const [detail, compared] = await Promise.all([
      api(`/engines/${unitId}/history`),
      api('/compare', { method: 'POST', body: JSON.stringify({ unit_id: unitId }) }),
    ])
    if (token !== loadToken) return
    history.value = detail
    comparison.value = compared
    if (!detail.features.includes(sensor.value)) sensor.value = detail.features.find(x => x.startsWith('sensor_'))
  } catch (problem) {
    if (token === loadToken) error.value = problem.message
  }
}

function applyRoute() {
  const [, tab = 'fleet', id] = location.hash.split('/')
  activeTab.value = tabs.some(t => t.id === tab) ? tab : 'fleet'
  if (activeTab.value === 'engine' && engines.value.length) {
    const wanted = Number(id) || selectedEngine.value || queue.value[0]
    if (wanted !== selectedEngine.value || !history.value) loadEngine(wanted)
    if (!id) window.history.replaceState(null, '', `#/engine/${wanted}`)
  }
}

async function runSimulation() {
  simBusy.value = true
  error.value = ''
  try {
    simulation.value = await api('/simulate-maintenance', { method: 'POST', body: JSON.stringify(form) })
    simDirty.value = false
  } catch (problem) {
    error.value = problem.message
  } finally {
    simBusy.value = false
  }
}
let suppress = false
async function setForm(values) {
  suppress = true
  Object.assign(form, values)
  Object.assign(form.costs, values.costs)
  await nextTick()
  suppress = false
}
async function resetSimulation() {
  await setForm(structuredClone(baseForm))
  runSimulation()
}

// Recalcula el simulador al cambiar el escenario (con retraso para no saturar la API).
let simTimer
watch(form, () => {
  if (suppress) return
  simDirty.value = true
  clearTimeout(simTimer)
  simTimer = setTimeout(runSimulation, 600)
}, { deep: true })

onMounted(async () => {
  window.addEventListener('hashchange', applyRoute)
  try {
    const [fleet, modelList, summary] = await Promise.all([api('/engines'), api('/models'), api('/evaluation/summary')])
    engines.value = fleet
    models.value = modelList
    evaluation.value = summary
    const base = summary.business_validation
    if (base?.costs_q) {
      await setForm({
        ...BASE_FORM(),
        lead_time_cycles: base.lead_time_cycles,
        costs: {
          intervention_q: base.costs_q.intervention_q,
          failure_q: base.costs_q.failure_q,
          unused_cycle_q: base.costs_q.unused_cycle_q,
          mlp_inference_q: base.inference_q_by_policy?.mlp ?? 0.01,
          lstm_inference_q: base.inference_q_by_policy?.lstm ?? 0.05,
        },
      })
    }
    baseForm = JSON.parse(JSON.stringify(form))
    loading.value = false
    applyRoute()
    // Precarga el motor más urgente para que "Motor" nunca aparezca vacío.
    if (!selectedEngine.value) loadEngine(queue.value[0])
    await runSimulation()
  } catch (problem) {
    error.value = problem.message
  } finally {
    loading.value = false
  }
})
onBeforeUnmount(() => {
  window.removeEventListener('hashchange', applyRoute)
  clearTimeout(simTimer)
})

const current = computed(() => tabs.find(t => t.id === activeTab.value))
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><span class="brand-mark">✦</span><div><strong>PredictGuard</strong><small>Laboratorio FD001</small></div></div>
      <div class="side-label">RECORRIDO</div>
      <nav aria-label="Vistas principales">
        <button v-for="(tab, index) in tabs" :key="tab.id" :class="['nav-item', { active: activeTab === tab.id }]"
          :aria-current="activeTab === tab.id ? 'page' : undefined" @click="go(tab.id)">
          <span class="nav-step">{{ index + 1 }}</span>
          <span><strong>{{ tab.label }}</strong><small>{{ tab.hint }}</small></span>
        </button>
      </nav>
      <div class="side-note"><div class="pulse"></div><div><strong>Datos simulados</strong><span>NASA C-MAPSS · FD001</span></div></div>
      <p class="sidebar-foot">Prototipo académico.<br />No usar para mantenimiento real.</p>
    </aside>

    <div class="main-wrap">
      <header class="topbar">
        <div class="crumbs">PredictGuard / <b>{{ current.label }}</b><template v-if="activeTab === 'engine' && selectedEngine"> / <b>Motor #{{ pad(selectedEngine) }}</b></template></div>
        <div class="topbar-right"><span class="status-dot"></span>{{ models.length ? 'Modelos MLP y LSTM cargados' : 'Conectando…' }}<span>·</span>RUL en ciclos</div>
      </header>
      <main>
        <div v-if="error" class="error-banner" role="alert"><span>{{ error }}</span><button @click="error = ''">Cerrar</button></div>
        <template v-if="loading"><div class="skeleton"></div><div class="skeleton"></div><div class="loading-panel">Cargando motores y modelos…</div></template>
        <FleetView v-else-if="activeTab === 'fleet'" v-model:threshold="threshold" :engines="engines" :evaluation="evaluation" @open="id => go('engine', id)" />
        <EngineView v-else-if="activeTab === 'engine'" v-model:sensor="sensor" :engines="engines" :queue="queue" :history="history" :comparison="comparison" :threshold="threshold" @select="id => go('engine', id)" />
        <ModelsView v-else-if="activeTab === 'lab'" :models="models" :evaluation="evaluation" />
        <SimulatorView v-else :form="form" :simulation="simulation" :busy="simBusy" :dirty="simDirty" @run="runSimulation" @reset="resetSimulation" />
      </main>
    </div>
  </div>
</template>
