<script setup>
import { computed, defineAsyncComponent } from 'vue'
import SensorChart from '../components/SensorChart.vue'
import ModelFlow from '../components/ModelFlow.vue'
import { PARTS, SENSORS, fmt, levelOf, minRul, pad, sensorLabel } from '../lib/engine.js'

const TurbofanView = defineAsyncComponent(() => import('../components/TurbofanView.vue'))

const props = defineProps({
  engines: { type: Array, required: true },
  queue: { type: Array, required: true },
  history: { type: Object, default: null },
  comparison: { type: Object, default: null },
  threshold: { type: Number, default: 15 },
  loading: { type: Boolean, default: false },
})
const sensor = defineModel('sensor', { type: String, default: 'sensor_11' })
const emit = defineEmits(['select', 'back', 'ask'])

const id = computed(() => props.history?.unit_id ?? null)
const observed = computed(() => props.history?.last_observed_cycle ?? 0)
const mlp = computed(() => props.comparison?.predictions?.mlp?.rul_predicted ?? null)
const lstm = computed(() => props.comparison?.predictions?.lstm?.rul_predicted ?? null)
const level = computed(() => (mlp.value === null ? null : levelOf({ mlp: mlp.value, lstm: lstm.value }, props.threshold)))
const gap = computed(() => (mlp.value === null ? 0 : Math.abs(mlp.value - lstm.value)))
const worst = computed(() => (mlp.value === null ? 0 : minRul({ mlp: mlp.value, lstm: lstm.value })))
const sensorKeys = computed(() => (props.history?.features ?? []).filter(key => key.startsWith('sensor_')))
const meta = computed(() => SENSORS[sensor.value] ?? { code: sensor.value, name: sensor.value, unit: '', part: null })

// Escala común para las barras de ciclo de vida.
const scaleMax = computed(() => Math.max(observed.value + Math.max(mlp.value ?? 0, lstm.value ?? 0), 1) * 1.04)
const lifeRows = computed(() => [['MLP', mlp.value], ['LSTM', lstm.value]].filter(([, v]) => v !== null).map(([name, value]) => {
  const total = observed.value + value
  const zone = Math.min(props.threshold, value)
  return {
    name, value, total,
    obs: (100 * observed.value) / scaleMax.value,
    rem: (100 * value) / scaleMax.value,
    zone: (100 * zone) / scaleMax.value,
    zoneLeft: (100 * (total - zone)) / scaleMax.value,
  }
}))

const stats = computed(() => {
  const values = (props.history?.records ?? []).map(r => Number(r[sensor.value]))
  if (values.length < 2) return null
  return { first: values[0], last: values[values.length - 1], min: Math.min(...values), max: Math.max(...values) }
})
const position = computed(() => props.queue.indexOf(id.value))
const prevId = computed(() => (position.value > 0 ? props.queue[position.value - 1] : null))
const nextId = computed(() => (position.value >= 0 && position.value < props.queue.length - 1 ? props.queue[position.value + 1] : null))

const reading = computed(() => {
  if (mlp.value === null) return ''
  const agree = gap.value <= 15
    ? `MLP y LSTM coinciden (diferencia de ${fmt(gap.value, 1)} ciclos).`
    : `MLP y LSTM discrepan en ${fmt(gap.value, 1)} ciclos: trata la estimación con cautela.`
  const where = worst.value <= props.threshold
    ? `La estimación más baja (${fmt(worst.value, 1)}) está por debajo del umbral de ${props.threshold} ciclos.`
    : worst.value <= props.threshold * 2
      ? `La estimación más baja (${fmt(worst.value, 1)}) está entre el umbral (${props.threshold}) y el doble.`
      : `La estimación más baja (${fmt(worst.value, 1)}) supera con holgura el umbral de ${props.threshold} ciclos.`
  return `${where} ${agree}`
})
</script>

<template>
  <div class="page-heading">
    <div>
      <div class="eyebrow">PASO 2 · DIAGNÓSTICO DE UN MOTOR</div>
      <h1>Motor #{{ pad(id ?? queue[0]) }} <span v-if="level" :class="['badge', `tone-${level.tone}`]" style="vertical-align:middle;margin-left:8px">{{ level.label }}</span></h1>
      <p>Qué tan cerca está de su fin de vida según los dos modelos, qué muestran sus señales y en qué parte del motor se miden.</p>
    </div>
    <div class="nav-engine">
      <button class="ghost-button ask-button" :disabled="!history" @click="emit('ask', `Explícame el estado del motor #${pad(id)} y qué haría falta hacer`)">✦ Preguntar al asistente</button>
      <button class="ghost-button" :disabled="prevId === null" @click="emit('select', prevId)">← Más urgente</button>
      <select class="engine-select" :value="id ?? ''" aria-label="Seleccionar motor" @change="emit('select', Number($event.target.value))">
        <option v-for="engine in engines" :key="engine.unit_id" :value="engine.unit_id">Motor #{{ pad(engine.unit_id) }} · RUL {{ fmt(minRul(engine.predictions), 0) }}</option>
      </select>
      <button class="ghost-button" :disabled="nextId === null" @click="emit('select', nextId)">Menos urgente →</button>
    </div>
  </div>

  <div v-if="!history" class="skeleton"></div>
  <template v-else>
    <div class="engine-summary">
      <div class="fact"><span>Ciclos observados</span><strong>{{ observed }}</strong><small>vida ya recorrida en el conjunto</small></div>
      <div class="fact"><span>RUL · MLP</span><strong>{{ fmt(mlp, 1) }}</strong><small>ciclos restantes estimados</small></div>
      <div class="fact"><span>RUL · LSTM</span><strong>{{ fmt(lstm, 1) }}</strong><small>ciclos restantes estimados</small></div>
      <div :class="['fact', level && `tone-${level.tone}`]"><span>Prioridad en la flota</span><strong>{{ position + 1 }}<small style="font-size:14px"> / {{ queue.length }}</small></strong><small>1 = menor RUL estimado</small></div>
    </div>

    <section class="panel">
      <div class="panel-head">
        <div><h2>Ciclo de vida estimado</h2><p>Ciclos ya observados + RUL que cada modelo estima por delante. La franja naranja marca los últimos {{ threshold }} ciclos (umbral de la flota).</p></div>
      </div>
      <div class="panel-body">
        <div class="life">
          <div v-for="row in lifeRows" :key="row.name" class="life-row">
            <b>{{ row.name }}</b>
            <div class="life-bar" :aria-label="`${row.name}: ${observed} ciclos observados más ${fmt(row.value, 1)} restantes`">
              <div class="life-seg obs" :style="{ width: `${row.obs}%` }"></div>
              <div class="life-seg rem" :style="{ left: `${row.obs}%`, width: `${row.rem}%` }"></div>
              <div class="life-seg zone" :style="{ left: `${row.zoneLeft}%`, width: `${row.zone}%` }"></div>
            </div>
            <div class="life-val">vida total ≈ <b>{{ fmt(row.total, 0) }}</b> ciclos</div>
          </div>
        </div>
        <div class="life-legend">
          <span><i style="background:#3b6286"></i>Observado</span>
          <span><i style="background:#1b5f58"></i>RUL estimado</span>
          <span><i style="background:#fb923c55;border-left:1px dashed #fb923c"></i>Zona bajo el umbral</span>
        </div>
        <div class="reading"><strong>Lectura:</strong> {{ reading }} Ambas estimaciones tienden a <strong>sobreestimar</strong> la vida restante (sesgo positivo en test).</div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div><h2>Dónde ocurre: el turbofan</h2><p>Modelo conceptual de un turbofan de dos ejes de alta derivación, el tipo de motor que simula C-MAPSS. Elige una variable y se enfoca la estación donde se mide.</p></div>
        <label class="inline-control">Variable
          <select v-model="sensor" aria-label="Variable del sensor">
            <option v-for="key in sensorKeys" :key="key" :value="key">{{ sensorLabel(key) }}</option>
          </select>
        </label>
      </div>
      <div class="panel-body">
        <TurbofanView v-model:sensor="sensor" :engine-id="id" :observed-cycle="observed" :records="history.records" />
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>{{ meta.code }} · {{ meta.name }}</h2>
          <p>Valor crudo del sensor en cada ciclo observado<span v-if="meta.part"> · estación: {{ PARTS[meta.part].name }}</span>.</p>
        </div>
      </div>
      <div v-if="stats" class="chart-meta">
        <div><span>Ciclo 1</span><strong>{{ fmt(stats.first, 3) }}</strong></div>
        <div><span>Último ciclo</span><strong>{{ fmt(stats.last, 3) }}</strong></div>
        <div><span>Cambio acumulado</span><strong>{{ stats.last - stats.first > 0 ? '+' : '' }}{{ fmt(stats.last - stats.first, 3) }} {{ meta.unit }}</strong></div>
        <div><span>Rango</span><strong>{{ fmt(stats.min, 2) }} – {{ fmt(stats.max, 2) }}</strong></div>
      </div>
      <div class="panel-body"><SensorChart :records="history.records" :sensor="sensor" :unit="meta.unit" /></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div><h2>De las señales a la predicción</h2><p>Ambos modelos reciben las mismas 30 filas × 19 variables (ciclo, 3 ajustes operativos y 15 sensores), no solo el sensor elegido arriba.</p></div>
      </div>
      <div class="panel-body"><ModelFlow :mlp-rul="mlp" :lstm-rul="lstm" /></div>
    </section>

    <div class="info-strip"><strong>Sobre las etiquetas:</strong> el conjunto oficial solo entrega el RUL real del último ciclo de cada motor de test, y esa etiqueta no se muestra aquí. Tampoco se dibujan RUL reales para ciclos anteriores.</div>
    <div class="info-strip warn"><strong>Modelo conceptual:</strong> la geometría ilustra los componentes de un turbofan, no el diseño exacto de un motor NASA. Las variables se asocian a la estación de medición que describe la documentación de C-MAPSS (Saxena et al., 2008), no a un punto físico preciso. Prototipo académico, no apto para decisiones de mantenimiento real.</div>
  </template>
</template>
