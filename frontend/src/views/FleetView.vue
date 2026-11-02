<script setup>
import { computed, ref } from 'vue'
import { LEVELS, fmt, levelOf, minRul, pad } from '../lib/engine.js'

const props = defineProps({
  engines: { type: Array, required: true },
  evaluation: { type: Object, default: null },
})
const threshold = defineModel('threshold', { type: Number, default: 15 })
const emit = defineEmits(['open'])

const search = ref('')
const filter = ref('all')
const sortKey = ref('rul')
const sortDir = ref(1)

const rows = computed(() => props.engines.map(engine => ({
  ...engine,
  rul: minRul(engine.predictions),
  gap: Math.abs(engine.predictions.mlp - engine.predictions.lstm),
  level: levelOf(engine.predictions, threshold.value),
})))
const counts = computed(() => ({
  all: rows.value.length,
  low: rows.value.filter(r => r.level.id === 'low').length,
  watch: rows.value.filter(r => r.level.id === 'watch').length,
  ok: rows.value.filter(r => r.level.id === 'ok').length,
}))
const maxRul = computed(() => Math.max(1, ...rows.value.map(r => Math.max(r.predictions.mlp, r.predictions.lstm))))
const visible = computed(() => {
  const term = search.value.trim()
  const key = sortKey.value
  const value = r => (key === 'rul' ? r.rul : key === 'id' ? r.unit_id : key === 'cycle' ? r.last_observed_cycle : key === 'gap' ? r.gap : r[key])
  return rows.value
    .filter(r => (filter.value === 'all' || r.level.id === filter.value) && String(r.unit_id).includes(term))
    .sort((a, b) => (value(a) - value(b)) * sortDir.value)
})
function sortBy(key) {
  if (sortKey.value === key) sortDir.value *= -1
  else { sortKey.value = key; sortDir.value = 1 }
}
const arrow = key => (sortKey.value === key ? (sortDir.value === 1 ? '↑' : '↓') : '')

// Histograma del RUL mínimo estimado por motor.
const BIN = 20
const hist = computed(() => {
  const bins = Array.from({ length: Math.ceil((maxRul.value + 1) / BIN) }, (_, i) => ({ from: i * BIN, to: (i + 1) * BIN, items: [] }))
  rows.value.forEach(r => bins[Math.min(bins.length - 1, Math.floor(r.rul / BIN))].items.push(r))
  return bins
})
const HW = 520, HH = 190, hp = { l: 30, r: 8, t: 12, b: 28 }
const histMax = computed(() => Math.max(1, ...hist.value.map(b => b.items.length)))
const hx = value => hp.l + (value / (hist.value.length * BIN)) * (HW - hp.l - hp.r)
const hy = count => hp.t + (1 - count / histMax.value) * (HH - hp.t - hp.b)
const barTone = bin => (bin.to <= threshold.value ? 'low' : bin.from >= threshold.value * 2 ? 'ok' : bin.from < threshold.value ? 'low' : 'watch')
const toneColor = { low: '#fb923c', watch: '#f2c75c', ok: '#27bba7' }

const metric = name => props.evaluation?.test?.metrics?.[name]?.mae_cycles
const meanRul = computed(() => rows.value.reduce((s, r) => s + r.rul, 0) / Math.max(1, rows.value.length))
</script>

<template>
  <div class="page-heading">
    <div>
      <div class="eyebrow">PASO 1 · VISTA GENERAL</div>
      <h1>Estado de la flota</h1>
      <p>Cada motor de test se observa en su último ciclo disponible. Ordena por RUL estimado para decidir cuál inspeccionar primero y abre su detalle.</p>
    </div>
    <span class="pill">FD001 · 100 motores de test</span>
  </div>

  <div class="stat-grid">
    <button :class="['stat-card', 'tone-low', { on: filter === 'low' }]" @click="filter = filter === 'low' ? 'all' : 'low'">
      <span>Bajo el umbral</span><strong>{{ counts.low }}</strong><small>RUL ≤ {{ threshold }} ciclos · clic para filtrar</small>
    </button>
    <button :class="['stat-card', 'tone-watch', { on: filter === 'watch' }]" @click="filter = filter === 'watch' ? 'all' : 'watch'">
      <span>Vigilar</span><strong>{{ counts.watch }}</strong><small>RUL entre {{ threshold }} y {{ threshold * 2 }} ciclos</small>
    </button>
    <button :class="['stat-card', 'tone-ok', { on: filter === 'ok' }]" @click="filter = filter === 'ok' ? 'all' : 'ok'">
      <span>Sobre el umbral</span><strong>{{ counts.ok }}</strong><small>RUL &gt; {{ threshold * 2 }} ciclos</small>
    </button>
    <div class="stat-card">
      <span>RUL medio de la flota</span><strong>{{ fmt(meanRul, 1) }}</strong>
      <small>MAE test: MLP {{ fmt(metric('mlp'), 1) }} · LSTM {{ fmt(metric('lstm'), 1) }} ciclos</small>
    </div>
  </div>

  <div class="fleet-grid">
    <section class="panel">
      <div class="panel-head">
        <div><h2>Distribución del RUL estimado</h2><p>Motores por rango de RUL (el menor entre MLP y LSTM), en ciclos.</p></div>
        <label class="inline-control">Umbral de atención
          <input v-model.number="threshold" type="number" min="1" max="200" aria-label="Umbral de atención en ciclos" /> ciclos
        </label>
      </div>
      <div class="panel-body">
        <svg class="hist" :viewBox="`0 0 ${HW} ${HH}`" role="img" aria-label="Histograma de motores por RUL estimado">
          <line :x1="hp.l" :x2="HW - hp.r" :y1="HH - hp.b" :y2="HH - hp.b" stroke="#2f4a63" />
          <g v-for="bin in hist" :key="bin.from">
            <rect :x="hx(bin.from) + 2" :y="hy(bin.items.length)" :width="Math.max(1, hx(bin.to) - hx(bin.from) - 4)"
              :height="HH - hp.b - hy(bin.items.length)" rx="3" :fill="toneColor[barTone(bin)]" opacity=".85" />
            <text v-if="bin.items.length" :x="(hx(bin.from) + hx(bin.to)) / 2" :y="hy(bin.items.length) - 4" text-anchor="middle">{{ bin.items.length }}</text>
            <text :x="hx(bin.from)" :y="HH - 10" text-anchor="middle">{{ bin.from }}</text>
          </g>
          <line :x1="hx(threshold)" :x2="hx(threshold)" :y1="hp.t" :y2="HH - hp.b" stroke="#fb923c" stroke-dasharray="4 3" />
          <text :x="hx(threshold) + 4" :y="hp.t + 8" style="fill:#fb923c">umbral {{ threshold }}</text>
                  </svg>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head"><div><h2>Cómo leer esta pantalla</h2></div></div>
      <div class="panel-body">
        <p>El <strong>RUL</strong> (vida útil restante) se mide en <strong>ciclos de operación</strong>, no en horas ni fechas. Un motor aparece <em>Bajo el umbral</em> si <code>min(RUL MLP, RUL LSTM) ≤ umbral</code>; <em>Vigilar</em> si está entre el umbral y el doble.</p>
        <p class="muted" style="margin-top:12px">El umbral solo ordena la atención en esta pantalla. No cambia las predicciones ni la política del simulador, y no significa que el motor fallará en ese número de ciclos.</p>
        <p class="muted" style="margin-top:12px">La columna <strong>Δ modelos</strong> marca en ámbar los motores donde MLP y LSTM discrepan más de 15 ciclos: conviene revisarlos con más cuidado.</p>
      </div>
    </section>
  </div>

  <section class="panel">
    <div class="panel-head">
      <div><h2>Motores observados</h2><p>{{ visible.length }} de {{ rows.length }} · clic en una fila para abrir el diagnóstico.</p></div>
      <div class="control-row">
        <div class="chips" role="group" aria-label="Filtrar por estado">
          <button :class="['chip', { on: filter === 'all' }]" @click="filter = 'all'">Todos<b>{{ counts.all }}</b></button>
          <button v-for="level in Object.values(LEVELS)" :key="level.id" :class="['chip', { on: filter === level.id }]" @click="filter = level.id">{{ level.label }}<b>{{ counts[level.id] }}</b></button>
        </div>
        <input v-model="search" class="search" type="search" placeholder="Buscar ID…" aria-label="Buscar motor por ID" />
      </div>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th :class="{ sorted: sortKey === 'id' }"><button @click="sortBy('id')">Motor {{ arrow('id') }}</button></th>
            <th :class="{ sorted: sortKey === 'cycle' }"><button @click="sortBy('cycle')">Ciclos observados {{ arrow('cycle') }}</button></th>
            <th :class="{ sorted: sortKey === 'rul' }"><button @click="sortBy('rul')">RUL estimado (mín.) {{ arrow('rul') }}</button></th>
            <th>MLP</th><th>LSTM</th>
            <th :class="{ sorted: sortKey === 'gap' }"><button @click="sortBy('gap')">Δ modelos {{ arrow('gap') }}</button></th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in visible" :key="row.unit_id" style="cursor:pointer" tabindex="0" @click="emit('open', row.unit_id)" @keydown.enter="emit('open', row.unit_id)">
            <td><strong>#{{ pad(row.unit_id) }}</strong></td>
            <td>{{ row.last_observed_cycle }}</td>
            <td>
              <div class="rul-cell">
                <div class="rul-track"><div :class="['rul-fill', `tone-${row.level.tone}`]" :style="{ width: `${(100 * row.rul) / maxRul}%` }"></div></div>
                <b>{{ fmt(row.rul, 1) }}</b>
              </div>
            </td>
            <td>{{ fmt(row.predictions.mlp, 1) }}</td>
            <td>{{ fmt(row.predictions.lstm, 1) }}</td>
            <td><span :class="['diff', { big: row.gap > 15 }]">{{ fmt(row.gap, 1) }}</span></td>
            <td><span :class="['badge', `tone-${row.level.tone}`]">{{ row.level.label }}</span></td>
          </tr>
          <tr v-if="!visible.length"><td colspan="7" style="text-align:center;padding:30px">Ningún motor coincide con el filtro.</td></tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
