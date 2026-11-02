<script setup>
import { computed, ref } from 'vue'
import { fmt } from '../lib/engine.js'

const props = defineProps({
  records: { type: Array, required: true },
  sensor: { type: String, required: true },
  unit: { type: String, default: '' },
})

const W = 820, H = 300
const pad = { l: 64, r: 18, t: 16, b: 44 }
const hover = ref(null)

const series = computed(() => props.records.map(row => ({ cycle: Number(row.cycle), value: Number(row[props.sensor]) })))
const bounds = computed(() => {
  const values = series.value.map(p => p.value)
  let min = Math.min(...values), max = Math.max(...values)
  if (min === max) { min -= 1; max += 1 }
  const margin = (max - min) * 0.08
  return { min: min - margin, max: max + margin }
})
const x = cycle => {
  const first = series.value[0]?.cycle ?? 0
  const last = series.value[series.value.length - 1]?.cycle ?? 1
  return pad.l + ((cycle - first) / Math.max(1, last - first)) * (W - pad.l - pad.r)
}
const y = value => pad.t + (1 - (value - bounds.value.min) / (bounds.value.max - bounds.value.min)) * (H - pad.t - pad.b)
const path = computed(() => series.value.map((p, i) => `${i ? 'L' : 'M'}${x(p.cycle).toFixed(1)},${y(p.value).toFixed(1)}`).join(' '))
const area = computed(() => {
  if (!series.value.length) return ''
  const last = series.value[series.value.length - 1]
  const first = series.value[0]
  return `${path.value} L${x(last.cycle)},${H - pad.b} L${x(first.cycle)},${H - pad.b} Z`
})
const yTicks = computed(() => Array.from({ length: 5 }, (_, i) => {
  const value = bounds.value.min + ((bounds.value.max - bounds.value.min) * i) / 4
  return { value, y: y(value) }
}))
const xTicks = computed(() => {
  const first = series.value[0]?.cycle ?? 1
  const last = series.value[series.value.length - 1]?.cycle ?? 1
  const steps = Math.min(6, Math.max(1, last - first))
  return Array.from({ length: steps + 1 }, (_, i) => {
    const cycle = Math.round(first + ((last - first) * i) / steps)
    return { cycle, x: x(cycle) }
  })
})
const digits = computed(() => (bounds.value.max - bounds.value.min < 5 ? 3 : bounds.value.max - bounds.value.min < 50 ? 2 : 0))

function onMove(event) {
  const rect = event.currentTarget.getBoundingClientRect()
  const px = ((event.clientX - rect.left) / rect.width) * W
  let best = 0
  series.value.forEach((p, i) => {
    if (Math.abs(x(p.cycle) - px) < Math.abs(x(series.value[best].cycle) - px)) best = i
  })
  hover.value = series.value[best]
}
const tipX = computed(() => (hover.value ? Math.min(x(hover.value.cycle) + 10, W - 132) : 0))
</script>

<template>
  <svg v-if="series.length > 1" class="chart-svg" :viewBox="`0 0 ${W} ${H}`" role="img"
    :aria-label="`Trayectoria de ${sensor} por ciclo`" @pointermove="onMove" @pointerleave="hover = null">
    <defs>
      <linearGradient id="sensor-area" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#5eead4" stop-opacity=".28" />
        <stop offset="1" stop-color="#5eead4" stop-opacity="0" />
      </linearGradient>
    </defs>
    <g v-for="tick in yTicks" :key="tick.y">
      <line class="grid" :x1="pad.l" :x2="W - pad.r" :y1="tick.y" :y2="tick.y" />
      <text :x="pad.l - 8" :y="tick.y + 4" text-anchor="end">{{ fmt(tick.value, digits) }}</text>
    </g>
    <text v-for="tick in xTicks" :key="tick.cycle" :x="tick.x" :y="H - 24" text-anchor="middle">{{ tick.cycle }}</text>
    <text :x="(pad.l + W - pad.r) / 2" :y="H - 6" text-anchor="middle">ciclo de operación</text>
    <path :d="area" fill="url(#sensor-area)" />
    <path class="line" :d="path" />
    <g v-if="hover">
      <line :x1="x(hover.cycle)" :x2="x(hover.cycle)" :y1="pad.t" :y2="H - pad.b" stroke="#2f4a63" stroke-dasharray="3 3" />
      <circle :cx="x(hover.cycle)" :cy="y(hover.value)" r="4.5" fill="#5eead4" stroke="#07111f" stroke-width="2" />
      <rect class="chart-tip" :x="tipX" :y="pad.t + 4" width="122" height="38" rx="6" />
      <text class="chart-tip-text" :x="tipX + 9" :y="pad.t + 20">Ciclo {{ hover.cycle }}</text>
      <text class="chart-tip-text" :x="tipX + 9" :y="pad.t + 35">{{ fmt(hover.value, 3) }} {{ unit }}</text>
    </g>
  </svg>
  <div v-else class="empty-state">No hay suficientes ciclos para graficar esta señal.</div>
</template>
