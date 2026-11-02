<script setup>
import { computed } from 'vue'
import { fmt, modelName, money } from '../lib/engine.js'

const props = defineProps({
  form: { type: Object, required: true },
  simulation: { type: Object, default: null },
  busy: { type: Boolean, default: false },
  dirty: { type: Boolean, default: false },
})
const emit = defineEmits(['run', 'reset'])

const policies = computed(() => props.simulation?.policies ?? [])
const maxCost = computed(() => Math.max(1, ...policies.value.map(p => p.mean_cost_q_per_engine)))
const cheapest = computed(() => policies.value.reduce((a, b) => (b.mean_cost_q_per_engine < (a?.mean_cost_q_per_engine ?? Infinity) ? b : a), null))
const figureUrl = name => `/api/evaluation/figures/${name}`
</script>

<template>
  <div class="page-heading">
    <div>
      <div class="eyebrow">PASO 4 · CUÁNTO CUESTA DECIDIR</div>
      <h1>Simulador de mantenimiento</h1>
      <p>Compara tres políticas de intervención (por calendario, por MLP y por LSTM) sobre 20 trayectorias completas de validación. Cambia los supuestos y los resultados se recalculan solos.</p>
    </div>
    <span class="pill">Costos hipotéticos en quetzales</span>
  </div>

  <div class="sim-layout">
    <section class="panel">
      <div class="panel-head"><div><h2>Escenario</h2><p>Ningún valor reentrena los modelos; solo cambia cuándo se interviene y cuánto cuesta.</p></div></div>
      <form class="panel-body" @submit.prevent="emit('run')">
        <fieldset>
          <legend>Cuándo intervenir</legend>
          <div class="field-grid">
            <label>Calendario · edad (ciclos)<input v-model.number="form.calendar_age" type="number" min="1" /><small>Interviene al alcanzar esta edad.</small></label>
            <label>MLP · umbral de RUL<input v-model.number="form.mlp_threshold" type="number" min="1" /><small>Interviene si RUL ≤ umbral.</small></label>
            <label>LSTM · umbral de RUL<input v-model.number="form.lstm_threshold" type="number" min="1" /><small>Interviene si RUL ≤ umbral.</small></label>
            <label>Margen de acción (ciclos)<input v-model.number="form.lead_time_cycles" type="number" min="1" /><small>Aviso mínimo para evitar la falla.</small></label>
          </div>
        </fieldset>
        <fieldset>
          <legend>Cuánto cuesta</legend>
          <div class="field-grid">
            <label>Intervención (Q)<input v-model.number="form.costs.intervention_q" type="number" min="0" /></label>
            <label>Falla (Q)<input v-model.number="form.costs.failure_q" type="number" min="0" /></label>
            <label>Vida descartada (Q/ciclo)<input v-model.number="form.costs.unused_cycle_q" type="number" min="0" /></label>
            <label>Inferencia MLP (Q)<input v-model.number="form.costs.mlp_inference_q" type="number" min="0" step="0.01" /></label>
            <label>Inferencia LSTM (Q)<input v-model.number="form.costs.lstm_inference_q" type="number" min="0" step="0.01" /></label>
          </div>
        </fieldset>
        <div class="sim-actions">
          <button class="primary-button" :disabled="busy">{{ busy ? 'Calculando…' : 'Calcular ahora' }}</button>
          <button type="button" class="ghost-button" @click="emit('reset')">Restablecer escenario base</button>
          <span class="sim-status" aria-live="polite">{{ busy ? 'Recalculando…' : dirty ? 'Cambios pendientes' : 'Resultados al día' }}</span>
        </div>
      </form>
    </section>

    <section class="panel">
      <div class="panel-head"><div><h2>Costo medio por motor</h2><p>Mismos 20 motores y mismos costos para las tres políticas. Menor es mejor.</p></div></div>
      <div class="panel-body">
        <div v-if="simulation" class="bars">
          <div v-for="item in policies" :key="item.policy">
            <div class="bar-head">
              <strong>{{ modelName(item.policy) }}<span v-if="item === cheapest" class="badge tone-ok">Menor costo</span></strong>
              <span>{{ money(item.mean_cost_q_per_engine) }}</span>
            </div>
            <div class="bar-track"><div :class="['bar-fill', item.policy]" :style="{ width: `${(100 * item.mean_cost_q_per_engine) / maxCost}%` }"></div></div>
            <div class="bar-foot">
              {{ item.policy === 'calendar' ? 'Edad' : 'Umbral RUL' }} {{ item.threshold_cycles }} ciclos ·
              <span :class="{ bad: item.failures > 0 }">{{ item.failures }} de {{ item.engine_count }} fallas</span> ·
              <template v-if="item.policy === 'calendar'">política de referencia</template>
              <template v-else>ahorro <b>{{ fmt(item.savings_vs_calendar_pct, 1) }} %</b> frente al calendario</template>
            </div>
          </div>
        </div>
        <div v-else class="empty-state">Calculando escenario…</div>
        <div class="info-strip warn" style="margin:20px 0 0"><strong>Cómo leerlo:</strong> una falla cuesta mucho más que una intervención, así que una política que avisa tarde puede salir cara aunque intervenga poco. Los umbrales base se eligieron sobre estos mismos motores, por lo que el ahorro puede ser optimista. Son supuestos, no datos de NASA ni ahorros reales.</div>
      </div>
    </section>
  </div>

  <section class="panel figure-panel">
    <h2>Curva de costo del escenario base</h2>
    <p>Costo medio por motor según el umbral de intervención. Un umbral bajo arriesga avisar tarde y sufrir fallas; uno alto interviene pronto y descarta vida útil.</p>
    <img :src="figureUrl('business_cost_curve.png')" alt="Costo medio por motor según umbral de intervención" />
  </section>
</template>
