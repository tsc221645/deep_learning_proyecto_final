<script setup>
import { computed } from 'vue'
import { fmt, modelName } from '../lib/engine.js'

const props = defineProps({
  models: { type: Array, required: true },
  evaluation: { type: Object, default: null },
})

const figureUrl = name => `/api/evaluation/figures/${name}`
const test = name => props.evaluation?.test?.metrics?.[name] ?? {}
const ci = computed(() => props.evaluation?.test?.bootstrap_95pct_mae_cycles?.lstm_minus_mlp_mae ?? null)
const best = computed(() => (test('lstm').mae_cycles <= test('mlp').mae_cycles ? 'lstm' : 'mlp'))

const rows = [
  { key: 'mae_cycles', label: 'MAE test', unit: 'ciclos' },
  { key: 'rmse_cycles', label: 'RMSE test', unit: 'ciclos' },
  { key: 'nasa_phm_score_sum', label: 'Score NASA', unit: '' },
  { key: 'bias_cycles', label: 'Sesgo', unit: 'ciclos' },
]
const maxOf = key => Math.max(1e-9, Math.abs(test('mlp')[key] ?? 0), Math.abs(test('lstm')[key] ?? 0))
const pct = (name, key) => `${(100 * Math.abs(test(name)[key] ?? 0)) / maxOf(key)}%`
const valMae = name => props.models.find(m => m.name === name)?.validation_mae_cycles

// Eje del intervalo bootstrap: simétrico alrededor de cero.
const span = computed(() => (ci.value ? Math.max(Math.abs(ci.value[0]), Math.abs(ci.value[1])) * 1.35 : 1))
const pos = value => `${50 + (50 * value) / span.value}%`
</script>

<template>
  <div class="page-heading">
    <div>
      <div class="eyebrow">PASO 3 · CUÁNTO CONFIAR EN LOS MODELOS</div>
      <h1>Comparar modelos</h1>
      <p>Resultados congelados sobre los 100 motores de test (una observación terminal por motor, con la etiqueta oficial). Menor es mejor en todas las métricas de error.</p>
    </div>
    <span class="pill">Evaluación final</span>
  </div>

  <div class="model-grid">
    <article v-for="model in models" :key="model.name" :class="['model-card', { best: model.name === best }]">
      <div class="model-head">
        <span class="model-symbol">{{ model.name === 'mlp' ? 'M' : 'L' }}</span>
        <div><h2>{{ modelName(model.name) }}</h2><small>versión {{ model.version }} · ventana de {{ model.window_size }} ciclos</small></div>
        <span v-if="model.name === best" class="badge tone-ok">Menor MAE</span>
      </div>
      <div v-for="row in rows" :key="row.key" class="metric-row">
        <span>{{ row.label }}</span>
        <div class="metric-track"><div :class="['metric-fill', model.name]" :style="{ width: pct(model.name, row.key) }"></div></div>
        <b>{{ fmt(test(model.name)[row.key], row.key === 'nasa_phm_score_sum' ? 0 : 2) }}</b>
      </div>
      <div class="metric-row"><span>MAE validación</span><div></div><b>{{ fmt(valMae(model.name)) }}</b></div>
    </article>
  </div>

  <section class="panel">
    <div class="panel-head">
      <div><h2>¿La diferencia es real?</h2><p>Intervalo bootstrap 95 % (por motor) de MAE<sub>LSTM</sub> − MAE<sub>MLP</sub>. Si incluye cero, no hay evidencia sólida de que un modelo supere al otro en MAE.</p></div>
    </div>
    <div v-if="ci" class="ci" aria-label="Intervalo de confianza de la diferencia de MAE">
      <div class="ci-axis"></div>
      <div class="ci-bar" :style="{ left: pos(ci[0]), width: `calc(${pos(ci[1])} - ${pos(ci[0])})` }"></div>
      <div class="ci-zero" :style="{ left: '50%' }"><span>0 = empate</span></div>
    </div>
    <div v-if="ci" class="ci-label">
      <span>{{ fmt(ci[0]) }} ciclos (LSTM mejor)</span>
      <strong>Intervalo: {{ fmt(ci[0]) }} a {{ fmt(ci[1]) }}{{ ci[0] < 0 && ci[1] > 0 ? ' · incluye cero' : '' }}</strong>
      <span>+{{ fmt(ci[1]) }} ciclos (MLP mejor)</span>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><div><h2>Glosario de métricas</h2></div></div>
    <div class="glossary">
      <div><b>MAE</b>Error absoluto medio entre RUL predicho y real, en ciclos.</div>
      <div><b>RMSE</b>Como el MAE pero castiga más los errores grandes.</div>
      <div><b>Score NASA (PHM08)</b>Penalización asimétrica acumulada: sobreestimar la vida restante cuesta más que subestimarla.</div>
      <div><b>Sesgo</b>Promedio de predicho − real. Positivo = tiende a sobreestimar la vida restante (optimista).</div>
    </div>
  </section>

  <div class="figure-grid">
    <section class="panel figure-panel wide"><h2>Predicción y residuales en test</h2><p>Cada punto es un motor de test, comparado con su etiqueta oficial.</p><img :src="figureUrl('test_comparison.png')" alt="Predicción y residuales de MLP y LSTM en test" /></section>
    <section class="panel figure-panel"><h2>Entrenamiento · MLP</h2><p>Pérdida de entrenamiento y validación por época.</p><img :src="figureUrl('mlp_learning_curves.png')" alt="Curvas de aprendizaje MLP" /></section>
    <section class="panel figure-panel"><h2>Entrenamiento · LSTM</h2><p>Pérdida de entrenamiento y validación por época.</p><img :src="figureUrl('lstm_learning_curves.png')" alt="Curvas de aprendizaje LSTM" /></section>
  </div>

  <div class="info-strip warn"><strong>Conclusión prudente:</strong> la LSTM tuvo menor MAE y mucho menor Score NASA en test, pero en validación el MLP fue ligeramente mejor y la diferencia de MAE en test no es estadísticamente concluyente. Ambos pueden sobreestimar la vida restante.</div>
</template>
