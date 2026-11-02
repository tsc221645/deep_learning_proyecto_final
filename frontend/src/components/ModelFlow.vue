<script setup>
import { fmt } from '../lib/engine.js'

defineProps({
  mlpRul: { type: Number, default: null },
  lstmRul: { type: Number, default: null },
  window: { type: Number, default: 30 },
  features: { type: Number, default: 19 },
})
</script>

<template>
  <div class="flow-grid">
    <article class="flow-card">
      <h3>MLP <span class="pill">aplana la ventana</span></h3>
      <p class="sub">Trata la ventana como un vector único: no distingue el orden de los ciclos.</p>
      <div class="flow-steps">
        <div class="flow-step"><strong>Ventana</strong><small>{{ window }} ciclos × {{ features }} señales</small></div>
        <span class="flow-arrow">→</span>
        <div class="flow-step"><strong>Aplanar</strong><small>{{ window * features }} entradas</small></div>
        <span class="flow-arrow">→</span>
        <div class="flow-step"><div class="dots"><i v-for="n in 7" :key="n"></i></div><strong>128 · 64</strong><small>capas densas ReLU</small></div>
        <span class="flow-arrow">→</span>
        <div class="flow-step out"><strong>{{ mlpRul === null ? '—' : fmt(mlpRul, 1) }}</strong><small>RUL (ciclos)</small></div>
      </div>
    </article>
    <article class="flow-card">
      <h3>LSTM <span class="pill">lee la secuencia</span></h3>
      <p class="sub">Recorre los ciclos en orden y conserva memoria de la tendencia reciente.</p>
      <div class="flow-steps">
        <div class="flow-step"><strong>Secuencia</strong><small>{{ window }} pasos de {{ features }} señales</small></div>
        <span class="flow-arrow">→</span>
        <div class="flow-step"><div class="dots"><i v-for="n in 5" :key="n"></i></div><strong>LSTM</strong><small>2 capas · 64 unidades</small></div>
        <span class="flow-arrow">→</span>
        <div class="flow-step"><strong>Último estado</strong><small>resume la ventana</small></div>
        <span class="flow-arrow">→</span>
        <div class="flow-step out"><strong>{{ lstmRul === null ? '—' : fmt(lstmRul, 1) }}</strong><small>RUL (ciclos)</small></div>
      </div>
    </article>
  </div>
</template>
