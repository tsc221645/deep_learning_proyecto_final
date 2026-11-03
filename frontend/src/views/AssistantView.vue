<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { chat, loadStatus, renderMarkdown, resetChat, sendMessage, stopChat } from '../lib/assistant.js'
import { fmt, minRul, pad } from '../lib/engine.js'

const props = defineProps({
  engines: { type: Array, required: true },
  selectedEngine: { type: Number, default: null },
  pending: { type: String, default: '' },
})
const threshold = defineModel('threshold', { type: Number, default: 15 })
const emit = defineEmits(['consumed'])

const unit = ref(props.selectedEngine)
const draft = ref('')
const scroller = ref(null)
const input = ref(null)

const sources = {
  claude: 'Generado por Claude con datos de los modelos',
  cache: 'Respuesta en caché (generada por Claude)',
  offline: 'Resumen predefinido, sin LLM',
}

const suggestions = computed(() => {
  const id = unit.value
  const list = [
    '¿Qué motores debo atender primero y por qué?',
    '¿Conviene más el mantenimiento predictivo que el calendario?',
    '¿Qué es el score NASA y por qué la LSTM sale mejor en él?',
    '¿Por qué los modelos tienden a sobreestimar la vida restante?',
    'Si una falla costara Q100,000, ¿cambia la conclusión?',
  ]
  if (id) list.unshift(`Explícame el estado del motor #${pad(id)} y qué haría falta hacer`)
  return list.slice(0, 6)
})
const ready = computed(() => chat.status?.available)
const unitTag = computed(() => {
  const engine = props.engines.find(e => e.unit_id === unit.value)
  return engine ? `RUL ${fmt(minRul(engine.predictions), 0)}` : ''
})

function submit(text = draft.value) {
  if (!text.trim() || chat.busy) return
  draft.value = ''
  sendMessage(text, { unitId: unit.value, threshold: threshold.value })
}
function onKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit() }
}
function scrollDown() {
  nextTick(() => { if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight })
}

watch(() => chat.messages.map(m => m.content.length).join(','), scrollDown)
watch(() => props.selectedEngine, value => { if (value) unit.value = value })
onMounted(async () => {
  if (!chat.status) loadStatus()
  if (props.pending) { const text = props.pending; emit('consumed'); submit(text) }
  scrollDown()
  input.value?.focus()
})
</script>

<template>
  <div class="page-heading">
    <div>
      <div class="eyebrow">PASO 5 · ASISTENTE DE MANTENIMIENTO</div>
      <h1>Pregúntale a los datos</h1>
      <p>Un asistente basado en un LLM que consulta los modelos MLP y LSTM, la flota y el simulador de costos para explicar el estado de los motores, justificar decisiones y decir qué haría falta hacer.</p>
    </div>
    <span :class="['pill', 'assistant-pill', { off: chat.status && !ready }]">
      <i></i>{{ !chat.status ? 'Conectando…' : ready ? `Claude · ${chat.status.model}` : 'Sin llave de API · modo sin conexión' }}
    </span>
  </div>

  <div class="assistant">
    <div class="assistant-context">
      <label class="inline-control">Motor en contexto
        <select v-model.number="unit" aria-label="Motor en contexto">
          <option :value="null">Ninguno (toda la flota)</option>
          <option v-for="engine in engines" :key="engine.unit_id" :value="engine.unit_id">Motor #{{ pad(engine.unit_id) }} · RUL {{ fmt(minRul(engine.predictions), 0) }}</option>
        </select>
      </label>
      <label class="inline-control">Umbral
        <input v-model.number="threshold" type="number" min="1" max="1000" aria-label="Umbral en ciclos" /> ciclos
      </label>
      <span v-if="unitTag" class="muted ctx-note">El asistente verá el motor #{{ pad(unit) }} ({{ unitTag }}).</span>
      <button class="ghost-button" style="margin-left:auto" :disabled="!chat.messages.length" @click="resetChat">Nueva conversación</button>
    </div>

    <div ref="scroller" class="chat-scroll" aria-live="polite">
      <div v-if="!chat.messages.length" class="chat-empty">
        <div class="chat-orb">✦</div>
        <h2>¿Qué quieres saber?</h2>
        <p>Puedo explicar un motor, priorizar la flota, comparar el mantenimiento predictivo contra el calendario o aclarar cualquier métrica. Cada cifra sale de los modelos reales, no de mi memoria.</p>
        <div class="suggestions">
          <button v-for="text in suggestions" :key="text" class="suggestion" @click="submit(text)">{{ text }}</button>
        </div>
      </div>

      <template v-else>
        <div v-for="(message, index) in chat.messages" :key="index" :class="['msg', message.role]">
          <div class="avatar">{{ message.role === 'user' ? 'Tú' : '✦' }}</div>
          <div class="bubble">
            <div v-if="message.tools?.length" class="tool-trail">
              <span v-for="(tool, i) in message.tools" :key="i" class="tool-chip">✓ {{ tool.label }}</span>
            </div>
            <div v-if="message.notice" class="chat-notice">{{ message.notice }}</div>
            <div v-if="message.role === 'user'" class="user-text">{{ message.content }}</div>
            <div v-else-if="message.content" class="md" v-html="renderMarkdown(message.content)"></div>
            <div v-if="message.pending && !message.content" class="typing"><i></i><i></i><i></i><span v-if="message.tools?.length">Analizando los resultados…</span></div>
            <div v-if="message.error" class="chat-error">{{ message.error }}</div>
            <div v-if="message.source" class="msg-foot">{{ sources[message.source] }}<template v-if="message.source !== 'offline' && message.model"> · {{ message.model }}</template></div>
          </div>
        </div>
      </template>
    </div>

    <form class="composer" @submit.prevent="submit()">
      <textarea ref="input" v-model="draft" rows="1" :placeholder="unit ? `Pregunta sobre el motor #${pad(unit)} o la flota…` : 'Pregunta sobre la flota, los modelos o los costos…'"
        aria-label="Mensaje para el asistente" maxlength="2000" @keydown="onKey"></textarea>
      <button v-if="chat.busy" type="button" class="ghost-button" @click="stopChat">Detener</button>
      <button v-else type="submit" class="primary-button" :disabled="!draft.trim()">Enviar</button>
    </form>
    <p class="chat-disclaimer">Datos simulados (NASA C-MAPSS FD001) y costos hipotéticos. El asistente puede equivocarse: no sustituye una inspección real.</p>
  </div>
</template>
