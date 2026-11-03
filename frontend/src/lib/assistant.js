// Estado y cliente del asistente de mantenimiento (chat con streaming NDJSON).
// Vive a nivel de módulo para que la conversación sobreviva al cambiar de pestaña.
import { reactive } from 'vue'

export const chat = reactive({ messages: [], busy: false, status: null })

let controller = null

export async function loadStatus() {
  try {
    const response = await fetch('/api/assistant/status')
    chat.status = response.ok ? await response.json() : { available: false, mode: 'offline' }
  } catch {
    chat.status = { available: false, mode: 'offline' }
  }
}

export function resetChat() {
  stopChat()
  chat.messages.splice(0)
}

export function stopChat() {
  if (controller) controller.abort()
  controller = null
  chat.busy = false
}

export async function sendMessage(text, { unitId = null, threshold = 15 } = {}) {
  const content = text.trim()
  if (!content || chat.busy) return
  const history = chat.messages.filter(m => m.content && !m.error).map(m => ({ role: m.role, content: m.content }))
  history.push({ role: 'user', content })
  chat.messages.push({ role: 'user', content })
  const reply = reactive({ role: 'assistant', content: '', tools: [], source: null, model: null, notice: '', error: '', pending: true })
  chat.messages.push(reply)
  chat.busy = true
  controller = new AbortController()
  try {
    const response = await fetch('/api/assistant/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({ messages: history, unit_id: unitId, threshold }),
    })
    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => ({}))
      throw new Error(typeof payload.detail === 'string' ? payload.detail : `Error ${response.status}`)
    }
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let index
      while ((index = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, index).trim()
        buffer = buffer.slice(index + 1)
        if (line) apply(reply, JSON.parse(line))
      }
    }
  } catch (problem) {
    if (problem.name !== 'AbortError') reply.error = problem.message || 'No se pudo contactar al asistente.'
  } finally {
    reply.pending = false
    chat.busy = false
    controller = null
  }
}

function apply(reply, event) {
  if (event.type === 'text') reply.content += event.delta
  else if (event.type === 'tool') reply.tools.push({ label: event.label, name: event.name })
  else if (event.type === 'notice') reply.notice = event.message
  else if (event.type === 'error') reply.error = event.message
  else if (event.type === 'done') { reply.source = event.source; reply.model = event.model ?? null }
}

// ---- Markdown mínimo y seguro: se escapa todo el HTML antes de dar formato ------
const escapeHtml = value => value.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
const inline = value => escapeHtml(value)
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/`(.+?)`/g, '<code>$1</code>')
  .replace(/(^|[\s(])_(.+?)_(?=[\s).,;:]|$)/g, '$1<em>$2</em>')

export function renderMarkdown(source) {
  const out = []
  let list = null
  let table = null
  const close = () => { if (list) { out.push(`</${list}>`); list = null } }
  const flushTable = () => {
    if (!table) return
    const [head, ...rows] = table
    const cells = (row, tag) => `<tr>${row.map(c => `<${tag}>${inline(c)}</${tag}>`).join('')}</tr>`
    out.push(`<div class="md-table"><table><thead>${cells(head, 'th')}</thead><tbody>${rows.map(r => cells(r, 'td')).join('')}</tbody></table></div>`)
    table = null
  }
  for (const raw of (source || '').split('\n')) {
    const line = raw.trimEnd()
    if (/^\s*\|.*\|\s*$/.test(line)) {
      close()
      const cols = line.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim())
      if (cols.every(c => /^:?-{2,}:?$/.test(c))) continue
      ;(table ||= []).push(cols)
      continue
    }
    flushTable()
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/)
    const heading = line.match(/^#{1,4}\s+(.*)$/)
    if (bullet || numbered) {
      const tag = bullet ? 'ul' : 'ol'
      if (list !== tag) { close(); out.push(`<${tag}>`); list = tag }
      out.push(`<li>${inline((bullet || numbered)[1])}</li>`)
    } else {
      close()
      if (heading) out.push(`<p><strong>${inline(heading[1])}</strong></p>`)
      else if (line.trim()) out.push(`<p>${inline(line)}</p>`)
    }
  }
  flushTable()
  close()
  return out.join('')
}
