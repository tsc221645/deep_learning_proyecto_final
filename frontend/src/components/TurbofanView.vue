<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { PARTS, PART_ORDER, SENSORS, sensorsOfPart, fmt, pad } from '../lib/engine.js'

const props = defineProps({
  engineId: { type: Number, required: true },
  observedCycle: { type: Number, required: true },
  sensor: { type: String, default: 'sensor_11' },
  records: { type: Array, default: () => [] },
})
const emit = defineEmits(['update:sensor'])

const canvasHost = ref(null)
const cutaway = ref(true)
const exploded = ref(false)
const flow = ref(true)
const spinning = ref(true)
const selectedPart = ref(SENSORS[props.sensor]?.part ?? 'hpc')
const hoverPart = ref(null)

// Seleccionar un sensor en otra parte de la pantalla enfoca su estación en el 3D.
watch(() => props.sensor, key => {
  const part = SENSORS[key]?.part
  if (part) selectedPart.value = part
})

const first = computed(() => props.records[0] ?? null)
const last = computed(() => props.records[props.records.length - 1] ?? null)
const stationSensors = computed(() => sensorsOfPart(selectedPart.value).map(key => {
  const now = last.value ? Number(last.value[key]) : null
  const start = first.value ? Number(first.value[key]) : null
  return { key, ...SENSORS[key], now, delta: now !== null && start !== null ? now - start : null }
}))
const signed = value => (value > 0 ? '+' : '') + fmt(value, 3)

// ---------------------------------------------------------------- escena ---
// Eje del motor = X (admisión a la izquierda, escape a la derecha).
let renderer, scene, camera, controls, raycaster, pointer, resizeObserver, frameId, clock
let envTexture
const cutPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0)
const partRoots = {}   // part -> { root, rotor, stator, center, spool, mats: [] }
const spoolAngle = { lp: 0, hp: 0 }
const baseEmissive = new WeakMap()
let marker, flowPoints, flowState
const disposables = []
let downAt = null
let dragging = false

const interp = (table, x) => {
  if (x <= table[0][0]) return table[0][1]
  for (let i = 1; i < table.length; i += 1) {
    if (x <= table[i][0]) {
      const [x0, y0] = table[i - 1]
      const [x1, y1] = table[i]
      return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0)
    }
  }
  return table[table.length - 1][1]
}

// Perfiles (x, r) del motor. Todo el modelo se deriva de estas tablas.
const NACELLE_OUT = [[-2.72, 0.96], [-2.66, 1.04], [-2.5, 1.1], [-2.15, 1.14], [-1.4, 1.14], [-0.5, 1.09], [0.4, 0.99], [0.95, 0.9]]
const NACELLE_IN = [[0.95, 0.86], [0.4, 0.93], [-0.5, 1.02], [-1.4, 1.05], [-2.0, 1.03], [-2.5, 0.99], [-2.68, 0.975]]
const COWL = [[-1.55, 0.44], [-1.45, 0.47], [-1.2, 0.49], [-0.5, 0.51], [0.4, 0.54], [1.2, 0.57], [1.9, 0.57], [2.3, 0.55], [2.7, 0.44], [3.0, 0.37]]

function partRoot(part, spool = null) {
  if (partRoots[part]) return partRoots[part]
  const root = new THREE.Group()
  const rotor = new THREE.Group()
  const stator = new THREE.Group()
  root.add(rotor, stator)
  scene.add(root)
  partRoots[part] = { root, rotor, stator, spool, mats: [], center: 0 }
  return partRoots[part]
}

function material(part, color, options = {}) {
  const mat = new THREE.MeshStandardMaterial({
    color,
    metalness: options.metalness ?? 0.85,
    roughness: options.roughness ?? 0.32,
    side: THREE.DoubleSide,
    emissive: options.emissive ?? 0x000000,
    emissiveIntensity: options.emissiveIntensity ?? 1,
    transparent: options.opacity !== undefined,
    opacity: options.opacity ?? 1,
  })
  baseEmissive.set(mat, { color: new THREE.Color(options.emissive ?? 0x000000), intensity: options.emissiveIntensity ?? 1 })
  partRoots[part].mats.push(mat)
  disposables.push(mat)
  return mat
}

function tag(object, part) {
  object.userData.part = part
  return object
}

// Cuerpo de revolución alrededor del eje X a partir de pares (x, r).
function lathe(points, segments = 96) {
  const geometry = new THREE.LatheGeometry(points.map(([x, r]) => new THREE.Vector2(r, x)), segments)
  geometry.rotateZ(-Math.PI / 2)
  disposables.push(geometry)
  return geometry
}

const smooth = (table, divisions = 8) => {
  const curve = new THREE.CatmullRomCurve3(table.map(([x, r]) => new THREE.Vector3(x, r, 0)), false, 'centripetal')
  return curve.getPoints(table.length * divisions).map(p => [p.x, p.y])
}

function shell(part, points, color, options = {}) {
  const { stator } = partRoots[part]
  const mesh = tag(new THREE.Mesh(lathe(points), material(part, color, options)), part)
  stator.add(mesh)
  return mesh
}

function cone(part, x0, x1, r0, r1, color, group = 'stator', options = {}) {
  const geometry = lathe([[x0, r0], [x1, r1]], 64)
  const mesh = tag(new THREE.Mesh(geometry, material(part, color, options)), part)
  partRoots[part][group].add(mesh)
  return mesh
}

// Álabe genérico con barrido y ensanche hacia la punta; se instancia por fila.
let bladeGeometry
function getBladeGeometry() {
  if (bladeGeometry) return bladeGeometry
  const shape = new THREE.Shape()
  shape.moveTo(-0.34, 0)
  shape.lineTo(0.34, 0)
  shape.bezierCurveTo(0.42, 0.35, 0.56, 0.75, 0.6, 1)
  shape.bezierCurveTo(0.3, 1.02, -0.1, 0.98, -0.28, 0.9)
  shape.bezierCurveTo(-0.4, 0.6, -0.4, 0.3, -0.34, 0)
  bladeGeometry = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false, curveSegments: 6 })
  bladeGeometry.translate(0, 0, -0.5)
  disposables.push(bladeGeometry)
  return bladeGeometry
}

const m4 = {
  rx: new THREE.Matrix4(), t: new THREE.Matrix4(), ry: new THREE.Matrix4(),
  ty: new THREE.Matrix4(), s: new THREE.Matrix4(), out: new THREE.Matrix4(),
}

// Una fila de álabes: `r0`→`r1` puede variar por fila para estrechar el canal.
function bladeRow(part, group, { x, count, r0, r1, chord, thick, stagger, color, offset = 0, options = {} }) {
  const mesh = tag(new THREE.InstancedMesh(getBladeGeometry(), material(part, color, options), count), part)
  for (let i = 0; i < count; i += 1) {
    const angle = ((i + offset) / count) * Math.PI * 2
    m4.rx.makeRotationX(angle)
    m4.t.makeTranslation(x, 0, 0)
    m4.ry.makeRotationY(stagger)
    m4.ty.makeTranslation(0, r0, 0)
    m4.s.makeScale(chord, r1 - r0, thick)
    m4.out.copy(m4.rx).multiply(m4.t).multiply(m4.ry).multiply(m4.ty).multiply(m4.s)
    mesh.setMatrixAt(i, m4.out)
  }
  mesh.instanceMatrix.needsUpdate = true
  mesh.frustumCulled = false
  partRoots[part][group].add(mesh)
  return mesh
}

function buildEngine() {
  const metals = {
    titanium: 0xd5dde3, steel: 0x9fb1bf, vane: 0x607485, hot: 0xcaa46a, bronze: 0xa99c88,
    case: 0x2f4255, nacelle: 0xc6d0d8, dark: 0x1b2937,
  }
  ;['nacelle', 'fan', 'bypass', 'lpc', 'hpc', 'combustor', 'hpt', 'lpt', 'nozzle'].forEach(part => {
    partRoot(part, { fan: 'lp', lpc: 'lp', lpt: 'lp', hpc: 'hp', hpt: 'hp' }[part] ?? null)
  })
  const centers = { nacelle: -0.9, fan: -2.1, bypass: -0.6, lpc: -1.15, hpc: -0.2, combustor: 0.7, hpt: 1.3, lpt: 2.0, nozzle: 2.9 }
  Object.entries(centers).forEach(([part, x]) => { partRoots[part].center = x })

  // Góndola: labio de entrada, cuerpo y borde de fuga; más un pilón de sujeción.
  const nacelle = [...smooth(NACELLE_OUT, 10), ...smooth(NACELLE_IN, 10)]
  nacelle.push(nacelle[0])
  shell('nacelle', nacelle, metals.nacelle, { metalness: 0.55, roughness: 0.38 })
  const pylon = tag(new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.36, 0.08), material('nacelle', metals.nacelle, { metalness: 0.5, roughness: 0.45 })), 'nacelle')
  disposables.push(pylon.geometry)
  pylon.position.set(-0.3, 1.2, 0)
  partRoots.nacelle.stator.add(pylon)

  // Ventilador: cono (spinner), disco y álabes anchos.
  const spinner = []
  for (let i = 0; i <= 14; i += 1) {
    const t = i / 14
    spinner.push([-2.74 + t * 0.78, 0.34 * Math.sin((t * Math.PI) / 2) ** 0.85])
  }
  const spinnerMesh = tag(new THREE.Mesh(lathe(spinner, 64), material('fan', metals.titanium, { metalness: 0.9, roughness: 0.2 })), 'fan')
  partRoots.fan.rotor.add(spinnerMesh)
  cone('fan', -2.1, -1.88, 0.33, 0.3, metals.steel, 'rotor')
  bladeRow('fan', 'rotor', { x: -2.0, count: 22, r0: 0.3, r1: 1.0, chord: 0.46, thick: 0.03, stagger: 1.0, color: metals.titanium, options: { roughness: 0.22 } })

  // Guías de salida del ventilador y borde de ataque del divisor de flujo.
  bladeRow('bypass', 'stator', { x: -1.15, count: 40, r0: 0.5, r1: 1.04, chord: 0.2, thick: 0.02, stagger: -0.35, color: metals.vane })
  shell('bypass', smooth(COWL.filter(p => p[0] <= 1.95), 8), metals.case, { metalness: 0.7, roughness: 0.42 })

  // Compresor de baja presión (booster): 3 etapas.
  cone('lpc', -1.5, -0.85, 0.25, 0.28, metals.dark, 'rotor')
  for (let i = 0; i < 3; i += 1) {
    const x = -1.38 + i * 0.2
    bladeRow('lpc', 'rotor', { x, count: 46, r0: 0.26, r1: 0.43, chord: 0.12, thick: 0.014, stagger: 0.95, color: metals.titanium })
    bladeRow('lpc', 'stator', { x: x + 0.1, count: 46, r0: 0.27, r1: 0.44, chord: 0.1, thick: 0.012, stagger: -0.95, color: metals.vane })
  }
  cone('lpc', -1.5, -0.78, 0.46, 0.45, metals.case, 'stator', { opacity: 0.9 })

  // Compresor de alta presión: 9 etapas con canal que se estrecha.
  cone('hpc', -0.75, 0.32, 0.2, 0.35, metals.dark, 'rotor')
  for (let i = 0; i < 9; i += 1) {
    const x = -0.68 + i * 0.12
    const hub = 0.21 + (i / 8) * 0.13
    const tip = 0.415 - (i / 8) * 0.02
    bladeRow('hpc', 'rotor', { x, count: 58, r0: hub, r1: tip, chord: 0.085, thick: 0.011, stagger: 0.9, color: metals.steel })
    bladeRow('hpc', 'stator', { x: x + 0.06, count: 58, r0: hub + 0.01, r1: tip + 0.01, chord: 0.075, thick: 0.01, stagger: -0.9, color: metals.vane })
  }
  cone('hpc', -0.78, 0.34, 0.425, 0.395, metals.case, 'stator', { opacity: 0.9 })

  // Cámara de combustión anular: carcasa, tubo de llama incandescente e inyectores.
  cone('combustor', 0.32, 1.12, 0.2, 0.22, metals.dark, 'rotor')
  cone('combustor', 0.3, 1.12, 0.47, 0.45, metals.case, 'stator', { opacity: 0.9 })
  const liner = [[1.1, 0.27], [0.62, 0.27], [0.5, 0.29], [0.44, 0.33], [0.5, 0.375], [0.62, 0.39], [1.1, 0.39]]
  shell('combustor', smooth(liner, 6), 0xff7a2f, {
    metalness: 0.2, roughness: 0.5, emissive: 0xff5a1a, emissiveIntensity: 1.6,
  })
  for (let i = 0; i < 16; i += 1) {
    const angle = (i / 16) * Math.PI * 2
    const injector = tag(new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.03, 0.14, 10), material('combustor', metals.hot)), 'combustor')
    disposables.push(injector.geometry)
    injector.rotation.z = Math.PI / 2
    injector.position.set(0.4, Math.cos(angle) * 0.34, Math.sin(angle) * 0.34)
    partRoots.combustor.stator.add(injector)
  }

  // Turbina de alta presión: 2 etapas con álabes guía previos.
  cone('hpt', 1.1, 1.5, 0.25, 0.27, metals.dark, 'rotor')
  for (let i = 0; i < 2; i += 1) {
    const x = 1.26 + i * 0.2
    bladeRow('hpt', 'stator', { x: x - 0.09, count: 38, r0: 0.26 + i * 0.01, r1: 0.4 + i * 0.01, chord: 0.1, thick: 0.02, stagger: -1.0, color: metals.vane })
    bladeRow('hpt', 'rotor', { x, count: 44, r0: 0.26 + i * 0.01, r1: 0.405 + i * 0.015, chord: 0.1, thick: 0.018, stagger: 0.9, color: metals.hot, options: { roughness: 0.4 } })
  }
  cone('hpt', 1.1, 1.52, 0.44, 0.45, metals.case, 'stator', { opacity: 0.9 })

  // Turbina de baja presión: 4 etapas con canal que se ensancha.
  cone('lpt', 1.55, 2.4, 0.28, 0.32, metals.dark, 'rotor')
  for (let i = 0; i < 4; i += 1) {
    const x = 1.68 + i * 0.2
    const hub = 0.285 + i * 0.012
    const tip = 0.46 + i * 0.03
    bladeRow('lpt', 'stator', { x: x - 0.09, count: 56, r0: hub, r1: tip, chord: 0.1, thick: 0.02, stagger: -0.9, color: metals.vane })
    bladeRow('lpt', 'rotor', { x, count: 64, r0: hub, r1: tip + 0.01, chord: 0.11, thick: 0.018, stagger: 0.85, color: metals.bronze })
  }
  cone('lpt', 1.52, 2.45, 0.465, 0.56, metals.case, 'stator', { opacity: 0.9 })

  // Tobera del núcleo y cono de escape.
  shell('nozzle', smooth(COWL.filter(p => p[0] >= 1.9), 8), metals.case, { metalness: 0.7, roughness: 0.4 })
  const plug = smooth([[2.4, 0.32], [2.7, 0.29], [3.05, 0.2], [3.35, 0.08], [3.52, 0]], 8)
  shell('nozzle', plug, metals.titanium, { metalness: 0.88, roughness: 0.28 })

  // Marcador de estación (anillo) para el componente seleccionado.
  marker = new THREE.Mesh(
    new THREE.TorusGeometry(1, 0.012, 10, 96),
    new THREE.MeshBasicMaterial({ color: 0x5eead4, transparent: true, opacity: 0.9, depthTest: false }),
  )
  marker.rotation.y = Math.PI / 2
  marker.renderOrder = 10
  disposables.push(marker.geometry, marker.material)
  scene.add(marker)

  // Partículas de aire: azul en derivación y compresión, naranja tras la combustión.
  const count = 900
  flowState = Array.from({ length: count }, (_, i) => ({
    lane: i % 3 === 0 ? 'core' : 'bypass', x: -3.2 + Math.random() * 6.8,
    theta: Math.random() * Math.PI * 2, k: Math.random(), speed: 0.7 + Math.random() * 0.8,
  }))
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
  flowPoints = new THREE.Points(geometry, new THREE.PointsMaterial({
    size: 0.03, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false, sizeAttenuation: true,
  }))
  flowPoints.frustumCulled = false
  disposables.push(geometry, flowPoints.material)
  scene.add(flowPoints)
  updateFlow(0)
  refreshHighlight()
  applyLayout()
}

const BLUE = new THREE.Color(0x7fd3ff)
const HOT = new THREE.Color(0xffa24a)
const EMBER = new THREE.Color(0xff6a3a)
const tmpColor = new THREE.Color()

function updateFlow(dt) {
  if (!flowPoints) return
  const pos = flowPoints.geometry.attributes.position
  const col = flowPoints.geometry.attributes.color
  flowState.forEach((p, i) => {
    p.x += dt * p.speed * (p.lane === 'core' && p.x > 0.5 ? 1.5 : 1)
    if (p.x > 3.6) { p.x = -3.2; p.theta = Math.random() * Math.PI * 2 }
    let r
    if (p.lane === 'bypass') {
      const outer = p.x < 0.95 ? interp(NACELLE_IN.slice().reverse(), p.x) : 0.78
      const inner = interp(COWL, p.x) + 0.01
      r = p.x < -2.6 ? 0.2 + p.k * 0.7 : inner + p.k * Math.max(0.05, outer - inner - 0.02)
      if (p.x > 0.95) r = inner + p.k * 0.2 + (p.x - 0.95) * 0.08
    } else {
      const hub = p.x < -1.5 ? 0.15 : p.x < 0.3 ? 0.24 + ((p.x + 0.7) / 1.0) * 0.1 : 0.27
      const tip = p.x < -1.5 ? 0.42 : p.x < 0.3 ? 0.4 : p.x < 1.5 ? 0.4 : 0.42 + (p.x - 1.5) * 0.1
      r = p.x > 2.4 ? interp([[2.4, 0.3], [3.0, 0.3], [3.5, 0.06]], p.x) + p.k * 0.2 : hub + p.k * (tip - hub)
    }
    pos.setXYZ(i, p.x, Math.cos(p.theta) * r, Math.sin(p.theta) * r)
    if (p.lane === 'bypass') tmpColor.copy(BLUE)
    else if (p.x < 0.45) tmpColor.copy(BLUE).lerp(HOT, Math.max(0, (p.x + 0.6) / 1.5))
    else if (p.x < 1.1) tmpColor.copy(HOT)
    else tmpColor.copy(HOT).lerp(EMBER, Math.min(1, (p.x - 1.1) / 1.5))
    col.setXYZ(i, tmpColor.r, tmpColor.g, tmpColor.b)
  })
  pos.needsUpdate = true
  col.needsUpdate = true
}

function refreshHighlight() {
  Object.entries(partRoots).forEach(([part, entry]) => {
    const active = part === selectedPart.value
    const hovered = part === hoverPart.value && !active
    entry.mats.forEach(mat => {
      const base = baseEmissive.get(mat)
      if (active) {
        mat.emissive.set(0x1fbfa8)
        mat.emissiveIntensity = base.intensity > 1 ? 1.6 : 0.5
      } else if (hovered) {
        mat.emissive.set(0x1a6f9e)
        mat.emissiveIntensity = 0.45
      } else {
        mat.emissive.copy(base.color)
        mat.emissiveIntensity = base.intensity
      }
    })
  })
  const entry = partRoots[selectedPart.value]
  if (entry && marker) {
    const radius = { nacelle: 1.18, fan: 1.05, bypass: 0.62, lpc: 0.5, hpc: 0.47, combustor: 0.5, hpt: 0.47, lpt: 0.6, nozzle: 0.5 }[selectedPart.value]
    marker.scale.setScalar(radius)
    marker.userData.x = entry.center
    marker.position.x = entry.center + entry.root.position.x
  }
}

function applyLayout() {
  const spread = exploded.value ? 0.55 : 0
  Object.values(partRoots).forEach(entry => {
    entry.root.position.x = (entry.center - 0.4) * spread
  })
  if (marker) marker.position.x = (marker.userData.x ?? 0) + (partRoots[selectedPart.value]?.root.position.x ?? 0)
}

function applyView() {
  renderer.clippingPlanes = cutaway.value ? [cutPlane] : []
}

function selectPart(part) {
  if (!PARTS[part]) return
  selectedPart.value = part
  const own = sensorsOfPart(part)
  if (own.length && SENSORS[props.sensor]?.part !== part) emit('update:sensor', own[0])
}

function pick(event) {
  const bounds = renderer.domElement.getBoundingClientRect()
  pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1)
  raycaster.setFromCamera(pointer, camera)
  const targets = Object.values(partRoots).map(entry => entry.root)
  const hit = raycaster.intersectObjects(targets, true).find(item => {
    if (!item.object.userData.part) return false
    // El plano de corte oculta la mitad delantera: no se puede seleccionar lo recortado.
    return !cutaway.value || item.point.z <= 0.001
  })
  return hit ? hit.object.userData.part : null
}

function onPointerDown(event) {
  downAt = { x: event.clientX, y: event.clientY }
  dragging = false
}
function onPointerMove(event) {
  if (downAt && Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y) > 5) dragging = true
  if (!downAt) {
    const part = pick(event)
    if (part !== hoverPart.value) {
      hoverPart.value = part
      refreshHighlight()
    }
  }
}
function onPointerUp(event) {
  if (downAt && !dragging) {
    const part = pick(event)
    if (part) selectPart(part)
  }
  downAt = null
}
function onPointerLeave() {
  downAt = null
  if (hoverPart.value) {
    hoverPart.value = null
    refreshHighlight()
  }
}

function onResize() {
  if (!renderer || !camera || !canvasHost.value) return
  const { width, height } = canvasHost.value.getBoundingClientRect()
  if (width <= 0 || height <= 0) return
  renderer.setSize(width, height, false)
  camera.aspect = width / height
  camera.updateProjectionMatrix()
}

function animate() {
  frameId = requestAnimationFrame(animate)
  const dt = Math.min(clock.getDelta(), 0.05)
  if (spinning.value) {
    spoolAngle.lp += dt * 1.6
    spoolAngle.hp += dt * 3.1
    Object.values(partRoots).forEach(entry => {
      if (entry.spool) entry.rotor.rotation.x = spoolAngle[entry.spool]
    })
  }
  flowPoints.visible = flow.value
  if (flow.value) updateFlow(dt)
  marker.material.opacity = 0.55 + 0.4 * Math.sin(clock.elapsedTime * 3)
  controls.update()
  renderer.render(scene, camera)
}

const HOME = { position: [-2.4, 1.7, 5.3], target: [0.1, 0, 0] }
function resetView() {
  camera.position.set(...HOME.position)
  controls.target.set(...HOME.target)
  controls.update()
}

watch([selectedPart, hoverPart], refreshHighlight)
watch(exploded, applyLayout)
watch(cutaway, () => renderer && applyView())

onMounted(() => {
  scene = new THREE.Scene()
  clock = new THREE.Clock()
  camera = new THREE.PerspectiveCamera(34, 1, 0.1, 60)
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.05
  renderer.setClearColor(0x08131f, 1)
  renderer.domElement.setAttribute('aria-label', 'Modelo 3D interactivo de un turbofan de dos ejes')
  renderer.domElement.addEventListener('pointerdown', onPointerDown)
  renderer.domElement.addEventListener('pointermove', onPointerMove)
  renderer.domElement.addEventListener('pointerup', onPointerUp)
  renderer.domElement.addEventListener('pointerleave', onPointerLeave)
  canvasHost.value.appendChild(renderer.domElement)

  const pmrem = new THREE.PMREMGenerator(renderer)
  envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
  scene.environment = envTexture
  scene.environmentIntensity = 0.9
  pmrem.dispose()

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.minDistance = 2.5
  controls.maxDistance = 14
  controls.maxPolarAngle = Math.PI * 0.92

  scene.add(new THREE.HemisphereLight(0xdcefff, 0x1b2a38, 0.9))
  const key = new THREE.DirectionalLight(0xffffff, 2.2)
  key.position.set(-3, 5, 6)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0x5eead4, 1.1)
  rim.position.set(4, 2, -5)
  scene.add(rim)
  const glow = new THREE.PointLight(0xff7a2f, 6, 3.2)
  glow.position.set(0.75, 0, 0)
  scene.add(glow)

  const grid = new THREE.GridHelper(16, 32, 0x2f5a6a, 0x16293a)
  grid.position.y = -1.45
  scene.add(grid)

  raycaster = new THREE.Raycaster()
  pointer = new THREE.Vector2()
  buildEngine()
  applyView()
  resetView()
  resizeObserver = new ResizeObserver(onResize)
  resizeObserver.observe(canvasHost.value)
  onResize()
  animate()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(frameId)
  resizeObserver?.disconnect()
  controls?.dispose()
  disposables.forEach(item => item.dispose?.())
  disposables.length = 0
  envTexture?.dispose()
  if (renderer) {
    const canvas = renderer.domElement
    canvas.removeEventListener('pointerdown', onPointerDown)
    canvas.removeEventListener('pointermove', onPointerMove)
    canvas.removeEventListener('pointerup', onPointerUp)
    canvas.removeEventListener('pointerleave', onPointerLeave)
    renderer.dispose()
    canvas.remove()
  }
  Object.keys(partRoots).forEach(key => delete partRoots[key])
})
</script>

<template>
  <div class="tf">
    <div class="tf-toolbar">
      <div class="seg" role="group" aria-label="Tipo de vista">
        <button :class="{ on: cutaway }" @click="cutaway = true">Corte</button>
        <button :class="{ on: !cutaway }" @click="cutaway = false">Exterior</button>
      </div>
      <div class="tf-toggles">
        <button :class="['chip-toggle', { on: exploded }]" :aria-pressed="exploded" @click="exploded = !exploded">Despiece</button>
        <button :class="['chip-toggle', { on: flow }]" :aria-pressed="flow" @click="flow = !flow">Flujo de aire</button>
        <button :class="['chip-toggle', { on: spinning }]" :aria-pressed="spinning" @click="spinning = !spinning">Rotación</button>
        <button class="chip-toggle" @click="resetView">↺ Vista</button>
      </div>
    </div>

    <div class="tf-stage">
      <div ref="canvasHost" class="tf-canvas"></div>
      <div class="tf-overlay"><span class="dot"></span>MOTOR #{{ pad(engineId) }}<i></i>CICLO {{ observedCycle }}</div>
      <div class="tf-hover" v-if="hoverPart">{{ PARTS[hoverPart].name }}</div>
      <div class="tf-legend" aria-hidden="true">
        <span><b class="sw cool"></b>Aire frío</span><span><b class="sw hot"></b>Gases calientes</span>
        <span><b class="sw lp"></b>Eje LP: fan · LPC · LPT</span><span><b class="sw hp"></b>Eje HP: HPC · HPT</span>
      </div>
      <div class="tf-hint">Arrastra para girar · rueda para acercar · clic en un componente</div>
    </div>

    <div class="tf-parts" role="tablist" aria-label="Componentes del turbofan">
      <button v-for="part in PART_ORDER" :key="part" role="tab" :aria-selected="selectedPart === part"
        :class="{ on: selectedPart === part }" @click="selectPart(part)">{{ PARTS[part].short }}</button>
    </div>

    <div class="tf-detail">
      <div>
        <span class="kicker">COMPONENTE SELECCIONADO</span>
        <h3>{{ PARTS[selectedPart].name }}</h3>
        <p>{{ PARTS[selectedPart].desc }}</p>
      </div>
      <div class="tf-stations">
        <span class="kicker">VARIABLES C-MAPSS MEDIDAS EN ESTA ESTACIÓN</span>
        <div v-if="stationSensors.length" class="station-list">
          <button v-for="item in stationSensors" :key="item.key" :class="['station', { on: item.key === sensor }]"
            @click="emit('update:sensor', item.key)">
            <b>{{ item.code }}</b>
            <span>{{ item.name }}</span>
            <em>{{ item.now === null ? '—' : fmt(item.now, 3) }} {{ item.unit }}
              <small v-if="item.delta !== null">{{ signed(item.delta) }} desde el ciclo 1</small></em>
          </button>
        </div>
        <p v-else class="muted">C-MAPSS no incluye variables de este componente entre las señales del modelo.</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tf { border: 1px solid var(--line); border-radius: 14px; background: #08131f; overflow: hidden; }
.tf-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; padding: 12px 14px; border-bottom: 1px solid var(--line); background: var(--surface-2); }
.seg { display: flex; padding: 3px; background: var(--bg-2); border: 1px solid var(--line); border-radius: 9px; }
.seg button { border: 0; background: transparent; color: var(--muted); border-radius: 7px; padding: 7px 14px; font-size: 12px; font-weight: 600; }
.seg button.on { background: var(--teal-bg); color: var(--teal); }
.tf-toggles { display: flex; gap: 6px; flex-wrap: wrap; }
.chip-toggle { border: 1px solid var(--line); background: var(--bg-2); color: var(--muted); border-radius: 99px; padding: 6px 12px; font-size: 12px; font-weight: 500; }
.chip-toggle:hover { color: var(--text); border-color: var(--line-2); }
.chip-toggle.on { background: var(--teal-bg); border-color: var(--teal-line); color: var(--teal); }
.tf-stage { position: relative; }
.tf-canvas { height: 460px; touch-action: none; }
.tf-canvas :deep(canvas) { display: block; width: 100%; height: 100%; outline: none; }
.tf-overlay { position: absolute; top: 12px; left: 14px; display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 700; letter-spacing: 1px; color: var(--muted); pointer-events: none; }
.tf-overlay .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 0 3px #5eead433; }
.tf-overlay i { width: 1px; height: 12px; background: var(--line-2); }
.tf-hover { position: absolute; top: 12px; right: 14px; padding: 6px 11px; border-radius: 8px; background: #0b1c2cee; border: 1px solid var(--line-2); font-size: 12px; pointer-events: none; }
.tf-legend { position: absolute; left: 14px; bottom: 36px; display: grid; gap: 4px; font-size: 11px; color: var(--muted); pointer-events: none; }
.tf-legend span { display: flex; align-items: center; gap: 7px; }
.sw { display: inline-block; width: 18px; height: 4px; border-radius: 2px; }
.sw.cool { background: #7fd3ff; } .sw.hot { background: #ffa24a; } .sw.lp { background: #d5dde3; } .sw.hp { background: #9fb1bf; }
.tf-hint { position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: 11px; color: var(--subtle); pointer-events: none; }
.tf-parts { display: flex; gap: 6px; flex-wrap: wrap; padding: 12px 14px; border-top: 1px solid var(--line); }
.tf-parts button { border: 1px solid var(--line); background: var(--bg-2); color: var(--muted); border-radius: 8px; padding: 7px 12px; font-size: 12px; font-weight: 500; }
.tf-parts button:hover { color: var(--text); border-color: var(--line-2); }
.tf-parts button.on { background: var(--teal-bg); border-color: var(--teal-line); color: var(--teal); }
.tf-detail { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr); gap: 22px; padding: 16px 18px 20px; border-top: 1px solid var(--line); background: var(--surface-2); }
.kicker { display: block; color: var(--teal); font-size: 10px; font-weight: 700; letter-spacing: 1.4px; }
.tf-detail h3 { margin: 6px 0 6px; font-size: 17px; }
.tf-detail p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
.station-list { display: grid; gap: 6px; margin-top: 8px; }
.station { display: grid; grid-template-columns: 62px 1fr auto; align-items: center; gap: 10px; text-align: left; padding: 8px 11px; border: 1px solid var(--line); border-radius: 8px; background: var(--bg-2); color: var(--text); font-size: 12px; }
.station:hover { border-color: var(--line-2); }
.station.on { border-color: var(--teal-line); background: var(--teal-bg); }
.station b { color: var(--teal); font-weight: 700; }
.station span { color: var(--muted); }
.station em { font-style: normal; text-align: right; font-variant-numeric: tabular-nums; }
.station small { display: block; color: var(--subtle); font-size: 10px; }
.muted { color: var(--subtle); }
@media (max-width: 820px) {
  .tf-canvas { height: 340px; }
  .tf-detail { grid-template-columns: 1fr; }
  .tf-legend, .tf-hint { display: none; }
  .station { grid-template-columns: 56px 1fr; }
  .station em { grid-column: 2; text-align: left; }
}
</style>
