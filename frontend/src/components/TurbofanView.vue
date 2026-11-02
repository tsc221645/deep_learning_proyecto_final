<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const props = defineProps({
  engineId: { type: Number, required: true },
  observedCycle: { type: Number, required: true },
  mlpRul: { type: Number, default: null },
  lstmRul: { type: Number, default: null },
  sensor: { type: String, default: 'sensor_11' },
  sensorValue: { type: Number, default: null },
})

const canvasHost = ref(null)
const cutaway = ref(true)
const selectedPart = ref('fan')
const partNames = {
  fan: 'Ventilador',
  compressor: 'Compresor',
  combustion: 'Cámara de combustión',
  turbine: 'Turbina',
  exhaust: 'Tobera de escape',
  casing: 'Carcasa',
}
const detail = {
  fan: 'Aumenta la presión y el flujo de aire que entra al motor.',
  compressor: 'Comprime el aire antes de la combustión.',
  combustion: 'Mezcla el aire comprimido con combustible y libera energía.',
  turbine: 'Extrae energía de los gases para mover el compresor y el ventilador.',
  exhaust: 'Acelera los gases para producir empuje.',
  casing: 'Protege y contiene los componentes internos.',
}

let renderer
let scene
let camera
let controls
let resizeObserver
let frameId
let engineGroup
let casing
let internalParts = []
let raycaster
let pointer

function addPart(geometry, material, position, part, parent = engineGroup) {
  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.set(...position)
  mesh.userData.part = part
  mesh.castShadow = true
  mesh.receiveShadow = true
  parent.add(mesh)
  return mesh
}

function axialCylinder(radius, length, material, x, part, segments = 48) {
  const mesh = addPart(
    new THREE.CylinderGeometry(radius, radius, length, segments, 1, true),
    material,
    [x, 0, 0],
    part,
  )
  mesh.rotation.z = Math.PI / 2
  return mesh
}

function axialRing(radius, tube, material, x, part) {
  const mesh = addPart(
    new THREE.TorusGeometry(radius, tube, 12, 64),
    material,
    [x, 0, 0],
    part,
  )
  mesh.rotation.y = Math.PI / 2
  return mesh
}

function addRotor(x, radius, bladeCount, part, material) {
  const rotor = new THREE.Group()
  rotor.position.x = x
  rotor.userData.part = part
  engineGroup.add(rotor)
  addPart(
    new THREE.CylinderGeometry(radius * 0.17, radius * 0.17, 0.12, 24),
    material,
    [0, 0, 0],
    part,
    rotor,
  ).rotation.z = Math.PI / 2

  for (let index = 0; index < bladeCount; index += 1) {
    const angle = (index * Math.PI * 2) / bladeCount
    const blade = addPart(
      new THREE.BoxGeometry(0.075, radius * 0.76, 0.12),
      material,
      [0, Math.cos(angle) * radius * 0.53, Math.sin(angle) * radius * 0.53],
      part,
      rotor,
    )
    blade.rotation.x = angle
  }
  return rotor
}

function buildEngine() {
  engineGroup = new THREE.Group()
  scene.add(engineGroup)
  internalParts = []

  const metal = new THREE.MeshStandardMaterial({
    color: 0x7d91a4,
    metalness: 0.78,
    roughness: 0.28,
  })
  const brightMetal = new THREE.MeshStandardMaterial({
    color: 0xc3d1db,
    metalness: 0.82,
    roughness: 0.22,
  })
  const warmMetal = new THREE.MeshStandardMaterial({
    color: 0xd98242,
    metalness: 0.5,
    roughness: 0.34,
    emissive: 0x421900,
  })
  const darkMetal = new THREE.MeshStandardMaterial({
    color: 0x25394b,
    metalness: 0.72,
    roughness: 0.31,
  })

  casing = new THREE.Mesh(
    new THREE.CylinderGeometry(1.05, 0.82, 4.65, 64, 1, false),
    new THREE.MeshPhysicalMaterial({
      color: 0x4c7489,
      metalness: 0.46,
      roughness: 0.28,
      transparent: true,
      opacity: 0.88,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  )
  casing.rotation.z = Math.PI / 2
  casing.userData.part = 'casing'
  casing.renderOrder = 2
  engineGroup.add(casing)

  const core = axialCylinder(0.28, 4.55, darkMetal, 0.12, 'turbine')
  internalParts.push(core)
  const coreNose = addPart(
    new THREE.ConeGeometry(0.38, 0.58, 48),
    brightMetal,
    [-2.32, 0, 0],
    'fan',
  )
  coreNose.rotation.z = -Math.PI / 2
  internalParts.push(coreNose)
  axialRing(0.93, 0.075, brightMetal, -2.18, 'fan')
  axialRing(0.77, 0.05, metal, 2.25, 'exhaust')

  const fan = addRotor(-2.05, 0.9, 20, 'fan', brightMetal)
  internalParts.push(fan)
  for (let stage = 0; stage < 4; stage += 1) {
    const x = -1.35 + stage * 0.48
    const radius = 0.72 - stage * 0.045
    const rotor = addRotor(x, radius, 14, 'compressor', metal)
    internalParts.push(rotor)
    internalParts.push(axialRing(radius, 0.022, brightMetal, x, 'compressor'))
  }

  const combustion = axialRing(0.54, 0.12, warmMetal, 0.72, 'combustion')
  internalParts.push(combustion)
  for (let index = 0; index < 8; index += 1) {
    const angle = (index * Math.PI * 2) / 8
    const injector = addPart(
      new THREE.CylinderGeometry(0.035, 0.035, 0.42, 10),
      warmMetal,
      [0.72, Math.cos(angle) * 0.52, Math.sin(angle) * 0.52],
      'combustion',
    )
    injector.rotation.x = angle
    internalParts.push(injector)
  }

  for (let stage = 0; stage < 3; stage += 1) {
    const x = 1.15 + stage * 0.36
    const radius = 0.62 - stage * 0.065
    const rotor = addRotor(x, radius, 12, 'turbine', brightMetal)
    internalParts.push(rotor)
  }

  const nozzle = addPart(
    new THREE.CylinderGeometry(0.35, 0.57, 0.78, 48, 1, true),
    metal,
    [2.28, 0, 0],
    'exhaust',
  )
  nozzle.rotation.z = Math.PI / 2
  internalParts.push(nozzle)
}

function setCutaway(value) {
  cutaway.value = value
  casing.material.transparent = value
  casing.material.opacity = value ? 0.17 : 0.9
  casing.material.color.set(value ? 0x73c7bd : 0x527387)
  internalParts.forEach(part => {
    part.visible = value
  })
}

function selectPart(part) {
  if (partNames[part]) selectedPart.value = part
}

function onPointerClick(event) {
  const bounds = renderer.domElement.getBoundingClientRect()
  pointer.set(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
  )
  raycaster.setFromCamera(pointer, camera)
  const hit = raycaster.intersectObjects(engineGroup.children, true)
    .find(item => item.object.visible && item.object.userData.part)
  if (hit) selectPart(hit.object.userData.part)
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
  controls?.update()
  renderer?.render(scene, camera)
}

function resetView() {
  if (!camera || !controls) return
  camera.position.set(4.4, 3.6, 6.3)
  controls.target.set(0, 0, 0)
  controls.update()
}

watch(cutaway, value => setCutaway(value))

onMounted(() => {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0xf1f6f8)
  camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100)
  camera.position.set(4.4, 3.6, 6.3)
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.1
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.domElement.setAttribute('aria-label', 'Modelo 3D interactivo de un turbofan')
  renderer.domElement.addEventListener('click', onPointerClick)
  canvasHost.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.07
  controls.minDistance = 3.2
  controls.maxDistance = 11
  controls.target.set(0, 0, 0)

  scene.add(new THREE.HemisphereLight(0xe5f4ff, 0x687a83, 2.1))
  const keyLight = new THREE.DirectionalLight(0xffffff, 3.4)
  keyLight.position.set(-3, 5, 6)
  keyLight.castShadow = true
  scene.add(keyLight)
  const fillLight = new THREE.DirectionalLight(0x6fd8cb, 1.6)
  fillLight.position.set(2, 1, -5)
  scene.add(fillLight)

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    new THREE.MeshStandardMaterial({ color: 0xf1f6f8, roughness: 1 }),
  )
  floor.rotation.x = -Math.PI / 2
  floor.position.y = -1.18
  floor.receiveShadow = true
  scene.add(floor)
  const grid = new THREE.GridHelper(12, 24, 0xb9cbd1, 0xd8e3e7)
  grid.position.y = -1.17
  scene.add(grid)

  raycaster = new THREE.Raycaster()
  pointer = new THREE.Vector2()
  buildEngine()
  setCutaway(cutaway.value)
  resizeObserver = new ResizeObserver(onResize)
  resizeObserver.observe(canvasHost.value)
  onResize()
  animate()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(frameId)
  resizeObserver?.disconnect()
  controls?.dispose()
  if (renderer) {
    renderer.domElement.removeEventListener('click', onPointerClick)
    renderer.dispose()
    renderer.domElement.remove()
  }
  scene?.traverse(object => {
    object.geometry?.dispose()
    if (Array.isArray(object.material)) {
      object.material.forEach(material => material.dispose())
    } else {
      object.material?.dispose()
    }
  })
})
</script>

<template>
  <div class="turbofan-visual">
    <div class="turbofan-toolbar">
      <div class="view-toggle" role="group" aria-label="Vista del turbofan">
        <button :class="{ active: cutaway }" @click="setCutaway(true)">Corte interno</button>
        <button :class="{ active: !cutaway }" @click="setCutaway(false)">Exterior</button>
      </div>
      <button class="camera-reset" @click="resetView" aria-label="Restablecer cámara">↺ <span>Restablecer vista</span></button>
    </div>
    <div ref="canvasHost" class="turbofan-canvas"></div>
    <div class="canvas-hint">Arrastra para rotar · rueda para acercar · clic en una sección para explorar</div>
    <div class="engine-overlay">
      <span class="overlay-dot"></span>
      MOTOR #{{ String(engineId).padStart(3, '0') }}
      <span class="overlay-separator"></span>
      CICLO {{ observedCycle }}
    </div>
    <aside class="part-card">
      <span class="part-kicker">COMPONENTE</span>
      <strong>{{ partNames[selectedPart] }}</strong>
      <p>{{ detail[selectedPart] }}</p>
    </aside>
  </div>

  <div class="model-explainer">
    <div class="model-flow-heading">
      <div>
        <span class="part-kicker">DEL SENSOR A LA PREDICCIÓN</span>
        <h3>Cómo estima la vida útil</h3>
      </div>
      <span class="model-chip">MLP · motor seleccionado</span>
    </div>
    <div class="flow-steps" aria-label="Flujo de inferencia MLP">
      <div class="flow-step"><span class="flow-icon">▤</span><strong>Ventana temporal</strong><small>30 ciclos × 19 señales</small></div>
      <span class="flow-arrow" aria-hidden="true">→</span>
      <div class="flow-step network-step"><span class="network-nodes"><i v-for="node in 7" :key="node"></i></span><strong>Capa densa</strong><small>128 neuronas · ReLU</small></div>
      <span class="flow-arrow" aria-hidden="true">→</span>
      <div class="flow-step network-step"><span class="network-nodes compact-nodes"><i v-for="node in 5" :key="node"></i></span><strong>Capa densa</strong><small>64 neuronas · ReLU</small></div>
      <span class="flow-arrow" aria-hidden="true">→</span>
      <div class="flow-step prediction-step"><span class="flow-value">{{ mlpRul === null ? '—' : `${Number(mlpRul).toFixed(1)}` }}</span><strong>RUL predicho</strong><small>ciclos restantes</small></div>
    </div>
    <div class="sensor-context">
      <span class="sensor-mark">⌁</span>
      <span><strong>{{ sensor }}</strong><small>Lectura más reciente: {{ sensorValue === null ? 'sin dato' : Number(sensorValue).toFixed(3) }}</small></span>
      <span class="sensor-disclaimer">Señal observada; su ubicación física no se especifica en C-MAPSS.</span>
    </div>
    <div class="other-model">La LSTM estima <strong>{{ lstmRul === null ? '—' : `${Number(lstmRul).toFixed(1)} ciclos` }}</strong> procesando la misma secuencia sin aplanarla.</div>
  </div>
</template>

<style scoped>
.turbofan-visual{position:relative;overflow:hidden;border:1px solid #dfe9ed;border-radius:12px;background:#f1f6f8}.turbofan-toolbar{position:absolute;z-index:2;top:14px;left:15px;right:15px;display:flex;justify-content:space-between;align-items:center}.view-toggle{display:flex;padding:3px;background:#fff;border:1px solid #dce7ea;border-radius:9px;box-shadow:0 2px 10px #18394a0d}.view-toggle button,.camera-reset{border:0;background:transparent;color:#63788a;border-radius:7px;padding:8px 11px;font-size:11px;font-weight:700}.view-toggle button.active{background:#e2f5f1;color:#0d9c88}.camera-reset{background:#fff;border:1px solid #dce7ea;box-shadow:0 2px 10px #18394a0d}.camera-reset span{margin-left:4px}.turbofan-canvas{height:420px;touch-action:none}.turbofan-canvas :deep(canvas){display:block;width:100%;height:100%;outline:none}.canvas-hint{position:absolute;bottom:13px;left:50%;transform:translateX(-50%);white-space:nowrap;color:#788b98;font-size:10px;background:#ffffffc9;border-radius:30px;padding:7px 12px}.engine-overlay{position:absolute;top:70px;left:19px;display:flex;align-items:center;gap:7px;color:#597080;font-size:9px;font-weight:800;letter-spacing:1px}.overlay-dot{width:7px;height:7px;background:#18b79f;border-radius:50%;box-shadow:0 0 0 3px #18b79f24}.overlay-separator{height:12px;width:1px;background:#c9d6db;margin:0 3px}.part-card{position:absolute;right:15px;bottom:47px;width:190px;padding:13px 15px;border:1px solid #dce7e9;border-radius:10px;background:#fffffff0;box-shadow:0 5px 20px #173a4a12}.part-kicker{display:block;color:#14a18c;font-size:9px;font-weight:800;letter-spacing:1.3px}.part-card strong{display:block;margin-top:5px;color:#20384a;font:700 14px Manrope,Arial,sans-serif}.part-card p{margin:5px 0 0;color:#708391;font-size:10px;line-height:1.5}.model-explainer{margin-top:16px;padding:20px 21px 17px;border:1px solid #e3ebef;border-radius:12px;background:#fff}.model-flow-heading{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.model-flow-heading h3{margin:5px 0 0;color:#1b3043;font:700 17px Manrope,Arial,sans-serif}.model-chip{padding:7px 10px;background:#e6f7f3;border:1px solid #d1efe8;border-radius:99px;color:#188d7e;font-size:10px;font-weight:700}.flow-steps{display:grid;grid-template-columns:1fr 22px 1fr 22px 1fr 22px 1fr;align-items:center;gap:8px}.flow-step{min-width:0;min-height:110px;padding:13px 12px;border:1px solid #e5edf0;border-radius:10px;background:#f9fbfc;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center}.flow-icon{color:#18a993;font-size:23px;margin-bottom:6px}.flow-step strong{color:#294052;font-size:11px}.flow-step small{margin-top:4px;color:#82929f;font-size:9px}.flow-arrow{color:#9db0b9;text-align:center;font-size:17px}.network-nodes{width:72px;height:29px;display:flex;justify-content:center;align-items:center;gap:5px;margin-bottom:6px}.network-nodes i{width:7px;height:7px;border-radius:50%;background:#32bca7;box-shadow:0 0 0 2px #d8f2ed}.network-nodes i:nth-child(2n){transform:translateY(7px);background:#6b9ed0}.network-nodes i:nth-child(3n){transform:translateY(-5px)}.compact-nodes{width:57px}.prediction-step{background:#e9f8f4;border-color:#cfeee6}.flow-value{color:#119b88;font:800 21px Manrope,Arial,sans-serif;margin-bottom:5px}.sensor-context{display:flex;align-items:center;gap:10px;margin-top:13px;padding:10px 12px;background:#f7fafb;border-radius:8px}.sensor-mark{color:#159d8a;font-size:22px}.sensor-context strong,.sensor-context small{display:block}.sensor-context strong{font-size:10px;color:#30485b}.sensor-context small{font-size:9px;color:#84949f;margin-top:3px}.sensor-disclaimer{margin-left:auto;max-width:230px;text-align:right;color:#86959e;font-size:9px;line-height:1.4}.other-model{margin-top:11px;text-align:right;color:#7a8a96;font-size:10px}.other-model strong{color:#477184}
@media(max-width:760px){.turbofan-canvas{height:340px}.camera-reset span{display:none}.part-card{width:154px;right:10px;bottom:42px;padding:10px}.canvas-hint{font-size:8px;bottom:9px}.flow-steps{grid-template-columns:1fr 14px 1fr;gap:7px}.flow-step{min-height:96px}.flow-arrow:nth-of-type(4){display:none}.flow-step:nth-of-type(5){grid-column:1}.flow-step:nth-of-type(7){grid-column:3}.sensor-context{align-items:flex-start;flex-wrap:wrap}.sensor-disclaimer{max-width:none;text-align:left;width:100%;margin-left:32px}.model-flow-heading{align-items:flex-start;gap:8px}.model-chip{white-space:nowrap;font-size:9px}}
</style>
