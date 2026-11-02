// Metadatos compartidos: componentes del turbofan y estaciones de medición C-MAPSS.
// Las estaciones provienen de la tabla de variables de Saxena et al. (2008),
// la documentación pública del benchmark. Indican en qué sección del motor se
// mide cada variable, no la posición física exacta de un sensor.

export const PARTS = {
  nacelle: {
    name: 'Góndola',
    short: 'Góndola',
    desc: 'Carcasa aerodinámica exterior. Guía el aire hacia el ventilador y contiene el flujo de derivación.',
  },
  fan: {
    name: 'Ventilador (fan)',
    short: 'Ventilador',
    desc: 'Rotor de álabes anchos movido por la turbina de baja presión. Acelera el aire: la mayor parte pasa por fuera del núcleo y aporta la mayor parte del empuje.',
  },
  bypass: {
    name: 'Ducto de derivación',
    short: 'Derivación',
    desc: 'Canal anular entre la góndola y el núcleo. Las guías de salida enderezan el flujo del ventilador antes de expulsarlo sin pasar por la combustión.',
  },
  lpc: {
    name: 'Compresor de baja presión (LPC)',
    short: 'LPC',
    desc: 'Etapas de refuerzo ("booster") tras el ventilador. Comprimen la fracción de aire que entra al núcleo, en el mismo eje que el ventilador.',
  },
  hpc: {
    name: 'Compresor de alta presión (HPC)',
    short: 'HPC',
    desc: 'Muchas etapas que reducen el canal y elevan presión y temperatura del aire antes de la combustión. Gira en el eje de alta presión.',
  },
  combustor: {
    name: 'Cámara de combustión',
    short: 'Combustión',
    desc: 'Cámara anular donde los inyectores mezclan combustible con el aire comprimido. Los gases salen a la temperatura más alta del ciclo.',
  },
  hpt: {
    name: 'Turbina de alta presión (HPT)',
    short: 'HPT',
    desc: 'Primera turbina tras la combustión. Extrae energía para mover el compresor de alta presión y requiere aire de enfriamiento.',
  },
  lpt: {
    name: 'Turbina de baja presión (LPT)',
    short: 'LPT',
    desc: 'Varias etapas que extraen la energía restante para mover el ventilador y el LPC por el eje de baja presión.',
  },
  nozzle: {
    name: 'Tobera y cono de escape',
    short: 'Tobera',
    desc: 'Acelera los gases calientes del núcleo hacia atrás para generar el resto del empuje.',
  },
}

export const PART_ORDER = ['nacelle', 'fan', 'bypass', 'lpc', 'hpc', 'combustor', 'hpt', 'lpt', 'nozzle']

export const SENSORS = {
  sensor_2: { code: 'T24', name: 'Temperatura total, salida del LPC', unit: '°R', part: 'lpc' },
  sensor_3: { code: 'T30', name: 'Temperatura total, salida del HPC', unit: '°R', part: 'hpc' },
  sensor_4: { code: 'T50', name: 'Temperatura total, salida de la LPT', unit: '°R', part: 'lpt' },
  sensor_6: { code: 'P15', name: 'Presión total, ducto de derivación', unit: 'psia', part: 'bypass' },
  sensor_7: { code: 'P30', name: 'Presión total, salida del HPC', unit: 'psia', part: 'hpc' },
  sensor_8: { code: 'Nf', name: 'Velocidad física del ventilador', unit: 'rpm', part: 'fan' },
  sensor_9: { code: 'Nc', name: 'Velocidad física del núcleo', unit: 'rpm', part: 'hpc' },
  sensor_11: { code: 'Ps30', name: 'Presión estática, salida del HPC', unit: 'psia', part: 'hpc' },
  sensor_12: { code: 'phi', name: 'Flujo de combustible / Ps30', unit: 'pps/psi', part: 'combustor' },
  sensor_13: { code: 'NRf', name: 'Velocidad corregida del ventilador', unit: 'rpm', part: 'fan' },
  sensor_14: { code: 'NRc', name: 'Velocidad corregida del núcleo', unit: 'rpm', part: 'hpc' },
  sensor_15: { code: 'BPR', name: 'Relación de derivación', unit: '', part: 'bypass' },
  sensor_17: { code: 'htBleed', name: 'Entalpía del aire sangrado', unit: '', part: 'hpc' },
  sensor_20: { code: 'W31', name: 'Refrigerante sangrado para la HPT', unit: 'lbm/s', part: 'hpt' },
  sensor_21: { code: 'W32', name: 'Refrigerante sangrado para la LPT', unit: 'lbm/s', part: 'lpt' },
}

export const sensorsOfPart = part =>
  Object.keys(SENSORS).filter(key => SENSORS[key].part === part)

export const sensorLabel = key => (SENSORS[key] ? `${SENSORS[key].code} · ${SENSORS[key].name}` : key)

// ---- Estado de un motor según el umbral de atención de la flota -------------
export const LEVELS = {
  low: { id: 'low', label: 'Bajo umbral', tone: 'warn' },
  watch: { id: 'watch', label: 'Vigilar', tone: 'watch' },
  ok: { id: 'ok', label: 'Sobre umbral', tone: 'ok' },
}

export const minRul = predictions => Math.min(predictions.mlp, predictions.lstm)

export function levelOf(predictions, threshold) {
  const rul = minRul(predictions)
  if (rul <= threshold) return LEVELS.low
  if (rul <= threshold * 2) return LEVELS.watch
  return LEVELS.ok
}

export const fmt = (value, digits = 2) =>
  Number(value ?? 0).toLocaleString('es-GT', { maximumFractionDigits: digits })
export const money = value => `Q${fmt(value)}`
export const pad = id => String(id ?? 0).padStart(3, '0')
export const modelName = name => (name === 'mlp' ? 'MLP' : name === 'lstm' ? 'LSTM' : 'Calendario')
