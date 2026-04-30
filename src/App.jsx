import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { parse, compile } from 'mathjs'
import { Download, Trash2, Play, Sliders, Info, ChevronDown } from 'lucide-react'

// ----- Presets -----
const PRESETS = [
  { name: 'Exponential growth', f: 'y', tMin: -3, tMax: 3, yMin: -4, yMax: 4 },
  { name: 'Exponential decay', f: '-y', tMin: -3, tMax: 3, yMin: -4, yMax: 4 },
  { name: 'Logistic', f: 'y * (1 - y)', tMin: -3, tMax: 6, yMin: -0.5, yMax: 1.5 },
  { name: 'Linear non-autonomous', f: 't - y', tMin: -4, tMax: 4, yMin: -4, yMax: 4 },
  { name: 'Riccati-like', f: 'y^2 - t', tMin: -3, tMax: 4, yMin: -3, yMax: 3 },
  { name: 'Sinusoidal mix', f: 'sin(t) * cos(y)', tMin: -2 * Math.PI, tMax: 2 * Math.PI, yMin: -3, yMax: 3 },
  { name: 'Singular at t=0', f: '-y / t', tMin: -3, tMax: 3, yMin: -3, yMax: 3 },
  { name: 'Three equilibria', f: '(y^2 - 1) * (y - 2)', tMin: -2, tMax: 4, yMin: -2, yMax: 3 },
  { name: 'Newton cooling', f: '-0.5 * (y - 2)', tMin: 0, tMax: 8, yMin: -1, yMax: 5 },
  { name: 'Pendulum-ish', f: '-sin(y) - 0.1 * y', tMin: 0, tMax: 12, yMin: -4, yMax: 4 },
]

// ----- Math utilities -----

function compileF(expr) {
  try {
    const node = parse(expr)
    const code = node.compile()
    // test eval
    code.evaluate({ t: 0, y: 0 })
    return { code, error: null, node }
  } catch (e) {
    return { code: null, error: e.message, node: null }
  }
}

function evalF(code, t, y) {
  try {
    const v = code.evaluate({ t, y })
    if (typeof v !== 'number' || !isFinite(v)) return NaN
    return v
  } catch {
    return NaN
  }
}

// Symbolic-ish autonomy check: sample f(t1,y) - f(t2,y) on a coarse grid.
// If max |Δ| is below a tiny relative tolerance, treat as autonomous.
function isAutonomous(code, tMin, tMax, yMin, yMax) {
  if (!code) return false
  const ys = 9
  const ts = 7
  const t1 = tMin
  const t2 = tMax
  let maxDelta = 0
  let maxAbs = 0
  for (let i = 0; i < ys; i++) {
    const y = yMin + ((yMax - yMin) * i) / (ys - 1)
    for (let j = 0; j < ts; j++) {
      const tj = tMin + ((tMax - tMin) * j) / (ts - 1)
      const a = evalF(code, t1, y)
      const b = evalF(code, t2, y)
      const c = evalF(code, tj, y)
      if (isFinite(a) && isFinite(b) && isFinite(c)) {
        maxDelta = Math.max(maxDelta, Math.abs(a - b), Math.abs(a - c))
        maxAbs = Math.max(maxAbs, Math.abs(a), Math.abs(b), Math.abs(c))
      }
    }
  }
  const tol = 1e-9 + 1e-7 * maxAbs
  return maxDelta < tol
}

// Find equilibria of f(y) (assumed autonomous) over [yMin, yMax] by bracketed bisection.
function findEquilibria(code, yMin, yMax) {
  const N = 800
  const samples = new Float64Array(N + 1)
  for (let i = 0; i <= N; i++) {
    const y = yMin + ((yMax - yMin) * i) / N
    samples[i] = evalF(code, 0, y)
  }
  const roots = []
  for (let i = 0; i < N; i++) {
    const a = samples[i]
    const b = samples[i + 1]
    if (!isFinite(a) || !isFinite(b)) continue
    if (a === 0) {
      const y = yMin + ((yMax - yMin) * i) / N
      roots.push(y)
      continue
    }
    if (a * b < 0) {
      // bisect
      let lo = yMin + ((yMax - yMin) * i) / N
      let hi = yMin + ((yMax - yMin) * (i + 1)) / N
      let fa = a, fb = b
      for (let k = 0; k < 60; k++) {
        const mid = 0.5 * (lo + hi)
        const fm = evalF(code, 0, mid)
        if (!isFinite(fm)) break
        if (fm === 0 || hi - lo < 1e-12) { lo = hi = mid; break }
        if (fa * fm < 0) { hi = mid; fb = fm } else { lo = mid; fa = fm }
      }
      roots.push(0.5 * (lo + hi))
    }
  }
  // Deduplicate close roots
  const dedup = []
  const tol = (yMax - yMin) * 1e-4
  for (const r of roots) {
    if (!dedup.some(d => Math.abs(d - r) < tol)) dedup.push(r)
  }
  // Classify each root by signs on either side
  const eps = (yMax - yMin) * 1e-3
  return dedup.map(y0 => {
    const left = evalF(code, 0, y0 - eps)
    const right = evalF(code, 0, y0 + eps)
    let kind
    if (left > 0 && right < 0) kind = 'stable'
    else if (left < 0 && right > 0) kind = 'unstable'
    else if (left > 0 && right > 0) kind = 'semi-up'
    else if (left < 0 && right < 0) kind = 'semi-down'
    else kind = 'unknown'
    return { y: y0, kind, left, right }
  })
}

// RK4 with adaptive-ish step (halving on big derivative).
function rk4Step(code, t, y, h) {
  const k1 = evalF(code, t, y)
  if (!isFinite(k1)) return null
  const k2 = evalF(code, t + h / 2, y + (h / 2) * k1)
  if (!isFinite(k2)) return null
  const k3 = evalF(code, t + h / 2, y + (h / 2) * k2)
  if (!isFinite(k3)) return null
  const k4 = evalF(code, t + h, y + h * k3)
  if (!isFinite(k4)) return null
  return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
}

function integrate(code, t0, y0, tEnd, yBounds, direction = 1) {
  const pts = [[t0, y0]]
  const span = Math.abs(tEnd - t0)
  if (span === 0) return pts
  let h = (span / 1500) * direction
  const minH = Math.abs(h) * 1e-3
  let t = t0, y = y0
  const guard = (yBounds[1] - yBounds[0]) * 4 // give margin off-screen
  const yLo = yBounds[0] - guard
  const yHi = yBounds[1] + guard
  let steps = 0
  const maxSteps = 8000
  while (steps < maxSteps && ((direction > 0 && t < tEnd) || (direction < 0 && t > tEnd))) {
    let hh = h
    if ((direction > 0 && t + hh > tEnd) || (direction < 0 && t + hh < tEnd)) hh = tEnd - t
    let yNext = rk4Step(code, t, y, hh)
    if (yNext === null || !isFinite(yNext)) break
    // adaptive: if the jump is too big, halve
    let tries = 0
    while (Math.abs(yNext - y) > (yBounds[1] - yBounds[0]) * 0.05 && Math.abs(hh) > minH && tries < 8) {
      hh *= 0.5
      yNext = rk4Step(code, t, y, hh)
      if (yNext === null) break
      tries++
    }
    if (yNext === null || !isFinite(yNext)) break
    t += hh
    y = yNext
    pts.push([t, y])
    if (y < yLo || y > yHi) break
    steps++
  }
  return pts
}

// ----- Color palette -----
const COLORS = [
  '#c1432f', '#2f6fc1', '#4a8a3a', '#a3578f', '#c98c2a',
  '#3a8a8a', '#8a3a5e', '#5e8a3a', '#c14f7a', '#3a5e8a',
]

// ----- Component -----
export default function App() {
  const [expr, setExpr] = useState('y * (1 - y)')
  const [tMin, setTMin] = useState(-3)
  const [tMax, setTMax] = useState(6)
  const [yMin, setYMin] = useState(-0.5)
  const [yMax, setYMax] = useState(1.5)
  const [density, setDensity] = useState(22)
  const [lengthMul, setLengthMul] = useState(0.8)
  const [showArrows, setShowArrows] = useState(false)
  const [showIsoclines, setShowIsoclines] = useState(false)
  const [autonomousOverride, setAutonomousOverride] = useState('auto') // 'auto' | 'on' | 'off'
  const [solutions, setSolutions] = useState([]) // { t0, y0, fwd, bwd, color }
  const [presetOpen, setPresetOpen] = useState(false)
  const [hoverPt, setHoverPt] = useState(null)

  const fieldCanvasRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const containerRef = useRef(null)
  const [canvasSize, setCanvasSize] = useState({ w: 720, h: 540 })

  // Compile f
  const compiled = useMemo(() => compileF(expr), [expr])

  // Detect autonomy
  const autonomous = useMemo(() => {
    if (autonomousOverride === 'on') return true
    if (autonomousOverride === 'off') return false
    return isAutonomous(compiled.code, tMin, tMax, yMin, yMax)
  }, [compiled.code, tMin, tMax, yMin, yMax, autonomousOverride])

  // Equilibria for phase line (only meaningful if autonomous)
  const equilibria = useMemo(() => {
    if (!autonomous || !compiled.code) return []
    return findEquilibria(compiled.code, yMin, yMax)
  }, [autonomous, compiled.code, yMin, yMax])

  // ----- Resize handling -----
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const w = Math.max(420, Math.floor(entry.contentRect.width))
        const h = Math.max(380, Math.floor(w * 0.65))
        setCanvasSize({ w, h })
      }
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  // ----- Coordinate transforms -----
  const PAD_L = 56, PAD_R = 24, PAD_T = 24, PAD_B = 44
  const phaseLineW = autonomous ? 80 : 0
  const plotW = canvasSize.w - PAD_L - PAD_R - phaseLineW
  const plotH = canvasSize.h - PAD_T - PAD_B

  const xToPx = useCallback(t => PAD_L + ((t - tMin) / (tMax - tMin)) * plotW, [tMin, tMax, plotW])
  const yToPx = useCallback(y => PAD_T + (1 - (y - yMin) / (yMax - yMin)) * plotH, [yMin, yMax, plotH])
  const pxToX = useCallback(px => tMin + ((px - PAD_L) / plotW) * (tMax - tMin), [tMin, tMax, plotW])
  const pxToY = useCallback(py => yMin + (1 - (py - PAD_T) / plotH) * (yMax - yMin), [yMin, yMax, plotH])

  // ----- Render slope field to canvas -----
  useEffect(() => {
    const cv = fieldCanvasRef.current
    if (!cv || !compiled.code) return
    const dpr = window.devicePixelRatio || 1
    cv.width = canvasSize.w * dpr
    cv.height = canvasSize.h * dpr
    cv.style.width = canvasSize.w + 'px'
    cv.style.height = canvasSize.h + 'px'
    const ctx = cv.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, canvasSize.w, canvasSize.h)

    // Plot background (warm cream paper)
    ctx.fillStyle = '#faf6ec'
    ctx.fillRect(PAD_L, PAD_T, plotW, plotH)

    // Grid
    ctx.strokeStyle = '#e3dccb'
    ctx.lineWidth = 1
    const tStep = niceStep(tMax - tMin)
    const yStep = niceStep(yMax - yMin)
    ctx.beginPath()
    for (let t = Math.ceil(tMin / tStep) * tStep; t <= tMax; t += tStep) {
      const x = xToPx(t)
      ctx.moveTo(x, PAD_T)
      ctx.lineTo(x, PAD_T + plotH)
    }
    for (let y = Math.ceil(yMin / yStep) * yStep; y <= yMax; y += yStep) {
      const py = yToPx(y)
      ctx.moveTo(PAD_L, py)
      ctx.lineTo(PAD_L + plotW, py)
    }
    ctx.stroke()

    // Zero axes (if in view)
    ctx.strokeStyle = '#bdb39c'
    ctx.lineWidth = 1.2
    if (tMin <= 0 && tMax >= 0) {
      const x0 = xToPx(0)
      ctx.beginPath(); ctx.moveTo(x0, PAD_T); ctx.lineTo(x0, PAD_T + plotH); ctx.stroke()
    }
    if (yMin <= 0 && yMax >= 0) {
      const y0 = yToPx(0)
      ctx.beginPath(); ctx.moveTo(PAD_L, y0); ctx.lineTo(PAD_L + plotW, y0); ctx.stroke()
    }

    // Axis labels & ticks
    ctx.fillStyle = '#5a5347'
    ctx.font = "500 11px 'JetBrains Mono', monospace"
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (let t = Math.ceil(tMin / tStep) * tStep; t <= tMax + 1e-9; t += tStep) {
      const x = xToPx(t)
      const label = formatTick(t, tStep)
      ctx.fillText(label, x, PAD_T + plotH + 6)
    }
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let y = Math.ceil(yMin / yStep) * yStep; y <= yMax + 1e-9; y += yStep) {
      const py = yToPx(y)
      const label = formatTick(y, yStep)
      ctx.fillText(label, PAD_L - 8, py)
    }

    // Axis frame
    ctx.strokeStyle = '#8a8170'
    ctx.lineWidth = 1.5
    ctx.strokeRect(PAD_L, PAD_T, plotW, plotH)

    // Axis names
    ctx.fillStyle = '#1a1715'
    ctx.font = "italic 600 14px 'Fraunces', serif"
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText('t', PAD_L + plotW / 2, PAD_T + plotH + 22)
    ctx.save()
    ctx.translate(16, PAD_T + plotH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('y', 0, 0)
    ctx.restore()

    // Slope field segments: uniform-length using θ = atan(f)
    const cellW = plotW / density
    const cellH = plotH / density
    const cellMin = Math.min(cellW, cellH)
    const segLen = cellMin * lengthMul

    ctx.strokeStyle = '#2a2520'
    ctx.lineWidth = 1.1
    ctx.lineCap = 'round'
    for (let i = 0; i < density; i++) {
      for (let j = 0; j < density; j++) {
        const t = tMin + ((tMax - tMin) * (i + 0.5)) / density
        const y = yMin + ((yMax - yMin) * (j + 0.5)) / density
        const slope = evalF(compiled.code, t, y)
        if (!isFinite(slope)) {
          // draw faint gray dot
          ctx.fillStyle = '#d8d2c0'
          ctx.beginPath()
          ctx.arc(xToPx(t), yToPx(y), 1.2, 0, Math.PI * 2)
          ctx.fill()
          continue
        }
        // Convert mathematical slope (dy/dt in user space) to screen-space angle
        // dy_screen = -slope * (plotH/(yMax-yMin)) * dt_screen / (plotW/(tMax-tMin))
        const screenSlope = -slope * (plotH / (yMax - yMin)) / (plotW / (tMax - tMin))
        const theta = Math.atan(screenSlope)
        const cx = xToPx(t)
        const cy = yToPx(y)
        const dx = (segLen / 2) * Math.cos(theta)
        const dy = (segLen / 2) * Math.sin(theta)
        ctx.beginPath()
        ctx.moveTo(cx - dx, cy - dy)
        ctx.lineTo(cx + dx, cy + dy)
        ctx.stroke()
        if (showArrows) {
          // arrow points in direction of increasing t
          const ah = Math.min(3.5, segLen * 0.22)
          const ang = theta // direction vector aligned with +t
          const tipX = cx + dx
          const tipY = cy + dy
          ctx.beginPath()
          ctx.moveTo(tipX, tipY)
          ctx.lineTo(tipX - ah * Math.cos(ang - 0.5), tipY - ah * Math.sin(ang - 0.5))
          ctx.moveTo(tipX, tipY)
          ctx.lineTo(tipX - ah * Math.cos(ang + 0.5), tipY - ah * Math.sin(ang + 0.5))
          ctx.stroke()
        }
      }
    }

    // Isoclines: curves where f(t, y) = c. Drawn AFTER the slope field so they
    // overlay the segments and read as distinct foreground curves. Each isocline
    // gets its own hue; the nullcline (c=0) is bold black. Curves are built by
    // marching across columns of t, finding y-roots in each column, then
    // connecting nearest roots between adjacent columns.
    if (showIsoclines) {
      const cValues = [-2, -1, -0.5, 0, 0.5, 1, 2]
      const isoColors = {
        '-2': '#3a5e8a', '-1': '#3a8a8a', '-0.5': '#5e8a3a',
        '0': '#1a1715',
        '0.5': '#c98c2a', '1': '#c1432f', '2': '#8a3a5e',
      }
      const fineN = 280
      const yN = 220
      const maxGapY = (yMax - yMin) * 0.06

      cValues.forEach(c => {
        const color = isoColors[String(c)] || '#5a5347'
        const isNull = c === 0
        ctx.strokeStyle = color
        ctx.lineWidth = isNull ? 2.6 : 2.0
        ctx.setLineDash(isNull ? [] : [6, 4])
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'

        const columns = []
        for (let i = 0; i <= fineN; i++) {
          const t = tMin + ((tMax - tMin) * i) / fineN
          const roots = []
          let prevY = null, prevVal = null
          for (let j = 0; j <= yN; j++) {
            const y = yMin + ((yMax - yMin) * j) / yN
            const v = evalF(compiled.code, t, y) - c
            if (prevVal !== null && isFinite(v) && isFinite(prevVal) && prevVal * v < 0) {
              let a = prevY, b = y, fa = prevVal
              for (let k = 0; k < 14; k++) {
                const m = 0.5 * (a + b)
                const fm = evalF(compiled.code, t, m) - c
                if (!isFinite(fm)) break
                if (fa * fm < 0) { b = m } else { a = m; fa = fm }
              }
              roots.push(0.5 * (a + b))
            }
            prevY = y
            prevVal = v
          }
          columns.push({ t, roots })
        }

        ctx.beginPath()
        for (let i = 0; i < columns.length - 1; i++) {
          const colA = columns[i]
          const colB = columns[i + 1]
          for (const ya of colA.roots) {
            let best = null, bestDist = Infinity
            for (const yb of colB.roots) {
              const d = Math.abs(yb - ya)
              if (d < bestDist) { bestDist = d; best = yb }
            }
            if (best !== null && bestDist < maxGapY) {
              ctx.moveTo(xToPx(colA.t), yToPx(ya))
              ctx.lineTo(xToPx(colB.t), yToPx(best))
            }
          }
        }
        ctx.stroke()

        // Label each isocline near the rightmost column with a root
        ctx.setLineDash([])
        for (let i = columns.length - 1; i >= 0; i--) {
          if (columns[i].roots.length > 0) {
            const tLab = columns[i].t
            const yLabel = columns[i].roots.reduce((a, b) => yToPx(a) < yToPx(b) ? a : b)
            const px = xToPx(tLab)
            const py = yToPx(yLabel)
            if (px < PAD_L + plotW - 6 && py > PAD_T + 10 && py < PAD_T + plotH - 6) {
              ctx.font = `${isNull ? '700' : '600'} 11px 'JetBrains Mono', monospace`
              ctx.textAlign = 'left'
              ctx.textBaseline = 'middle'
              const label = `c=${c}`
              const tw = ctx.measureText(label).width
              ctx.fillStyle = 'rgba(250, 246, 236, 0.92)'
              ctx.fillRect(px + 4, py - 8, tw + 6, 16)
              ctx.strokeStyle = color
              ctx.lineWidth = 1
              ctx.strokeRect(px + 4, py - 8, tw + 6, 16)
              ctx.fillStyle = color
              ctx.fillText(label, px + 7, py)
            }
            break
          }
        }
      })
      ctx.setLineDash([])
    }

    // ----- Phase line panel (autonomous only) -----
    if (autonomous) {
      const px0 = PAD_L + plotW + 28
      const lineX = px0 + 24
      // Background strip
      ctx.fillStyle = '#f0e9d6'
      ctx.fillRect(px0 - 6, PAD_T, phaseLineW - 16, plotH)
      ctx.strokeStyle = '#8a8170'
      ctx.lineWidth = 1.5
      ctx.strokeRect(px0 - 6, PAD_T, phaseLineW - 16, plotH)

      // Title
      ctx.fillStyle = '#1a1715'
      ctx.font = "600 11px 'JetBrains Mono', monospace"
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText('PHASE', px0 + (phaseLineW - 16) / 2 - 6, 4)

      // Vertical line
      ctx.strokeStyle = '#1a1715'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(lineX, PAD_T + 6)
      ctx.lineTo(lineX, PAD_T + plotH - 6)
      ctx.stroke()

      // Build interval list bounded by yMin, equilibria, yMax
      const sorted = [...equilibria].sort((a, b) => a.y - b.y)
      const points = [yMin, ...sorted.map(e => e.y), yMax]
      // Arrows: in each subinterval, sample mid-y, sign of f tells direction.
      // Arrow length scales with the subinterval height in pixels so it reads
      // clearly without overrunning adjacent equilibria.
      for (let i = 0; i < points.length - 1; i++) {
        const yLo = points[i]
        const yHi = points[i + 1]
        const yMid = 0.5 * (yLo + yHi)
        const sgn = evalF(compiled.code, 0, yMid)
        if (!isFinite(sgn) || sgn === 0) continue
        // Direction: + means y increasing (arrow points up on a y-axis where up = larger y)
        const dir = sgn > 0 ? -1 : 1 // screen y is flipped: smaller py = larger y
        const arrowY = yToPx(yMid)
        // Available pixel space in this subinterval; leave margin near equilibria.
        const intervalPx = Math.abs(yToPx(yLo) - yToPx(yHi))
        const halfLen = Math.max(8, Math.min(28, intervalPx * 0.32))
        const ahLen = Math.max(7, halfLen * 0.55)
        const ahWidth = Math.max(5, halfLen * 0.42)
        ctx.strokeStyle = '#c1432f'
        ctx.fillStyle = '#c1432f'
        ctx.lineWidth = 3.5
        ctx.lineCap = 'round'
        // shaft (stops short of the tip so the arrowhead reads as solid)
        ctx.beginPath()
        ctx.moveTo(lineX, arrowY - dir * halfLen)
        ctx.lineTo(lineX, arrowY + dir * (halfLen - ahLen * 0.6))
        ctx.stroke()
        // arrowhead
        const tipY = arrowY + dir * halfLen
        const baseY = tipY - dir * ahLen
        ctx.beginPath()
        ctx.moveTo(lineX, tipY)
        ctx.lineTo(lineX - ahWidth, baseY)
        ctx.lineTo(lineX + ahWidth, baseY)
        ctx.closePath()
        ctx.fill()
      }

      // Equilibria dots
      sorted.forEach(eq => {
        const py = yToPx(eq.y)
        ctx.lineWidth = 2
        if (eq.kind === 'stable') {
          ctx.fillStyle = '#1a1715'
          ctx.beginPath(); ctx.arc(lineX, py, 5.5, 0, Math.PI * 2); ctx.fill()
        } else if (eq.kind === 'unstable') {
          ctx.fillStyle = '#faf6ec'
          ctx.strokeStyle = '#1a1715'
          ctx.beginPath(); ctx.arc(lineX, py, 5.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
        } else {
          // semi-stable: half-filled
          ctx.fillStyle = '#1a1715'
          ctx.beginPath()
          ctx.arc(lineX, py, 5.5, eq.kind === 'semi-up' ? -Math.PI / 2 : Math.PI / 2, eq.kind === 'semi-up' ? Math.PI / 2 : 3 * Math.PI / 2)
          ctx.fill()
          ctx.strokeStyle = '#1a1715'
          ctx.beginPath(); ctx.arc(lineX, py, 5.5, 0, Math.PI * 2); ctx.stroke()
        }
        // Label y value
        ctx.fillStyle = '#1a1715'
        ctx.font = "500 10px 'JetBrains Mono', monospace"
        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'
        ctx.fillText(`y=${eq.y.toFixed(3)}`, lineX + 10, py)
      })

      // Also draw faint horizontal guides from equilibria into the field
      ctx.strokeStyle = 'rgba(193, 67, 47, 0.18)'
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1
      sorted.forEach(eq => {
        const py = yToPx(eq.y)
        if (py < PAD_T || py > PAD_T + plotH) return
        ctx.beginPath()
        ctx.moveTo(PAD_L, py)
        ctx.lineTo(PAD_L + plotW, py)
        ctx.stroke()
      })
      ctx.setLineDash([])
    }
  }, [compiled.code, tMin, tMax, yMin, yMax, density, lengthMul, showArrows, showIsoclines, autonomous, equilibria, canvasSize, plotW, plotH, xToPx, yToPx])

  // ----- Render solution curves on overlay -----
  useEffect(() => {
    const cv = overlayCanvasRef.current
    if (!cv) return
    const dpr = window.devicePixelRatio || 1
    cv.width = canvasSize.w * dpr
    cv.height = canvasSize.h * dpr
    cv.style.width = canvasSize.w + 'px'
    cv.style.height = canvasSize.h + 'px'
    const ctx = cv.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, canvasSize.w, canvasSize.h)

    // Clip to plot area so curves don't bleed
    ctx.save()
    ctx.beginPath()
    ctx.rect(PAD_L, PAD_T, plotW, plotH)
    ctx.clip()

    solutions.forEach(sol => {
      ctx.strokeStyle = sol.color
      ctx.lineWidth = 2.4
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      const all = [...sol.bwd.slice().reverse(), ...sol.fwd.slice(1)]
      ctx.beginPath()
      all.forEach(([t, y], idx) => {
        const px = xToPx(t)
        const py = yToPx(y)
        if (idx === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()
      // Initial-condition dot
      ctx.fillStyle = sol.color
      ctx.beginPath()
      ctx.arc(xToPx(sol.t0), yToPx(sol.y0), 4, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#faf6ec'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })
    ctx.restore()

    // Hover crosshair
    if (hoverPt) {
      ctx.save()
      ctx.beginPath()
      ctx.rect(PAD_L, PAD_T, plotW, plotH)
      ctx.clip()
      ctx.strokeStyle = 'rgba(26, 23, 21, 0.35)'
      ctx.setLineDash([2, 4])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(hoverPt.px, PAD_T)
      ctx.lineTo(hoverPt.px, PAD_T + plotH)
      ctx.moveTo(PAD_L, hoverPt.py)
      ctx.lineTo(PAD_L + plotW, hoverPt.py)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.restore()
    }
  }, [solutions, canvasSize, xToPx, yToPx, plotW, plotH, hoverPt])

  // ----- Click handler: integrate forward and backward -----
  const handleClick = (e) => {
    if (!compiled.code) return
    const rect = overlayCanvasRef.current.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    if (px < PAD_L || px > PAD_L + plotW || py < PAD_T || py > PAD_T + plotH) return
    const t0 = pxToX(px)
    const y0 = pxToY(py)
    const fwd = integrate(compiled.code, t0, y0, tMax + (tMax - tMin) * 0.05, [yMin, yMax], 1)
    const bwd = integrate(compiled.code, t0, y0, tMin - (tMax - tMin) * 0.05, [yMin, yMax], -1)
    const color = COLORS[solutions.length % COLORS.length]
    setSolutions(prev => [...prev, { t0, y0, fwd, bwd, color }])
  }

  const handleMouseMove = (e) => {
    const rect = overlayCanvasRef.current.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    if (px < PAD_L || px > PAD_L + plotW || py < PAD_T || py > PAD_T + plotH) {
      setHoverPt(null)
      return
    }
    setHoverPt({ px, py, t: pxToX(px), y: pxToY(py) })
  }

  const handleMouseLeave = () => setHoverPt(null)

  // ----- SVG Export -----
  const exportSVG = () => {
    if (!compiled.code) return
    const svg = buildSVG({
      expr, tMin, tMax, yMin, yMax, density, lengthMul,
      compiled, autonomous, equilibria, solutions,
      showArrows, showIsoclines,
      canvasSize, PAD_L, PAD_R, PAD_T, PAD_B, phaseLineW, plotW, plotH,
    })
    const blob = new Blob([svg], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `slope-field-${Date.now()}.svg`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ----- Apply preset -----
  const applyPreset = (p) => {
    setExpr(p.f)
    setTMin(p.tMin); setTMax(p.tMax); setYMin(p.yMin); setYMax(p.yMax)
    setSolutions([])
    setPresetOpen(false)
  }

  return (
    <div style={{ minHeight: '100vh', padding: '24px 28px 48px', maxWidth: 1320, margin: '0 auto' }}>
      <Header />
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 320px', gap: 24, marginTop: 20 }}>
        {/* Plot */}
        <div ref={containerRef} style={{ position: 'relative', background: '#1a1715', padding: 14, borderRadius: 6, boxShadow: '0 18px 40px -20px rgba(26,23,21,0.5)' }}>
          <div style={{ position: 'relative', borderRadius: 4, overflow: 'hidden', background: '#faf6ec' }}>
            <canvas ref={fieldCanvasRef} style={{ display: 'block' }} />
            <canvas
              ref={overlayCanvasRef}
              onClick={handleClick}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
              style={{ display: 'block', position: 'absolute', top: 0, left: 0, cursor: 'crosshair' }}
            />
          </div>
          <ReadoutBar hoverPt={hoverPt} compiled={compiled} expr={expr} autonomous={autonomous} />
        </div>

        {/* Side panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Panel>
            <PanelLabel>Equation</PanelLabel>
            <ExprInput expr={expr} setExpr={setExpr} compiled={compiled} />
            <div style={{ position: 'relative', marginTop: 10 }}>
              <button onClick={() => setPresetOpen(p => !p)} style={presetBtnStyle}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>PRESETS</span>
                <ChevronDown size={14} />
              </button>
              {presetOpen && (
                <div style={presetMenuStyle}>
                  {PRESETS.map(p => (
                    <button key={p.name} onClick={() => applyPreset(p)} style={presetItemStyle}>
                      <span style={{ fontWeight: 600, fontSize: 12 }}>{p.name}</span>
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: '#7a7264' }}>
                        y' = {p.f}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </Panel>

          <Panel>
            <PanelLabel>Window</PanelLabel>
            <RangeRow label="t" min={tMin} max={tMax} setMin={setTMin} setMax={setTMax} />
            <RangeRow label="y" min={yMin} max={yMax} setMin={setYMin} setMax={setYMax} />
          </Panel>

          <Panel>
            <PanelLabel>Field</PanelLabel>
            <SliderRow label="Density" value={density} min={6} max={60} step={1} onChange={setDensity} display={`${density}×${density}`} />
            <SliderRow label="Length" value={lengthMul} min={0.3} max={1.2} step={0.05} onChange={setLengthMul} display={lengthMul.toFixed(2)} />
            <CheckRow label="Direction arrows" checked={showArrows} onChange={setShowArrows} />
            <CheckRow label="Isoclines" checked={showIsoclines} onChange={setShowIsoclines} />
          </Panel>

          <Panel>
            <PanelLabel>
              Phase line {autonomous ? <span style={badgeOn}>auto</span> : <span style={badgeOff}>n/a</span>}
            </PanelLabel>
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
              {[
                { v: 'auto', label: 'Auto' },
                { v: 'on', label: 'Force on' },
                { v: 'off', label: 'Off' },
              ].map(opt => (
                <button
                  key={opt.v}
                  onClick={() => setAutonomousOverride(opt.v)}
                  style={{
                    ...segmentBtnStyle,
                    background: autonomousOverride === opt.v ? '#1a1715' : 'transparent',
                    color: autonomousOverride === opt.v ? '#faf6ec' : '#1a1715',
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            {autonomous && equilibria.length > 0 && (
              <EqLegend equilibria={equilibria} />
            )}
            {autonomous && equilibria.length === 0 && (
              <p style={{ fontSize: 11, color: '#7a7264', margin: 0 }}>No equilibria found in current y range.</p>
            )}
          </Panel>

          <Panel>
            <PanelLabel>Solutions</PanelLabel>
            <div style={{ fontSize: 11, color: '#5a5347', marginBottom: 10, lineHeight: 1.5 }}>
              Click anywhere in the plot to integrate the IVP through that point — both directions.
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={() => setSolutions([])} style={iconBtnStyle} title="Clear all">
                <Trash2 size={14} /> <span style={{ fontSize: 11 }}>Clear</span>
              </button>
              <button onClick={exportSVG} style={iconBtnStyle} title="Export SVG">
                <Download size={14} /> <span style={{ fontSize: 11 }}>SVG</span>
              </button>
            </div>
            <div style={{ marginTop: 10, fontSize: 11, color: '#5a5347' }}>
              {solutions.length} curve{solutions.length === 1 ? '' : 's'}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  )
}

// ----- Subcomponents -----

function Header() {
  return (
    <header style={{ display: 'flex', alignItems: 'baseline', gap: 16, borderBottom: '1px solid #1a1715', paddingBottom: 14 }}>
      <h1 style={{
        fontFamily: "'Fraunces', serif",
        fontWeight: 600,
        fontSize: 36,
        margin: 0,
        letterSpacing: '-0.02em',
        fontStyle: 'italic',
      }}>
        Slope Field
      </h1>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
        color: '#5a5347',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
      }}>
        y' = f(t, y) — Math 252
      </span>
      <div style={{ flex: 1 }} />
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
        color: '#7a7264',
        letterSpacing: '0.05em',
      }}>
        College of San Mateo
      </span>
    </header>
  )
}

function Panel({ children }) {
  return (
    <div style={{
      background: '#fffdf6',
      border: '1px solid #d8d2c0',
      borderRadius: 4,
      padding: 14,
      boxShadow: '0 1px 0 #e8e2d0',
    }}>
      {children}
    </div>
  )
}

function PanelLabel({ children }) {
  return (
    <div style={{
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 10,
      letterSpacing: '0.12em',
      textTransform: 'uppercase',
      color: '#1a1715',
      fontWeight: 600,
      marginBottom: 10,
      display: 'flex',
      alignItems: 'center',
      gap: 8,
    }}>
      {children}
    </div>
  )
}

function ExprInput({ expr, setExpr, compiled }) {
  return (
    <div>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        background: '#faf6ec',
        border: `1.5px solid ${compiled.error ? '#c1432f' : '#1a1715'}`,
        borderRadius: 3,
        padding: '8px 10px',
      }}>
        <span style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontWeight: 500, fontSize: 16 }}>
          y′ =
        </span>
        <input
          value={expr}
          onChange={e => setExpr(e.target.value)}
          spellCheck={false}
          style={{
            flex: 1,
            border: 'none',
            outline: 'none',
            background: 'transparent',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 13,
            color: '#1a1715',
          }}
        />
      </div>
      {compiled.error && (
        <div style={{ marginTop: 6, fontSize: 11, color: '#c1432f', fontFamily: "'JetBrains Mono', monospace" }}>
          {compiled.error}
        </div>
      )}
    </div>
  )
}

function RangeRow({ label, min, max, setMin, setMax }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
      <span style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontWeight: 500, fontSize: 14, width: 16 }}>{label}</span>
      <NumberCell value={min} onChange={setMin} />
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#7a7264' }}>→</span>
      <NumberCell value={max} onChange={setMax} />
    </div>
  )
}

function NumberCell({ value, onChange }) {
  const [local, setLocal] = useState(String(value))
  useEffect(() => { setLocal(String(value)) }, [value])
  return (
    <input
      value={local}
      onChange={e => setLocal(e.target.value)}
      onBlur={() => {
        const n = Number(local)
        if (Number.isFinite(n)) onChange(n)
        else setLocal(String(value))
      }}
      onKeyDown={e => { if (e.key === 'Enter') e.target.blur() }}
      style={{
        flex: 1,
        background: '#faf6ec',
        border: '1px solid #d8d2c0',
        borderRadius: 2,
        padding: '4px 6px',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 12,
        color: '#1a1715',
        outline: 'none',
        textAlign: 'center',
        minWidth: 0,
      }}
    />
  )
}

function SliderRow({ label, value, min, max, step, onChange, display }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: '#1a1715' }}>{label}</span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#5a5347' }}>{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: '#c1432f' }}
      />
    </div>
  )
}

function CheckRow({ label, checked, onChange }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#1a1715', cursor: 'pointer', marginTop: 4 }}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} style={{ accentColor: '#1a1715' }} />
      {label}
    </label>
  )
}

function ReadoutBar({ hoverPt, compiled, expr, autonomous }) {
  return (
    <div style={{
      display: 'flex',
      gap: 18,
      marginTop: 10,
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 11,
      color: '#d8d2c0',
      paddingLeft: 4,
    }}>
      {hoverPt ? (
        <>
          <span>t = {hoverPt.t.toFixed(3)}</span>
          <span>y = {hoverPt.y.toFixed(3)}</span>
          <span>y′ = {(() => { const v = compiled.code ? evalF(compiled.code, hoverPt.t, hoverPt.y) : NaN; return isFinite(v) ? v.toFixed(3) : '—' })()}</span>
        </>
      ) : (
        <span style={{ color: '#7a7264' }}>hover on plot for readout · click to integrate</span>
      )}
      <span style={{ flex: 1 }} />
      <span>{autonomous ? 'autonomous' : 'non-autonomous'}</span>
    </div>
  )
}

function EqLegend({ equilibria }) {
  return (
    <div style={{ fontSize: 11, color: '#1a1715' }}>
      <div style={{ marginBottom: 6, fontWeight: 600 }}>{equilibria.length} equilibria found</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {equilibria.map((eq, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: "'JetBrains Mono', monospace" }}>
            <EqDot kind={eq.kind} />
            <span>y* = {eq.y.toFixed(4)}</span>
            <span style={{ color: '#7a7264', marginLeft: 'auto' }}>{labelKind(eq.kind)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function EqDot({ kind }) {
  if (kind === 'stable') return <span style={{ width: 10, height: 10, background: '#1a1715', borderRadius: '50%', display: 'inline-block' }} />
  if (kind === 'unstable') return <span style={{ width: 10, height: 10, background: '#fffdf6', border: '1.5px solid #1a1715', borderRadius: '50%', display: 'inline-block' }} />
  return <span style={{ width: 10, height: 10, background: 'linear-gradient(90deg, #1a1715 50%, #fffdf6 50%)', border: '1.5px solid #1a1715', borderRadius: '50%', display: 'inline-block' }} />
}

function labelKind(k) {
  if (k === 'stable') return 'stable'
  if (k === 'unstable') return 'unstable'
  return 'semi-stable'
}

// ----- Helpers -----

function niceStep(span) {
  const raw = span / 8
  const exp = Math.floor(Math.log10(raw))
  const base = raw / Math.pow(10, exp)
  let nice
  if (base < 1.5) nice = 1
  else if (base < 3.5) nice = 2
  else if (base < 7.5) nice = 5
  else nice = 10
  return nice * Math.pow(10, exp)
}

function formatTick(v, step) {
  if (Math.abs(v) < step * 1e-6) return '0'
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + (step < 1 ? 1 : 0))
  return v.toFixed(decimals)
}

// ----- SVG export -----
function buildSVG({ expr, tMin, tMax, yMin, yMax, density, lengthMul, compiled, autonomous, equilibria, solutions, showArrows, showIsoclines, canvasSize, PAD_L, PAD_R, PAD_T, PAD_B, phaseLineW, plotW, plotH }) {
  const xToPx = t => PAD_L + ((t - tMin) / (tMax - tMin)) * plotW
  const yToPx = y => PAD_T + (1 - (y - yMin) / (yMax - yMin)) * plotH
  const tStep = niceStep(tMax - tMin)
  const yStep = niceStep(yMax - yMin)
  const W = canvasSize.w, H = canvasSize.h

  const parts = []
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="Inter Tight, sans-serif">`)
  parts.push(`<rect width="${W}" height="${H}" fill="#faf6ec"/>`)
  parts.push(`<rect x="${PAD_L}" y="${PAD_T}" width="${plotW}" height="${plotH}" fill="#faf6ec" stroke="#8a8170" stroke-width="1.5"/>`)
  // Grid
  let g = ''
  for (let t = Math.ceil(tMin / tStep) * tStep; t <= tMax; t += tStep) {
    const x = xToPx(t)
    g += `<line x1="${x}" y1="${PAD_T}" x2="${x}" y2="${PAD_T + plotH}" stroke="#e3dccb"/>`
  }
  for (let y = Math.ceil(yMin / yStep) * yStep; y <= yMax; y += yStep) {
    const py = yToPx(y)
    g += `<line x1="${PAD_L}" y1="${py}" x2="${PAD_L + plotW}" y2="${py}" stroke="#e3dccb"/>`
  }
  parts.push(g)
  // Slope field
  let f = ''
  const cellMin = Math.min(plotW / density, plotH / density)
  const segLen = cellMin * lengthMul
  for (let i = 0; i < density; i++) {
    for (let j = 0; j < density; j++) {
      const t = tMin + ((tMax - tMin) * (i + 0.5)) / density
      const y = yMin + ((yMax - yMin) * (j + 0.5)) / density
      const slope = evalF(compiled.code, t, y)
      if (!isFinite(slope)) continue
      const screenSlope = -slope * (plotH / (yMax - yMin)) / (plotW / (tMax - tMin))
      const theta = Math.atan(screenSlope)
      const cx = xToPx(t), cy = yToPx(y)
      const dx = (segLen / 2) * Math.cos(theta)
      const dy = (segLen / 2) * Math.sin(theta)
      f += `<line x1="${(cx - dx).toFixed(2)}" y1="${(cy - dy).toFixed(2)}" x2="${(cx + dx).toFixed(2)}" y2="${(cy + dy).toFixed(2)}" stroke="#2a2520" stroke-width="1.1" stroke-linecap="round"/>`
    }
  }
  parts.push(f)
  // Solution curves
  let s = ''
  solutions.forEach(sol => {
    const all = [...sol.bwd.slice().reverse(), ...sol.fwd.slice(1)]
    const d = all.map(([t, y], idx) => `${idx === 0 ? 'M' : 'L'}${xToPx(t).toFixed(2)},${yToPx(y).toFixed(2)}`).join(' ')
    s += `<path d="${d}" fill="none" stroke="${sol.color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>`
    s += `<circle cx="${xToPx(sol.t0).toFixed(2)}" cy="${yToPx(sol.y0).toFixed(2)}" r="4" fill="${sol.color}" stroke="#faf6ec" stroke-width="1.5"/>`
  })
  parts.push(s)
  // Phase line
  if (autonomous) {
    const px0 = PAD_L + plotW + 28
    const lineX = px0 + 24
    parts.push(`<rect x="${px0 - 6}" y="${PAD_T}" width="${phaseLineW - 16}" height="${plotH}" fill="#f0e9d6" stroke="#8a8170"/>`)
    parts.push(`<line x1="${lineX}" y1="${PAD_T + 6}" x2="${lineX}" y2="${PAD_T + plotH - 6}" stroke="#1a1715" stroke-width="2"/>`)
    const sorted = [...equilibria].sort((a, b) => a.y - b.y)
    sorted.forEach(eq => {
      const py = yToPx(eq.y)
      if (eq.kind === 'stable') {
        parts.push(`<circle cx="${lineX}" cy="${py}" r="5.5" fill="#1a1715"/>`)
      } else if (eq.kind === 'unstable') {
        parts.push(`<circle cx="${lineX}" cy="${py}" r="5.5" fill="#faf6ec" stroke="#1a1715" stroke-width="2"/>`)
      } else {
        parts.push(`<circle cx="${lineX}" cy="${py}" r="5.5" fill="#1a1715"/><circle cx="${lineX}" cy="${py}" r="5.5" fill="none" stroke="#1a1715" stroke-width="2"/>`)
      }
    })
  }
  // Title
  parts.push(`<text x="${PAD_L}" y="${PAD_T - 8}" font-family="Fraunces, serif" font-style="italic" font-weight="600" font-size="14" fill="#1a1715">y′ = ${escapeXml(expr)}</text>`)
  parts.push(`</svg>`)
  return parts.join('')
}

function escapeXml(s) {
  return String(s).replace(/[<>&'"]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', "'": '&apos;', '"': '&quot;' }[c]))
}

// ----- Styles -----
const presetBtnStyle = {
  width: '100%',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  background: 'transparent',
  border: '1px solid #1a1715',
  borderRadius: 3,
  padding: '6px 10px',
  cursor: 'pointer',
  color: '#1a1715',
  fontWeight: 600,
}
const presetMenuStyle = {
  position: 'absolute',
  top: 'calc(100% + 4px)',
  left: 0,
  right: 0,
  background: '#fffdf6',
  border: '1px solid #1a1715',
  borderRadius: 3,
  zIndex: 10,
  maxHeight: 320,
  overflowY: 'auto',
  boxShadow: '0 12px 24px -10px rgba(26,23,21,0.3)',
}
const presetItemStyle = {
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  padding: '8px 10px',
  background: 'transparent',
  border: 'none',
  borderBottom: '1px solid #ece5d2',
  cursor: 'pointer',
  textAlign: 'left',
  gap: 2,
  color: '#1a1715',
}
const segmentBtnStyle = {
  flex: 1,
  border: '1px solid #1a1715',
  borderRadius: 3,
  padding: '4px 8px',
  fontSize: 11,
  fontFamily: "'JetBrains Mono', monospace",
  cursor: 'pointer',
  fontWeight: 600,
}
const iconBtnStyle = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 6,
  background: 'transparent',
  border: '1px solid #1a1715',
  borderRadius: 3,
  padding: '6px 10px',
  cursor: 'pointer',
  color: '#1a1715',
  fontWeight: 600,
}
const badgeOn = {
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 9,
  background: '#1a1715',
  color: '#faf6ec',
  padding: '2px 6px',
  borderRadius: 2,
  letterSpacing: '0.05em',
  marginLeft: 'auto',
  fontWeight: 600,
}
const badgeOff = { ...badgeOn, background: '#d8d2c0', color: '#5a5347' }
