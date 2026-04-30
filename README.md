# Slope Field

Interactive slope field and phase line tool for first-order ODEs `y' = f(t, y)`.
Built for Math 252 (Differential Equations / Calc II) at the College of San Mateo.

## Features

- Live `f(t, y)` parser (math.js) with inline error display
- Adjustable `t` and `y` window
- Density and segment-length sliders with auto-scaling
- Uniform-length segments (rendered with `θ = atan(f)`) so steep slopes don't dominate
- Optional direction arrows (true vector field visualization)
- Optional isoclines including the nullcline
- Click to integrate the IVP through `(t₀, y₀)` — RK4 in *both* directions
- Multiple solution curves with distinct colors; clear-all button
- Automatic detection of autonomous equations (`f` independent of `t`)
- Phase line panel for autonomous ODEs:
  - Equilibria found by sign-change bisection
  - Stable / unstable / semi-stable classification with filled / open / split dots
  - Direction arrows in each subinterval
  - Faint horizontal guides connecting the phase line to the slope field
- 10 built-in presets (logistic, three-equilibria, Newton cooling, Riccati-like, etc.)
- SVG export — drops cleanly into LaTeX worksheets and exam keys

## Local development

```bash
npm install
npm run dev
```

## Build for deployment

```bash
npm run build
```

Output goes to `dist/`.

## Netlify

Build command: `npm run build`
Publish directory: `dist`
