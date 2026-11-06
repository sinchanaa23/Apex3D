# Apex3D frontend

React, TypeScript, Vite, Tailwind and React Three Fiber. Owned by Person D.

## Run it

Needs Node 22 or newer.

```bash
npm install
```

Without a backend, using test responses. A yellow banner marks every page:

```bash
npm run dev:mock
```

Against Person C's API on `http://127.0.0.1:8000`. Set `APEX3D_API` to use another address:

```bash
npm run dev
```

Build for FastAPI to serve, which writes `dist/`:

```bash
npm run build
```

Run the unit tests:

```bash
npm test
```

## Test responses stay out of the build

`dev/mockApi.ts` is a Vite plugin that only runs on the dev server in `mock` mode. `src/` never imports it. Before the freeze, confirm this finds nothing:

```bash
npm run build && grep -rlE "MOCK|__mock|Dev mock" dist
```

To see other states while using the mock, open one of these in the browser, then go back to the app:

| Address | Options |
|---|---|
| `/__mock/set?recognize=` | `confident`, `ambiguous`, `rejected`, `nocar`, `error` |
| `/__mock/set?classical=` | `ok`, `error` |
| `/__mock/set?references=` | `ok`, `empty`, `error` |
| `/__mock/set?job=` | `ok`, `fail` |
| `/__mock/set?metrics=` | `partial`, `full`, `none`, `malformed`, `error` |

## Where things are

| Path | What it does |
|---|---|
| `src/api/` | The only code that knows the API formats. See [API_NOTES.md](API_NOTES.md) |
| `src/state/session.tsx` | Photo, identity and reference choices, held in memory only |
| `src/lib/image.ts` | Upload checks, plus rotating and resizing photos before upload |
| `src/pages/` | One file per page |
| `src/components/MeshViewer.tsx` | The 3D viewer |
| `src/components/LineChart.tsx` | Charts for the Research Lab |

## The hero car

The Home page shows a rotating 3D car (`src/components/hero/`). It is real geometry, not an image:

- **Model:** `public/models/car.glb`, derived from Khronos' "Car Concept" (CC BY 4.0). Credit and the list of changes are in `public/models/CREDITS.md`, and the credit also appears at the bottom of the Home page. Keep both.
- **Effects:** the orange grid is computed in a shader from each point's position on the car, so it follows the surface. The scan line, pulses, white edge lines, floor grid and particles all run on the GPU.
- **Loading:** the scene is a separate download, so the headline and buttons appear first. If WebGL is unavailable the hero shows text only.
- **Motion:** respects reduced-motion (still pose, no particles). Rendering pauses when the hero is off screen.
- **Input:** the canvas ignores pointer input, so scrolling is never blocked. Mouse movement nudges the camera slightly.
