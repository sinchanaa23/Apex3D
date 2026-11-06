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
| `src/api/` | The only code that knows the API formats. See [API_EXPECTATIONS.md](API_EXPECTATIONS.md) |
| `src/state/session.tsx` | Photo, identity and reference choices, held in memory only |
| `src/lib/image.ts` | Upload checks, plus rotating and resizing photos before upload |
| `src/pages/` | One file per page |
| `src/components/MeshViewer.tsx` | The 3D viewer |
| `src/components/LineChart.tsx` | Charts for the Research Lab |
