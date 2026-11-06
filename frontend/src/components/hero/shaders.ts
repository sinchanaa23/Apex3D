import {
  AdditiveBlending,
  Color,
  LineBasicMaterial,
  Matrix4,
  ShaderMaterial,
  type IUniform,
  type MeshPhysicalMaterial,
  type WebGLProgramParametersWithUniforms,
} from 'three'

export const ORANGE = '#ff6a2b'

/** Shared by every hero material, so one update per frame drives the whole car. */
export interface HoloUniforms {
  uTime: IUniform<number>
  /** Height of the scanning line in car space. */
  uScan: IUniform<number>
  /** 0 to 1 while a pulse expands from the car. Negative when idle. */
  uPulse: IUniform<number>
  /** Inverse of the car group's world matrix, so lines stay fixed to the car while it turns and while wheels spin. */
  uInvCar: IUniform<Matrix4>
  uOrange: IUniform<Color>
}

export function createHoloUniforms(): HoloUniforms {
  return {
    uTime: { value: 0 },
    uScan: { value: -5 },
    uPulse: { value: -1 },
    uInvCar: { value: new Matrix4() },
    uOrange: { value: new Color(ORANGE) },
  }
}

const SIGNALS = /* glsl */ `
uniform float uTime;
uniform float uScan;
uniform float uPulse;
uniform vec3 uOrange;
varying vec3 vCar;

// One-pixel-wide line at every integer of v, antialiased with screen derivatives.
float gridLine(float v, float w) {
  float d = abs(fract(v - 0.5) - 0.5);
  return 1.0 - smoothstep(0.0, fwidth(v) * w, d);
}
float scanBand(float y) { float d = y - uScan; return exp(-d * d * 700.0); }
// The grid lights up as the scan passes and fades behind it.
float scanTrail(float y) { float d = uScan - y; return d > 0.0 ? exp(-d * 1.8) : 0.0; }
float pulseShell(vec3 p) {
  if (uPulse < 0.0) return 0.0;
  float d = length(p - vec3(0.0, 0.55, 0.25)) - uPulse * 5.0;
  return exp(-d * d * 10.0) * (1.0 - uPulse);
}
`

const VERT_COMMON = /* glsl */ `
uniform mat4 uInvCar;
varying vec3 vCar;
`

const VERT_BODY = /* glsl */ `
vCar = (uInvCar * modelMatrix * vec4(transformed, 1.0)).xyz;
`

/** Contour lines are slices through the surface at fixed heights, lengths and widths, so they follow the real shape. */
const SURFACE_EMISSIVE = /* glsl */ `
{
  vec3 p = vCar;
  float g = max(gridLine(p.y * 8.0, 1.1), max(gridLine(p.z * 6.0, 1.0) * 0.85, gridLine(p.x * 6.0, 0.9) * 0.6));
  float band = scanBand(p.y);
  float trail = scanTrail(p.y);
  float pulse = pulseShell(p);
  float breathe = 0.5 + 0.5 * sin(uTime * 0.8 + p.z * 1.3);
  float energy = 0.16 + 0.12 * breathe + 0.95 * band + 0.55 * trail + 0.9 * pulse;
  vec3 col = mix(uOrange, vec3(1.0, 0.93, 0.85), clamp(band * 0.7 + pulse * 0.35, 0.0, 1.0));
  float fres = pow(1.0 - clamp(dot(normalize(vNormal), normalize(vViewPosition)), 0.0, 1.0), 2.6);
  totalEmissiveRadiance += col * (g * energy * uGrid * 1.25 + (band * 0.18 + pulse * 0.22) * uGrid);
  totalEmissiveRadiance += uOrange * fres * 0.5 * uRim;
}
`

export interface HoloOptions {
  /** How strongly the grid lights this material. */
  grid: number
  /** Strength of the orange edge glow. */
  rim: number
}

/** Adds the scan, grid and pulse to a physically based material, keeping its real lighting. */
export function patchSurface(material: MeshPhysicalMaterial, uniforms: HoloUniforms, { grid, rim }: HoloOptions) {
  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    Object.assign(shader.uniforms, uniforms, { uGrid: { value: grid }, uRim: { value: rim } })
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${VERT_COMMON}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${VERT_BODY}`)
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\nuniform float uGrid;\nuniform float uRim;\n${SIGNALS}`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n${SURFACE_EMISSIVE}`)
  }
  material.customProgramCacheKey = () => `apex-holo-surface-${grid}-${rim}`
  // Keeps the surface just behind the edge lines drawn on top of it.
  material.polygonOffset = true
  material.polygonOffsetFactor = 1
  material.polygonOffsetUnits = 1
}

/** White technical edge lines along panel seams, which brighten with the same scan. */
export function createEdgeMaterial(uniforms: HoloUniforms) {
  const material = new LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 1, depthWrite: false, blending: AdditiveBlending })
  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${VERT_COMMON}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${VERT_BODY}`)
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${SIGNALS}`)
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        {
          float band = scanBand(vCar.y);
          float pulse = pulseShell(vCar);
          diffuseColor.rgb = mix(vec3(0.95, 0.9, 0.85), uOrange * 1.4, clamp(band + pulse + scanTrail(vCar.y) * 0.5, 0.0, 1.0));
          diffuseColor.a *= 0.12 + 0.75 * band + 0.4 * scanTrail(vCar.y) + 0.6 * pulse;
        }`,
      )
  }
  material.customProgramCacheKey = () => 'apex-holo-edges'
  return material
}

/** Perspective grid under the car, with rings drifting outward and an expanding pulse. */
export function createFloorMaterial(uniforms: HoloUniforms) {
  return new ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
    uniforms: { uTime: uniforms.uTime, uPulse: uniforms.uPulse, uOrange: uniforms.uOrange },
    vertexShader: /* glsl */ `
      varying vec2 vW;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform float uPulse;
      uniform vec3 uOrange;
      varying vec2 vW;
      float gridLine(float v, float w) {
        float d = abs(fract(v - 0.5) - 0.5);
        return 1.0 - smoothstep(0.0, fwidth(v) * w, d);
      }
      void main() {
        float r = length(vW);
        float fade = 1.0 - smoothstep(0.8, 4.6, r);
        float minor = max(gridLine(vW.x * 2.0, 1.0), gridLine(vW.y * 2.0, 1.0));
        float major = max(gridLine(vW.x * 0.5, 1.4), gridLine(vW.y * 0.5, 1.4));
        float rings = gridLine(r * 1.4 - uTime * 0.12, 1.2);
        float pr = r - uPulse * 6.4;
        // A thin ring that eases in from the centre rather than flashing there.
        float pulse = uPulse >= 0.0 ? exp(-pr * pr * 22.0) * (1.0 - uPulse) * smoothstep(0.0, 0.2, uPulse) : 0.0;
        float glow = exp(-r * r * 0.7) * 0.1;
        vec3 col = vec3(1.0, 0.94, 0.88) * minor * 0.16 + uOrange * (major * 0.5 + rings * 0.3 + pulse * 1.1 + glow);
        float a = clamp(minor * 0.2 + major * 0.5 + rings * 0.3 + pulse + glow, 0.0, 1.0) * fade;
        gl_FragColor = vec4(col, a);
      }`,
  })
}

/** Drifting motes. Positions are computed on the GPU from a time uniform, so there is no per-frame JavaScript. */
export function createParticleMaterial(uniforms: HoloUniforms, pixelRatio: number) {
  return new ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
    uniforms: { uTime: uniforms.uTime, uOrange: uniforms.uOrange, uPixelRatio: { value: pixelRatio } },
    vertexShader: /* glsl */ `
      attribute float aSeed;
      uniform float uTime;
      uniform float uPixelRatio;
      varying float vSeed;
      varying float vFade;
      void main() {
        vec3 p = position;
        float rise = 0.04 + aSeed * 0.1;
        p.y = mod(p.y + 1.1 + uTime * rise, 3.8) - 1.1;
        float a = uTime * 0.07 * (0.4 + aSeed) + aSeed * 6.2831;
        float c = cos(a), s = sin(a);
        p.xz = mat2(c, -s, s, c) * p.xz;
        vSeed = aSeed;
        vFade = smoothstep(-1.1, -0.6, p.y) * (1.0 - smoothstep(1.9, 2.7, p.y));
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_PointSize = (2.0 + aSeed * 3.0) * uPixelRatio * (7.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uOrange;
      varying float vSeed;
      varying float vFade;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.0, d) * vFade;
        vec3 col = vSeed > 0.72 ? vec3(1.0, 0.95, 0.9) : uOrange;
        gl_FragColor = vec4(col, a * 0.7);
      }`,
  })
}
