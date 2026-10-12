// Single spherical gas bubble in a sound field.
// Rayleigh-Plesset or Keller-Miksis equation, polytropic gas, surface tension, viscosity, vapour pressure.
// Theoretical model only: built from published equations, no measured data and no data from any research centre.
// Pure functions, no DOM. SI units inside.

// Liquid pressure at the bubble wall and the parts needed for its time derivative.
function wallPressure(R, V, P) {
  const pg = P.pg0 * Math.pow(P.R0 / R, 3 * P.kappa);           // gas pressure
  const pB = pg + P.pv - 2 * P.sigma / R - 4 * P.mu * V / R;      // liquid pressure at the wall
  return { pg, pB };
}

// Far-field pressure: ambient plus the sound field. Rarefaction first, so the bubble grows first.
const pInf = (t, P) => P.p0 - P.pA * Math.sin(2 * Math.PI * P.f * t);

// Wall acceleration. model 'KM' = Keller-Miksis (driving evaluated at t, retardation R/c neglected),
// model 'RP' = Rayleigh-Plesset (incompressible liquid).
function accel(t, R, V, P) {
  const { pg, pB } = wallPressure(R, V, P);
  const dp = pB - pInf(t, P);
  if (P.model === 'RP') return (dp / P.rho - 1.5 * V * V) / R;
  const c = P.c, rho = P.rho;
  // d(pB)/dt = dpg/dR V + 2 sigma V / R^2 - 4 mu (Rdd / R - V^2 / R^2); the Rdd part moves to the left side
  const dpgdR = -3 * P.kappa * pg / R;
  const rhs = (1 + V / c) * dp / rho
            + (R / (rho * c)) * (dpgdR * V + 2 * P.sigma * V / (R * R) + 4 * P.mu * V * V / (R * R))
            - 1.5 * (1 - V / (3 * c)) * V * V;
  const lhs = (1 - V / c) * R + 4 * P.mu / (rho * c);
  return rhs / lhs;
}

// Build a parameter object from user units. Everything SI afterwards.
function makeParams(u) {
  const P = {
    rho: u.rho, sigma: u.sigma, mu: u.mu, pv: u.pv, c: u.c, p0: u.p0,
    R0: u.R0, kappa: u.kappa, f: u.f, pA: u.pA, model: u.model || 'KM',
  };
  P.pg0 = (u.gasScale === undefined ? 1 : u.gasScale) * (P.p0 + 2 * P.sigma / P.R0 - P.pv);  // equilibrium gas pressure
  return P;
}

// Dormand-Prince 5(4) with step-size control, in dimensionless variables
// r = R/R0, v = (dR/dt) / (R0 f), tau = t f. Returns the accepted points.
function simulate(P, cycles, opts) {
  opts = opts || {};
  const rtol = opts.rtol || 1e-9, atol = opts.atol || 1e-12;
  const maxSteps = opts.maxSteps || 400000;
  const R0 = P.R0, f = P.f;
  const F = (tau, y) => {
    const R = R0 * y[0], V = R0 * f * y[1];
    if (!(R > 0)) return null;
    const a = accel(tau / f, R, V, P);
    return [y[1], a / (R0 * f * f)];
  };
  // Dormand-Prince coefficients
  const c2 = 1 / 5, c3 = 3 / 10, c4 = 4 / 5, c5 = 8 / 9;
  const a21 = 1 / 5;
  const a31 = 3 / 40, a32 = 9 / 40;
  const a41 = 44 / 45, a42 = -56 / 15, a43 = 32 / 9;
  const a51 = 19372 / 6561, a52 = -25360 / 2187, a53 = 64448 / 6561, a54 = -212 / 729;
  const a61 = 9017 / 3168, a62 = -355 / 33, a63 = 46732 / 5247, a64 = 49 / 176, a65 = -5103 / 18656;
  const b1 = 35 / 384, b3 = 500 / 1113, b4 = 125 / 192, b5 = -2187 / 6784, b6 = 11 / 84;
  const e1 = 71 / 57600, e3 = -71 / 16695, e4 = 71 / 1920, e5 = -17253 / 339200, e6 = 22 / 525, e7 = -1 / 40;
  let tau = 0, y = [opts.r0 === undefined ? 1 : opts.r0, opts.v0 === undefined ? 0 : opts.v0];
  let h = 1e-4;
  const tEnd = cycles;
  const out = { tau: [0], r: [y[0]], v: [y[1]], steps: 0, rejected: 0, stopped: null };
  let k1 = F(tau, y);
  if (!k1) { out.stopped = 'invalid start'; return out; }
  while (tau < tEnd) {
    if (out.steps > maxSteps) { out.stopped = 'too many steps'; break; }
    if (tau + h > tEnd) h = tEnd - tau;
    const Y = (k, coef) => [y[0] + h * coef.reduce((s, cf, i) => s + cf * k[i][0], 0), y[1] + h * coef.reduce((s, cf, i) => s + cf * k[i][1], 0)];
    const k2 = F(tau + c2 * h, Y([k1], [a21]));
    const k3 = k2 && F(tau + c3 * h, Y([k1, k2], [a31, a32]));
    const k4 = k3 && F(tau + c4 * h, Y([k1, k2, k3], [a41, a42, a43]));
    const k5 = k4 && F(tau + c5 * h, Y([k1, k2, k3, k4], [a51, a52, a53, a54]));
    const k6 = k5 && F(tau + h, Y([k1, k2, k3, k4, k5], [a61, a62, a63, a64, a65]));
    let yn = null, k7 = null;
    if (k6) {
      yn = Y([k1, k2, k3, k4, k5, k6], [b1, 0, b3, b4, b5, b6]);
      k7 = yn[0] > 0 ? F(tau + h, yn) : null;
    }
    if (!k7) { h *= 0.25; out.rejected++; if (h < 1e-16) { out.stopped = 'step size underflow'; break; } continue; }
    let err = 0;
    for (let i = 0; i < 2; i++) {
      const ei = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
      const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(yn[i]));
      err = Math.max(err, Math.abs(ei) / sc);
    }
    if (err <= 1) {
      tau += h; y = yn; k1 = k7; out.steps++;
      out.tau.push(tau); out.r.push(y[0]); out.v.push(y[1]);
      if (y[0] > (opts.rStop || 1e3)) { out.stopped = 'unbounded growth'; break; }
      h *= Math.min(5, Math.max(0.2, 0.9 * Math.pow(err || 1e-10, -0.2)));
    } else {
      out.rejected++;
      h *= Math.max(0.1, 0.9 * Math.pow(err, -0.25));
    }
  }
  return out;
}

// Summary numbers of a run.
function summarize(P, run) {
  let rmax = 1, rmin = 1, vmax = 0, iMax = 0;
  for (let i = 0; i < run.r.length; i++) {
    if (run.r[i] > rmax) { rmax = run.r[i]; iMax = i; }
    vmax = Math.max(vmax, Math.abs(run.v[i]));
  }
  // smallest radius reached after the first maximum; NaN if the run ends at the maximum or grows without bound
  rmin = Infinity;
  for (let i = iMax + 1; i < run.r.length; i++) rmin = Math.min(rmin, run.r[i]);
  if (!isFinite(rmin) || run.stopped === 'unbounded growth') rmin = NaN;
  // if the smallest radius is the last point and the wall is still moving inwards, the collapse was cut off
  const last = run.r.length - 1;
  const stillShrinking = isFinite(rmin) && run.r[last] === rmin && run.v[last] < 0;
  const speed = vmax * P.R0 * P.f;
  return { rmax, rmin, stillShrinking, speed, mach: speed / P.c, steps: run.steps, stopped: run.stopped };
}

// Linear (small-amplitude) natural frequency in Hz, inviscid.
function naturalFrequency(R0, u) {
  const pg0 = u.p0 + 2 * u.sigma / R0 - u.pv;
  const w2 = (3 * u.kappa * pg0 - 2 * u.sigma / R0) / (u.rho * R0 * R0);
  return w2 > 0 ? Math.sqrt(w2) / (2 * Math.PI) : NaN;
}

// Radius whose linear natural frequency equals f (bisection on a log scale).
function resonantRadius(f, u) {
  let lo = 1e-8, hi = 1e-2;
  if (!(naturalFrequency(lo, u) > f && naturalFrequency(hi, u) < f)) return NaN;
  for (let i = 0; i < 100; i++) {
    const m = Math.sqrt(lo * hi);
    if (naturalFrequency(m, u) > f) lo = m; else hi = m;
  }
  return Math.sqrt(lo * hi);
}

// Blake threshold (quasi-static, isothermal gas): critical radius and the sound amplitude needed.
function blake(R0, u) {
  const pg0 = u.p0 + 2 * u.sigma / R0 - u.pv;
  const Rc = R0 * Math.sqrt(3 * pg0 * R0 / (2 * u.sigma));
  const pLiquidMin = u.pv - 4 * u.sigma / (3 * Rc);   // lowest liquid pressure the bubble can be in equilibrium with
  return { Rc, pA: u.p0 - pLiquidMin };
}

// Smallest sound amplitude for which R reaches 2 R0 within the given number of cycles.
function dynamicThreshold(R0, u, cycles) {
  const reaches = (pA) => {
    const P = makeParams(Object.assign({}, u, { R0, pA }));
    const run = simulate(P, cycles, { rStop: 2.5, rtol: 1e-7, atol: 1e-10, maxSteps: 60000 });
    if (run.stopped === 'unbounded growth') return true;
    return summarize(P, run).rmax >= 2;
  };
  let lo = 1e2, hi = 1e7;           // Pa
  if (reaches(lo)) return lo;
  if (!reaches(hi)) return NaN;
  for (let i = 0; i < 22; i++) {
    const m = Math.sqrt(lo * hi);
    if (reaches(m)) hi = m; else lo = m;
    if (hi / lo < 1.005) break;
  }
  return Math.sqrt(lo * hi);
}

if (typeof module !== 'undefined') {
  module.exports = { wallPressure, pInf, accel, makeParams, simulate, summarize, naturalFrequency, resonantRadius, blake, dynamicThreshold };
}
