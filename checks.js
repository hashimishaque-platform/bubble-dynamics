// The checks the page runs on load, for Node. Usage: node checks.js
'use strict';
const B = require('./bubble.js');
const w = { rho: 998.2, sigma: 0.0728, mu: 1.002e-3, pv: 2339, c: 1482, p0: 101325, kappa: 1.4 };
const rows = [];
const fmt = (x, d) => Number(x).toPrecision(d);
// 1. small free oscillation, inviscid Rayleigh-Plesset: period against linear theory
{
  const u = Object.assign({}, w, { mu: 0, R0: 3e-6, pA: 0, model: 'RP' });
  const f0 = B.naturalFrequency(u.R0, u);
  const run = B.simulate(B.makeParams(Object.assign({}, u, { f: f0 })), 10, { r0: 1.001 });
  const cr = [];
  for (let i = 1; i < run.r.length; i++) if (run.r[i - 1] > 1 && run.r[i] <= 1) cr.push(run.tau[i - 1] + (run.tau[i] - run.tau[i - 1]) * (run.r[i - 1] - 1) / (run.r[i - 1] - run.r[i]));
  rows.push(['Period of a small free oscillation / linear-theory period', '1', fmt((cr[cr.length - 1] - cr[0]) / (cr.length - 1), 7)]);
}
// 2. Minnaert constant for air in water without surface tension
{
  const u = Object.assign({}, w, { sigma: 0, pv: 0 });
  rows.push(['Minnaert constant f0 R0 (m/s)', fmt(Math.sqrt(3 * 1.4 * 101325 / 998.2) / (2 * Math.PI), 4), fmt(B.naturalFrequency(1e-3, u) * 1e-3, 4)]);
}
// 3. Rayleigh collapse of an almost empty cavity
{
  const R0 = 1e-4, tc = 0.914681 * R0 * Math.sqrt(w.rho / w.p0);
  const run = B.simulate(B.makeParams(Object.assign({}, w, { sigma: 0, mu: 0, pv: 0, R0, pA: 0, model: 'RP', f: 1 / tc, gasScale: 1e-9 })), 1.2, { rtol: 1e-11, atol: 1e-14 });
  let k = 0; for (let i = 0; i < run.r.length; i++) if (run.r[i] < run.r[k]) k = i;
  rows.push(["Collapse time of an empty cavity / Rayleigh's 0.915 R0 (rho/p0)^0.5", '1', fmt(run.tau[k], 6)]);
}
// 4. Blake threshold with slow driving (half a cycle at 20 kHz, 1 um bubble, isothermal gas)
{
  const R0 = 1e-6, wi = Object.assign({}, w, { kappa: 1 });
  const bl = B.blake(R0, wi);
  for (const k of [0.97, 1.03]) {
    const P = B.makeParams(Object.assign({}, wi, { R0, f: 2e4, pA: k * bl.pA, model: 'RP' }));
    const run = B.simulate(P, 0.5, { rStop: 50 });
    const s = B.summarize(P, run);
    rows.push([`Slow driving at ${k} x the Blake threshold: largest radius / R0`, k < 1 ? `below ${fmt(bl.Rc / R0, 3)}` : `far beyond ${fmt(bl.Rc / R0, 3)}`, run.stopped === 'unbounded growth' ? 'unbounded growth' : fmt(s.rmax, 3)]);
  }
}
// 5. Keller-Miksis approaches Rayleigh-Plesset for a very large speed of sound
{
  const base = Object.assign({}, w, { R0: 5e-6, f: 4e4, pA: 1.5e5 });
  const a = B.summarize(B.makeParams(base), B.simulate(B.makeParams(Object.assign({}, base, { model: 'RP' })), 1));
  const b = B.summarize(B.makeParams(base), B.simulate(B.makeParams(Object.assign({}, base, { model: 'KM', c: 1e9 })), 1));
  rows.push(['Largest radius, Keller-Miksis with c = 1e9 m/s / Rayleigh-Plesset', '1', fmt(b.rmax / a.rmax, 7)]);
}
const wid = Math.max(...rows.map((r) => r[0].length));
console.log('Check'.padEnd(wid), '| Expected       | This program');
for (const r of rows) console.log(r[0].padEnd(wid), '|', r[1].padEnd(14), '|', r[2]);
