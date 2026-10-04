"""
bubble_dynamics.py
==================

Acoustic cavitation of a single spherical gas bubble in an ultrasonic or
megasonic field: Rayleigh–Plesset / Keller–Miksis dynamics, the Blake
threshold, and maps of where cavitation stays gentle (stable) and where it
turns damaging (inertial / transient).

Model
-----
A bubble of equilibrium radius R0 sits in a liquid whose far-field pressure is

    p_inf(t) = p0 - Pa * sin(2*pi*f*t)            (rarefaction half-cycle first)

Keller–Miksis equation (first-order liquid compressibility; Keller & Miksis 1980):

    (1 - U/c) R R'' + 3/2 (1 - U/(3c)) U^2
          = (1 + U/c) (p_B - p_inf)/rho + R/(rho c) * d p_B/dt ,     U = R'

Rayleigh–Plesset equation (incompressible limit c -> infinity):

    R R'' + 3/2 U^2 = (p_B - p_inf)/rho

Liquid pressure at the bubble wall:

    p_B = p_g + p_v - 2 sigma/R - 4 (mu + mu_th) U/R

Gas content: polytropic with a van der Waals hard core h = R0/8.86 (air),

    p_g = p_g0 * [(R0^3 - h^3)/(R^3 - h^3)]^kappa ,   p_g0 = p0 - p_v + 2 sigma/R0

Thermal behaviour of the gas (`thermal=` option):
  * "prosperetti" (default) – effective polytropic exponent kappa(R0, f) and
    thermal-damping viscosity mu_th(R0, f) from linear theory
    (Prosperetti, Crum & Commander, JASA 83, 502, 1988). Small, slow bubbles
    behave isothermally (kappa -> 1), large/fast ones adiabatically (kappa -> gamma).
  * "adiabatic"  – kappa = gamma, mu_th = 0
  * "isothermal" – kappa = 1,     mu_th = 0

Thresholds and criteria
-----------------------
Blake threshold (static, isothermal; Blake 1949, see Leighton 1994 / Brennen 1995):

    P_Blake = p0 - p_v + (2/(3*sqrt(3))) * sqrt( (2 sigma/R0)^3 / (p0 - p_v + 2 sigma/R0) )

It is frequency-independent; it is the true threshold only when the drive is
slow compared with the bubble's own response (f << f_res).

Dynamic (inertial-cavitation) threshold: the bubble is classed as inertial –
violent, damaging collapse – when it expands to at least twice its equilibrium
radius during the simulated cycles, R_max/R0 >= 2 (Flynn 1975; Apfel 1981).

Linear natural (resonance) frequency, surface tension included:

    omega0^2 = [3 kappa p_g0 R0^3/(R0^3 - h^3) - 2 sigma/R0] / (rho R0^2)

Numerics
--------
The ODEs are written in non-dimensional form (x = R/R0, time in acoustic
cycles) and integrated with an adaptive Rosenbrock method (Rodas4: order 4(3),
L-stable, stiffly accurate) with an analytic Jacobian. It copes with the
stiff, over-damped small-bubble limit as well as the violent collapse.
If numba is installed the core is JIT-compiled and maps are computed in
parallel; without numba everything still works (single runs are fine, maps
are slow). `simulate(..., method="Radau")` cross-checks against SciPy.

Assumptions / limits: spherical bubble, uniform gas, constant vapour pressure,
no mass transfer (rectified diffusion, dissolution) or fragmentation, single
bubble in an unbounded liquid (no wall, no neighbours), kR << 1.

Usage
-----
    python bubble_dynamics.py                          # demo + figures
    python bubble_dynamics.py --simulate 2 1000 300    # R0=2 um, f=1000 kHz, Pa=300 kPa
    python bubble_dynamics.py --figure maps.png --freqs 40e3 1e6

References
----------
Keller & Miksis, JASA 68, 628 (1980) · Prosperetti, Crum & Commander, JASA 83, 502 (1988)
Flynn, JASA 57, 1379 (1975) · Apfel, Methods Exp. Phys. 19, 355 (1981)
Leighton, The Acoustic Bubble (1994) · Brennen, Cavitation and Bubble Dynamics (1995)
Lauterborn & Kurz, Rep. Prog. Phys. 73, 106501 (2010)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

try:  # optional JIT acceleration
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - fallback without numba
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(func):
            return func

        return deco


P_ATM = 101325.0
INERTIAL_RATIO = 2.0  # Flynn / Apfel criterion R_max/R0 >= 2
ATOL_X = 1e-7  # absolute tolerance on R/R0
ATOL_U = 1e-4  # absolute tolerance on (dR/dt)/(R0 f)

# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Liquid:
    """Host liquid. SI units."""

    name: str = "Water, 20 °C"
    rho: float = 998.2  # density, kg/m^3
    mu: float = 1.002e-3  # dynamic viscosity, Pa s
    sigma: float = 0.0728  # surface tension, N/m
    c: float = 1482.0  # speed of sound, m/s
    pv: float = 2339.0  # vapour pressure, Pa
    T0: float = 293.15  # temperature, K


WATER_20C = Liquid()
WATER_40C = Liquid("Water, 40 °C", 992.2, 0.653e-3, 0.0696, 1529.0, 7385.0, 313.15)
WATER_60C = Liquid("Water, 60 °C", 983.2, 0.467e-3, 0.0662, 1551.0, 19946.0, 333.15)
WATER_SURFACTANT = Liquid("Water + surfactant, 20 °C", 998.2, 1.002e-3, 0.035, 1482.0, 2339.0, 293.15)
IPA_20C = Liquid("Isopropanol, 20 °C", 786.0, 2.37e-3, 0.0230, 1170.0, 4400.0, 293.15)

LIQUIDS = {
    "water20": WATER_20C,
    "water40": WATER_40C,
    "water60": WATER_60C,
    "surfactant": WATER_SURFACTANT,
    "ipa": IPA_20C,
}


@dataclass(frozen=True)
class Gas:
    """Non-condensable gas inside the bubble (default: air)."""

    name: str = "Air"
    gamma: float = 1.4  # ratio of specific heats
    D_ref: float = 2.15e-5  # thermal diffusivity at p_ref, m^2/s
    p_ref: float = P_ATM
    hard_core_ratio: float = 8.86  # R0/h for the van der Waals hard core


AIR = Gas()


@dataclass(frozen=True)
class Settings:
    """Physical model choices."""

    model: str = "KM"  # "KM" (Keller–Miksis) or "RP" (Rayleigh–Plesset)
    thermal: str = "prosperetti"  # "prosperetti", "adiabatic" or "isothermal"
    p0: float = P_ATM  # static (ambient) pressure, Pa
    liquid: Liquid = field(default_factory=lambda: WATER_20C)
    gas: Gas = field(default_factory=lambda: AIR)

    def with_(self, **kw) -> "Settings":
        return replace(self, **kw)


DEFAULT = Settings()

# ---------------------------------------------------------------------------
# Equilibrium, Blake threshold, thermal parameters, resonance
# ---------------------------------------------------------------------------


def gas_pressure_eq(R0, s: Settings = DEFAULT):
    """Equilibrium partial pressure of the gas, p_g0 = p0 - p_v + 2 sigma/R0."""
    R0 = np.asarray(R0, dtype=float)
    return s.p0 - s.liquid.pv + 2.0 * s.liquid.sigma / R0


def blake_threshold(R0, s: Settings = DEFAULT):
    """Static (Blake) cavitation threshold: acoustic amplitude above which a
    nucleus of radius R0 grows without bound under quasi-static tension. [Pa]"""
    R0 = np.asarray(R0, dtype=float)
    lap = 2.0 * s.liquid.sigma / R0
    pg0 = s.p0 - s.liquid.pv + lap
    return s.p0 - s.liquid.pv + (2.0 / (3.0 * math.sqrt(3.0))) * np.sqrt(lap**3 / pg0)


def blake_critical_radius(R0, s: Settings = DEFAULT):
    """Radius at which quasi-static equilibrium is lost (the Blake critical radius). [m]"""
    R0 = np.asarray(R0, dtype=float)
    pg0 = gas_pressure_eq(R0, s)
    return R0 * np.sqrt(3.0 * pg0 * R0 / (2.0 * s.liquid.sigma))


def _zcothz_minus_1(z):
    """z*coth(z) - 1 for complex z with Re z > 0, accurate for small |z|."""
    z = np.asarray(z, dtype=complex)
    out = np.empty_like(z)
    small = np.abs(z) < 0.1
    zs = z[small]
    z2 = zs * zs
    out[small] = z2 / 3.0 - z2**2 / 45.0 + 2.0 * z2**3 / 945.0 - z2**4 / 4725.0
    zl = z[~small]
    e = np.exp(-2.0 * zl)
    out[~small] = zl * (1.0 + e) / (1.0 - e) - 1.0
    return out


def thermal_parameters(R0, f, s: Settings = DEFAULT):
    """Effective polytropic exponent kappa and thermal-damping viscosity mu_th.

    'prosperetti' uses linear theory (Prosperetti, Crum & Commander 1988):
        Phi    = 3 gamma / (1 - 3 (gamma-1) i chi [ sqrt(i/chi) coth(sqrt(i/chi)) - 1 ])
        chi    = D / (omega R0^2),   D = thermal diffusivity of the gas in the bubble
        kappa  = Re(Phi) / 3
        mu_th  = p_g0 Im(Phi) / (4 omega)
    Returns arrays broadcast over R0 and f.
    """
    R0, f = np.broadcast_arrays(np.asarray(R0, float), np.asarray(f, float))
    g = s.gas.gamma
    if s.thermal == "adiabatic":
        return np.full(R0.shape, g), np.zeros(R0.shape)
    if s.thermal == "isothermal":
        return np.ones(R0.shape), np.zeros(R0.shape)
    if s.thermal != "prosperetti":
        raise ValueError(f"unknown thermal model {s.thermal!r}")
    omega = 2.0 * np.pi * f
    pg0 = gas_pressure_eq(R0, s)
    p_in = s.p0 + 2.0 * s.liquid.sigma / R0  # total pressure inside (gas + vapour)
    D = s.gas.D_ref * s.gas.p_ref / p_in  # diffusivity scales with 1/density
    chi = D / (omega * R0**2)
    z = (1.0 + 1.0j) / np.sqrt(2.0 * chi)  # sqrt(i/chi)
    Phi = 3.0 * g / (1.0 - 3.0 * (g - 1.0) * 1.0j * chi * _zcothz_minus_1(z))
    kappa = Phi.real / 3.0
    mu_th = pg0 * Phi.imag / (4.0 * omega)
    return kappa, mu_th


def natural_frequency(R0, s: Settings = DEFAULT, f_drive: Optional[float] = None):
    """Linear (undamped) natural frequency of a bubble of radius R0 [Hz].

    With the 'prosperetti' model kappa depends on frequency; it is evaluated at
    f_drive if given, otherwise solved self-consistently (f = f0(kappa(f)))."""
    R0 = np.atleast_1d(np.asarray(R0, float))
    liq = s.liquid
    h3 = (R0 / s.gas.hard_core_ratio) ** 3
    pg0 = gas_pressure_eq(R0, s)

    def f0_of(kappa):
        stiff = 3.0 * kappa * pg0 * R0**3 / (R0**3 - h3) - 2.0 * liq.sigma / R0
        return np.sqrt(stiff / (liq.rho * R0**2)) / (2.0 * np.pi)

    if s.thermal != "prosperetti":
        kappa, _ = thermal_parameters(R0, 1.0, s)
        return f0_of(kappa)
    if f_drive is not None:
        kappa, _ = thermal_parameters(R0, f_drive, s)
        return f0_of(kappa)
    f = f0_of(np.ones_like(R0))  # fixed-point iteration from the isothermal value
    for _ in range(60):
        kappa, _ = thermal_parameters(R0, f, s)
        f_new = f0_of(kappa)
        if np.all(np.abs(f_new - f) < 1e-10 * f):
            break
        f = f_new
    return f_new


def resonance_radius(f: float, s: Settings = DEFAULT) -> float:
    """Equilibrium radius whose natural frequency equals f (Minnaert size incl.
    surface tension and thermal effects) [m]."""
    lo, hi = math.log(1e-9), math.log(1e-1)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f0 = natural_frequency(math.exp(mid), s, f_drive=f)[0]
        if f0 > f:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Equations of motion (non-dimensional): x = R/R0, u = (dR/dt)/(R0 f), tau = f t
# ---------------------------------------------------------------------------
# Parameter vector layout (float64 array of length 13)
I_RHO, I_MU, I_SIG, I_C, I_PV, I_P0, I_PG0, I_KAP, I_R0, I_H, I_PA, I_F, I_KM = range(13)
TWO_PI = 2.0 * math.pi


def param_vector(R0: float, f: float, Pa: float, s: Settings = DEFAULT) -> np.ndarray:
    """Pack all quantities the integrator needs into one float64 array."""
    kappa, mu_th = thermal_parameters(R0, f, s)
    liq = s.liquid
    pg0 = float(gas_pressure_eq(R0, s))
    if pg0 <= 0:
        raise ValueError("no equilibrium: vapour pressure exceeds p0 + 2 sigma/R0")
    P = np.empty(13)
    P[I_RHO], P[I_MU], P[I_SIG], P[I_C] = liq.rho, liq.mu + float(mu_th), liq.sigma, liq.c
    P[I_PV], P[I_P0], P[I_PG0], P[I_KAP] = liq.pv, s.p0, pg0, float(kappa)
    P[I_R0], P[I_H], P[I_PA], P[I_F] = R0, R0 / s.gas.hard_core_ratio, Pa, f
    P[I_KM] = 1.0 if s.model.upper() == "KM" else 0.0
    return P


@njit(cache=True)
def _accel(tau, x, u, P):
    """Non-dimensional wall acceleration a = du/dtau (x = R/R0, u = R'/(R0 f), tau = f t)."""
    R0 = P[8]
    f = P[11]
    R = R0 * x
    U = R0 * f * u
    h = P[9]
    h3 = h * h * h
    D = R * R * R - h3
    if D <= 0.0:
        return np.nan
    rho = P[0]
    mu = P[1]
    sig = P[2]
    c = P[3]
    kap = P[7]
    pg = P[6] * ((R0 * R0 * R0 - h3) / D) ** kap
    pinf = P[5] - P[10] * math.sin(TWO_PI * tau)
    pB = pg + P[4] - 2.0 * sig / R - 4.0 * mu * U / R
    if P[12] > 0.5:  # Keller–Miksis
        dpg = -3.0 * kap * pg * R * R * U / D
        num = ((1.0 + U / c) * (pB - pinf) / rho
               + R / (rho * c) * (dpg + 2.0 * sig * U / (R * R) + 4.0 * mu * U * U / (R * R))
               - 1.5 * (1.0 - U / (3.0 * c)) * U * U)
        acc = num / ((1.0 - U / c) * R + 4.0 * mu / (rho * c))
    else:  # Rayleigh–Plesset
        acc = ((pB - pinf) / rho - 1.5 * U * U) / R
    return acc / (R0 * f * f)


@njit(cache=True)
def _accel_jac(tau, x, u, P):
    """a, da/dx, da/du and the explicit time derivative da/dtau (analytic)."""
    R0 = P[8]
    f = P[11]
    R = R0 * x
    U = R0 * f * u
    h = P[9]
    h3 = h * h * h
    D = R * R * R - h3
    if D <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    rho = P[0]
    mu = P[1]
    sig = P[2]
    c = P[3]
    kap = P[7]
    Pa = P[10]
    pg = P[6] * ((R0 * R0 * R0 - h3) / D) ** kap
    ph = TWO_PI * tau
    pinf = P[5] - Pa * math.sin(ph)
    dpinf = -Pa * TWO_PI * math.cos(ph)  # d p_inf / d tau
    pB = pg + P[4] - 2.0 * sig / R - 4.0 * mu * U / R
    dpg_dR = -3.0 * kap * pg * R * R / D
    dpB_dR = dpg_dR + 2.0 * sig / (R * R) + 4.0 * mu * U / (R * R)
    dpB_dU = -4.0 * mu / R
    if P[12] > 0.5:  # Keller–Miksis
        dpg = dpg_dR * U  # = dp_g/dt
        Q = dpg + 2.0 * sig * U / (R * R) + 4.0 * mu * U * U / (R * R)
        num = ((1.0 + U / c) * (pB - pinf) / rho + R * Q / (rho * c)
               - 1.5 * (1.0 - U / (3.0 * c)) * U * U)
        den = (1.0 - U / c) * R + 4.0 * mu / (rho * c)
        acc = num / den
        ddpg_dR = -3.0 * kap * U * pg * (2.0 * R / D - 3.0 * (kap + 1.0) * R ** 4 / (D * D))
        dQ_dR = ddpg_dR - 4.0 * sig * U / R ** 3 - 8.0 * mu * U * U / R ** 3
        dQ_dU = dpg_dR + 2.0 * sig / (R * R) + 8.0 * mu * U / (R * R)
        dnum_dR = (1.0 + U / c) * dpB_dR / rho + (Q + R * dQ_dR) / (rho * c)
        dnum_dU = ((pB - pinf) / (rho * c) + (1.0 + U / c) * dpB_dU / rho + R * dQ_dU / (rho * c)
                   - (3.0 * U - 1.5 * U * U / c))
        dacc_dR = (dnum_dR - acc * (1.0 - U / c)) / den
        dacc_dU = (dnum_dU + acc * R / c) / den
        dacc_dtau = -(1.0 + U / c) * dpinf / rho / den
    else:  # Rayleigh–Plesset
        acc = ((pB - pinf) / rho - 1.5 * U * U) / R
        dacc_dR = dpB_dR / (rho * R) - acc / R
        dacc_dU = (dpB_dU / rho - 3.0 * U) / R
        dacc_dtau = -dpinf / (rho * R)
    sc = 1.0 / (R0 * f * f)
    # chain rule to non-dimensional variables: dR/dx = R0, dU/du = R0 f
    return acc * sc, dacc_dR * R0 * sc, dacc_dU * R0 * f * sc, dacc_dtau * sc


# Rodas4 (Hairer & Wanner): 6-stage, order 4(3), L-stable and stiffly accurate
_GAM = 0.25
_C2, _C3, _C4 = 0.386, 0.21, 0.63
_D1, _D2, _D3, _D4 = 0.25, -0.1043, 0.1035, -0.3620000000000023e-01
_A21 = 0.1544e1
_A31, _A32 = 0.9466785280815826, 0.2557011698983284
_A41, _A42, _A43 = 0.3314825187068521e1, 0.2896124015972201e1, 0.9986419139977817
_A51, _A52, _A53, _A54 = 0.1221224509226641e1, 0.6019134481288629e1, 0.1253708332932087e2, -0.6878860361058950
_C21 = -0.56688e1
_C31, _C32 = -0.2430093356833875e1, -0.2063599157091915
_C41, _C42, _C43 = -0.1073529058151375, -0.9594562251023355e1, -0.2047028614809616e2
_C51, _C52, _C53, _C54 = 0.7496443313967647e1, -0.1024680431464352e2, -0.3399990352819905e2, 0.1170890893206160e2
_C61, _C62, _C63, _C64, _C65 = (0.8083246795921522e1, -0.7981132988064893e1, -0.3152159432874371e2,
                                0.1631930543123136e2, -0.6058818238834054e1)


@njit(cache=True)
def _hermite_extremum(x0, x1, d0, d1, want_max):
    """Extremum of the cubic Hermite interpolant on [0, 1]; d0, d1 are the end
    slopes already multiplied by the step length."""
    a = 2.0 * (x0 - x1) + d0 + d1
    b = 3.0 * (x1 - x0) - 2.0 * d0 - d1
    best = max(x0, x1) if want_max else min(x0, x1)
    if abs(a) < 1e-300:
        if abs(b) > 1e-300:
            s = -d0 / (2.0 * b)
            if 0.0 < s < 1.0:
                v = (b * s + d0) * s + x0
                best = max(best, v) if want_max else min(best, v)
        return best
    disc = b * b - 3.0 * a * d0
    if disc < 0.0:
        return best
    sq = math.sqrt(disc)
    for s in ((-b + sq) / (3.0 * a), (-b - sq) / (3.0 * a)):
        if 0.0 < s < 1.0:
            v = ((a * s + b) * s + d0) * s + x0
            best = max(best, v) if want_max else min(best, v)
    return best


@njit(cache=True)
def _integrate(P, n_cycles, rtol, atol_x, atol_u, hmax, max_steps, x_cap, rec):
    """Adaptive Rodas4 integration starting from rest at R = R0.

    rec: array (N, 3) receiving (tau, x, u) at accepted steps (N may be 0).
    Returns (status, n_steps, n_rec, tau_end, x_max, x_min, u_min, u_max);
    status 0 = ok, 1 = stopped at x_cap, 2 = max_steps reached, 3 = step underflow.
    """
    tau = 0.0
    x = 1.0
    u = 0.0
    h = 1e-5
    x_max = 1.0
    x_min = 1.0
    u_min = 0.0
    u_max = 0.0
    n_rec_max = rec.shape[0]
    n_rec = 0
    if n_rec_max > 0:
        rec[0, 0] = 0.0
        rec[0, 1] = 1.0
        rec[0, 2] = 0.0
        n_rec = 1
    status = 0
    n_steps = 0
    x_floor = P[9] / P[8]  # hard-core radius / R0
    err = 1.0
    xn = x
    un = u
    while tau < n_cycles * (1.0 - 1e-12):
        if n_steps >= max_steps:
            status = 2
            break
        if h > hmax:
            h = hmax
        if tau + h > n_cycles:
            h = n_cycles - tau
        a0, dadx, dadu, at = _accel_jac(tau, x, u, P)
        while True:
            if h < 1e-15 * max(1.0, tau):
                status = 3
                break
            gi = 1.0 / (_GAM * h)
            # M = gi*I - J with J = [[0, 1], [dadx, dadu]];  M^-1 r via Cramer
            m21 = -dadx
            m22 = gi - dadu
            det = gi * m22 + m21
            # stage 1
            r1 = u
            r2 = a0 + h * _D1 * at
            k1x = (m22 * r1 + r2) / det
            k1u = (-m21 * r1 + gi * r2) / det
            # stage 2
            xs = x + _A21 * k1x
            us = u + _A21 * k1u
            a = _accel(tau + _C2 * h, xs, us, P)
            r1 = us + _C21 * k1x / h
            r2 = a + h * _D2 * at + _C21 * k1u / h
            k2x = (m22 * r1 + r2) / det
            k2u = (-m21 * r1 + gi * r2) / det
            # stage 3
            xs = x + _A31 * k1x + _A32 * k2x
            us = u + _A31 * k1u + _A32 * k2u
            a = _accel(tau + _C3 * h, xs, us, P)
            r1 = us + (_C31 * k1x + _C32 * k2x) / h
            r2 = a + h * _D3 * at + (_C31 * k1u + _C32 * k2u) / h
            k3x = (m22 * r1 + r2) / det
            k3u = (-m21 * r1 + gi * r2) / det
            # stage 4
            xs = x + _A41 * k1x + _A42 * k2x + _A43 * k3x
            us = u + _A41 * k1u + _A42 * k2u + _A43 * k3u
            a = _accel(tau + _C4 * h, xs, us, P)
            r1 = us + (_C41 * k1x + _C42 * k2x + _C43 * k3x) / h
            r2 = a + h * _D4 * at + (_C41 * k1u + _C42 * k2u + _C43 * k3u) / h
            k4x = (m22 * r1 + r2) / det
            k4u = (-m21 * r1 + gi * r2) / det
            # stage 5 (at tau + h)
            xs = x + _A51 * k1x + _A52 * k2x + _A53 * k3x + _A54 * k4x
            us = u + _A51 * k1u + _A52 * k2u + _A53 * k3u + _A54 * k4u
            a = _accel(tau + h, xs, us, P)
            r1 = us + (_C51 * k1x + _C52 * k2x + _C53 * k3x + _C54 * k4x) / h
            r2 = a + (_C51 * k1u + _C52 * k2u + _C53 * k3u + _C54 * k4u) / h
            k5x = (m22 * r1 + r2) / det
            k5u = (-m21 * r1 + gi * r2) / det
            xs += k5x
            us += k5u
            # stage 6: embedded error
            a = _accel(tau + h, xs, us, P)
            r1 = us + (_C61 * k1x + _C62 * k2x + _C63 * k3x + _C64 * k4x + _C65 * k5x) / h
            r2 = a + (_C61 * k1u + _C62 * k2u + _C63 * k3u + _C64 * k4u + _C65 * k5u) / h
            ex = (m22 * r1 + r2) / det
            eu = (-m21 * r1 + gi * r2) / det
            xn = xs + ex
            un = us + eu
            sx = atol_x + rtol * max(abs(x), abs(xn))
            su = atol_u + rtol * max(abs(u), abs(un))
            err = max(abs(ex) / sx, abs(eu) / su)
            if not (err == err) or err > 1e300 or not (xn > x_floor):
                h *= 0.25
                continue
            if err <= 1.0:
                break
            h *= max(0.1, 0.9 * err ** (-0.25))
        if status == 3:
            break
        # accepted: local extrema from the cubic Hermite interpolant
        if u > 0.0 and un <= 0.0:
            v = _hermite_extremum(x, xn, u * h, un * h, True)
            if v > x_max:
                x_max = v
        elif u < 0.0 and un >= 0.0:
            v = _hermite_extremum(x, xn, u * h, un * h, False)
            if v < x_min:
                x_min = v
        tau += h
        x = xn
        u = un
        n_steps += 1
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if u < u_min:
            u_min = u
        if u > u_max:
            u_max = u
        if n_rec < n_rec_max:
            rec[n_rec, 0] = tau
            rec[n_rec, 1] = x
            rec[n_rec, 2] = u
            n_rec += 1
        if x_max >= x_cap:
            status = 1
            break
        if err > 1e-12:
            h *= min(5.0, max(0.2, 0.9 * err ** (-0.25)))
        else:
            h *= 5.0
    return status, n_steps, n_rec, tau, x_max, x_min, u_min, u_max


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    R0: float
    f: float
    Pa: float
    t: np.ndarray  # s
    R: np.ndarray  # m
    Rdot: np.ndarray  # m/s
    p_inf: np.ndarray  # Pa
    Rmax_ratio: float
    Rmin_ratio: float
    max_collapse_speed: float  # m/s (largest inward wall speed)
    mach: float  # max_collapse_speed / c
    kappa: float
    mu_th: float
    blake: float  # Pa
    status: int
    n_steps: int

    @property
    def regime(self) -> str:
        return classify(self.Rmax_ratio)

    def summary(self) -> str:
        return (
            f"R0 = {self.R0*1e6:.3g} um, f = {self.f/1e3:.4g} kHz, Pa = {self.Pa/1e3:.4g} kPa\n"
            f"  kappa = {self.kappa:.3f}, mu_th = {self.mu_th*1e3:.3g} mPa s\n"
            f"  R_max/R0 = {self.Rmax_ratio:.3f}, R_min/R0 = {self.Rmin_ratio:.3f}, "
            f"max collapse speed = {self.max_collapse_speed:.4g} m/s (Mach {self.mach:.3g})\n"
            f"  Blake threshold = {self.blake/1e3:.4g} kPa (Pa/P_Blake = {self.Pa/self.blake:.3g})\n"
            f"  regime: {self.regime}  [steps: {self.n_steps}, status {self.status}]"
        )


def classify(rmax_ratio: float) -> str:
    """Regime label from the expansion ratio (Flynn criterion at R_max/R0 = 2)."""
    if rmax_ratio >= INERTIAL_RATIO:
        return "damaging: inertial (transient) cavitation"
    if rmax_ratio >= 1.1:
        return "gentle: stable, nonlinear oscillation"
    return "gentle: stable, near-linear pulsation"


def simulate(
    R0: float,
    f: float,
    Pa: float,
    s: Settings = DEFAULT,
    cycles: float = 10.0,
    method: str = "rodas4",
    rtol: float = 1e-7,
    max_steps: int = 2_000_000,
) -> SimResult:
    """Integrate one bubble for `cycles` acoustic periods.

    method: "rodas4" (built-in Rosenbrock, fast) or any SciPy solve_ivp method
            ("Radau", "LSODA", "BDF") for cross-checking. The trajectory is
            recorded at every accepted step (up to 500 000 points)."""
    P = param_vector(R0, f, Pa, s)
    kappa, mu_th = (float(v) for v in thermal_parameters(R0, f, s))
    if method == "rodas4":
        rec = np.empty((min(max_steps + 1, 500_001), 3))
        st, n, nrec, _, xmax, xmin, umin, _ = _integrate(
            P, float(cycles), rtol, ATOL_X, ATOL_U, 0.02, max_steps, 1e9, rec
        )
        tau, x, u = rec[:nrec, 0], rec[:nrec, 1], rec[:nrec, 2]
    else:
        from scipy.integrate import solve_ivp

        def fun(tau_, y):
            return [y[1], _accel(tau_, y[0], y[1], P)]

        sol = solve_ivp(fun, (0.0, cycles), [1.0, 0.0], method=method, rtol=rtol,
                        atol=[ATOL_X * 1e-3, ATOL_U * 1e-3], max_step=0.02, dense_output=True)
        st, n = (0 if sol.success else 3), sol.t.size
        tau = np.union1d(sol.t, np.linspace(0.0, cycles, int(cycles * 4000) + 1))
        x, u = sol.sol(tau)
        xmax, xmin, umin = x.max(), x.min(), u.min()
    t = tau / f
    p_inf = s.p0 - Pa * np.sin(TWO_PI * tau)
    speed = -umin * R0 * f
    return SimResult(
        R0, f, Pa, t, x * R0, u * R0 * f, p_inf, float(xmax), float(xmin), float(speed),
        float(speed / s.liquid.c), kappa, mu_th, float(blake_threshold(R0, s)), int(st), int(n),
    )


@njit(parallel=True, cache=True)
def _map_kernel(Ps, n_cycles, rtol, max_steps, x_cap, out):
    n = Ps.shape[0]
    empty = np.empty((0, 3))
    for k in prange(n):
        res = _integrate(Ps[k], n_cycles, rtol, ATOL_X, ATOL_U, 0.02, max_steps, x_cap, empty)
        out[k, 0] = res[4]  # R_max/R0
        out[k, 1] = res[5]  # R_min/R0
        out[k, 2] = -res[6] * Ps[k, 8] * Ps[k, 11]  # max inward wall speed (m/s)
        out[k, 3] = res[0]  # status


def _batch(R0s, fs, Pas, s, cycles, rtol, max_steps, x_cap):
    R0s, fs, Pas = np.broadcast_arrays(np.asarray(R0s, float), np.asarray(fs, float), np.asarray(Pas, float))
    shape = R0s.shape
    R0f, ff, Paf = R0s.ravel(), fs.ravel(), Pas.ravel()
    kappa, mu_th = thermal_parameters(R0f, ff, s)
    Ps = np.empty((R0f.size, 13))
    liq = s.liquid
    Ps[:, I_RHO], Ps[:, I_MU], Ps[:, I_SIG], Ps[:, I_C] = liq.rho, liq.mu + mu_th, liq.sigma, liq.c
    Ps[:, I_PV], Ps[:, I_P0], Ps[:, I_PG0], Ps[:, I_KAP] = liq.pv, s.p0, gas_pressure_eq(R0f, s), kappa
    Ps[:, I_R0], Ps[:, I_H], Ps[:, I_PA], Ps[:, I_F] = R0f, R0f / s.gas.hard_core_ratio, Paf, ff
    Ps[:, I_KM] = 1.0 if s.model.upper() == "KM" else 0.0
    out = np.empty((R0f.size, 4))
    _map_kernel(Ps, float(cycles), rtol, max_steps, x_cap, out)
    return {
        "Rmax_ratio": out[:, 0].reshape(shape),
        "Rmin_ratio": out[:, 1].reshape(shape),
        "collapse_speed": out[:, 2].reshape(shape),
        "status": out[:, 3].reshape(shape).astype(int),
    }


def size_pressure_map(R0s, Pas, f, s: Settings = DEFAULT, cycles=10.0, rtol=1e-5,
                      max_steps=400_000, x_cap=50.0):
    """R_max/R0 etc. on a grid of equilibrium radii (columns) x amplitudes (rows)
    at a fixed frequency. Returns dict of 2-D arrays shaped (len(Pas), len(R0s))."""
    R0g, Pag = np.meshgrid(np.asarray(R0s, float), np.asarray(Pas, float))
    return _batch(R0g, f, Pag, s, cycles, rtol, max_steps, x_cap)


def frequency_pressure_map(fs, Pas, R0, s: Settings = DEFAULT, cycles=10.0, rtol=1e-5,
                           max_steps=400_000, x_cap=50.0):
    """Same as size_pressure_map but over frequency (columns) at a fixed R0."""
    fg, Pag = np.meshgrid(np.asarray(fs, float), np.asarray(Pas, float))
    return _batch(R0, fg, Pag, s, cycles, rtol, max_steps, x_cap)


def inertial_threshold(R0: float, f: float, s: Settings = DEFAULT, cycles=10.0,
                       Pa_lo=1e3, Pa_hi=5e7, n_scan=48, rel_tol=2e-3, rtol=1e-6) -> float:
    """Smallest acoustic amplitude for which R_max/R0 >= 2 within `cycles`
    (scan on a log grid, then bisection). Returns nan if not reached."""
    grid = np.geomspace(Pa_lo, Pa_hi, n_scan)
    r = _batch(R0, f, grid, s, cycles, rtol, 400_000, 2.0 * INERTIAL_RATIO)["Rmax_ratio"]
    idx = np.nonzero(r >= INERTIAL_RATIO)[0]
    if idx.size == 0:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(grid[0])
    lo, hi = grid[i - 1], grid[i]
    while hi / lo - 1.0 > rel_tol:
        mid = math.sqrt(lo * hi)
        rm = _batch(R0, f, mid, s, cycles, rtol, 400_000, 2.0 * INERTIAL_RATIO)["Rmax_ratio"]
        if float(rm) >= INERTIAL_RATIO:
            hi = mid
        else:
            lo = mid
    return float(math.sqrt(lo * hi))


# ---------------------------------------------------------------------------
# Figure (matplotlib) — same encoding as the web app
# ---------------------------------------------------------------------------
#: R_max/R0 bin edges and colours: blue arm = stable/gentle, red arm = inertial/damaging.
REGIME_EDGES = [1.0, 1.01, 1.03, 1.1, 1.3, 2.0, 3.0, 5.0, 10.0, 20.0, np.inf]
REGIME_COLORS_LIGHT = ["#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6",
                       "#fac0ba", "#f3958d", "#e86761", "#d33739", "#a82428"]
_INK = {"surface": "#fcfcfb", "page": "#f9f9f7", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}


def _style_axes(ax, plt):
    ax.set_facecolor(_INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_INK["muted"], labelcolor=_INK["secondary"], labelsize=8.5, width=0.8)
    ax.xaxis.label.set_color(_INK["secondary"])
    ax.yaxis.label.set_color(_INK["secondary"])
    ax.title.set_color(_INK["primary"])


def _fmt_pressure(v, _pos=None):
    return f"{v/1e6:g} MPa" if v >= 1e6 else f"{v/1e3:g} kPa"


def _fmt_freq(v, _pos=None):
    return f"{v/1e6:.3g} MHz" if v >= 1e6 else f"{v/1e3:.3g} kHz"


def _fmt_radius(v, _pos=None):
    return f"{v*1e3:g} mm" if v >= 1e-3 else f"{v*1e6:g} µm"


def _heat(ax, X, Y, Z, plt):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    def edges(v):
        lv = np.log(v)
        mid = 0.5 * (lv[1:] + lv[:-1])
        return np.exp(np.concatenate([[lv[0] - (mid[0] - lv[0])], mid, [lv[-1] + (lv[-1] - mid[-1])]]))

    cmap = ListedColormap(REGIME_COLORS_LIGHT)
    norm = BoundaryNorm(REGIME_EDGES[:-1] + [1e9], cmap.N)
    ax.pcolormesh(edges(X), edges(Y), np.clip(Z, 1.0, 1e8), cmap=cmap, norm=norm, shading="flat",
                  rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")


def _halo_line(ax, x, y, style, label=None, lw=1.6):
    import matplotlib.patheffects as pe

    ln, = ax.plot(x, y, color=_INK["primary"], lw=lw, ls=style, label=label, solid_capstyle="round",
                  dash_capstyle="round")
    ln.set_path_effects([pe.Stroke(linewidth=lw + 2.6, foreground=_INK["surface"]), pe.Normal()])
    return ln


def plot_overview(path="bubble_regime_maps.png", freqs=(40e3, 1e6), R0_fmap=1e-6,
                  s: Settings = DEFAULT, cycles=10.0, n_r=72, n_p=60, n_f=60, points=None, dpi=160):
    """Four-panel figure: two size–pressure regime maps, a frequency–pressure map
    and example radius–time curves. Returns the computed data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    R0s = np.geomspace(0.1e-6, 1e-3, n_r)
    Pas = np.geomspace(1e4, 1e7, n_p)
    fs = np.geomspace(1e4, 5e6, n_f)
    maps = [size_pressure_map(R0s, Pas, f, s, cycles) for f in freqs]
    fmap = frequency_pressure_map(fs, Pas, R0_fmap, s, cycles)
    if points is None:
        points = [(R0_fmap, freqs[-1], 100e3, "A"), (R0_fmap, freqs[-1], 300e3, "B")]
    sims = [simulate(R0, f, Pa, s, cycles=min(cycles, 6.0)) for R0, f, Pa, _ in points]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(12.5, 10.6), facecolor=_INK["page"])
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.09], hspace=0.42, wspace=0.2,
                          left=0.07, right=0.98, top=0.91, bottom=0.07)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
           fig.add_subplot(gs[1, 1])]
    fig.suptitle("Where acoustic cavitation stays gentle and where it turns damaging", x=0.07, ha="left",
                 fontsize=15, fontweight="bold", color=_INK["primary"], y=0.975)
    liq = s.liquid
    fig.text(0.07, 0.938, f"{'Keller–Miksis' if s.model.upper()=='KM' else 'Rayleigh–Plesset'} bubble, "
             f"air in {liq.name.lower()}, p0 = {s.p0/1e5:.3g} bar, {cycles:g} acoustic cycles from rest; "
             "colour = largest expansion R_max/R0 (inertial when ≥ 2)", fontsize=9.5, color=_INK["secondary"])

    blake = blake_threshold(R0s, s)
    for k, (ax, f, m) in enumerate(zip(axs[:2], freqs, maps)):
        _heat(ax, R0s, Pas, m["Rmax_ratio"], plt)
        ax.contour(R0s, Pas, np.log(np.clip(m["Rmax_ratio"], 1, 1e6)), levels=[math.log(INERTIAL_RATIO)],
                   colors=[_INK["primary"]], linewidths=1.7)
        _halo_line(ax, R0s, blake, (0, (4, 2.5)))
        rres = resonance_radius(f, s)
        ax.axvline(rres, color=_INK["primary"], lw=1.2, ls=(0, (1, 2.2)))
        ax.text(rres * 1.12, Pas[-1] * 0.62, f"resonant size\n{rres*1e6:.3g} µm", fontsize=8,
                color=_INK["primary"], va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc=_INK["surface"], ec="none", alpha=0.85))
        kr_lim = 0.3 * liq.c / (2 * np.pi * f)
        if kr_lim < R0s[-1]:
            ax.axvspan(kr_lim, R0s[-1], color=_INK["surface"], alpha=0.55, lw=0)
            ax.text(math.sqrt(kr_lim * R0s[-1]), Pas[1] * 1.5, "kR₀ > 0.3\nmodel\nnot valid",
                    ha="center", fontsize=7.5, color=_INK["secondary"])
        ax.set_xlim(R0s[0], R0s[-1])
        ax.set_ylim(Pas[0], Pas[-1])
        ax.set_title(f"{'ab'[k]}   Bubble size vs pressure at {_fmt_freq(f)}", loc="left", fontsize=11,
                     fontweight="bold", pad=8)
        ax.set_xlabel("equilibrium (nucleus) radius R₀")
        ax.set_ylabel("acoustic pressure amplitude Pa")
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_radius))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pressure))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        _style_axes(ax, plt)
    # mark the example points on the map whose frequency they use
    for (R0, f, Pa, lab) in points:
        k = int(np.argmin([abs(math.log(f / ff)) for ff in freqs]))
        ax = axs[k]
        ax.plot([R0], [Pa], marker="o", ms=7, mfc=_INK["primary"], mec=_INK["surface"], mew=2, zorder=5)
        ax.annotate(lab, (R0, Pa), xytext=(-12, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=9, fontweight="bold", color=_INK["primary"],
                    bbox=dict(boxstyle="round,pad=0.2", fc=_INK["surface"], ec="none", alpha=0.9))
    # labels for Blake on panel a
    ax = axs[0]
    j = int(np.searchsorted(R0s, 0.25e-6))
    ax.annotate("Blake threshold (static)", (R0s[j], blake[j]), xytext=(18, 30), textcoords="offset points",
                fontsize=8, color=_INK["primary"], arrowprops=dict(arrowstyle="-", color=_INK["primary"], lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc=_INK["surface"], ec="none", alpha=0.9))

    # c: frequency–pressure map
    ax = axs[2]
    _heat(ax, fs, Pas, fmap["Rmax_ratio"], plt)
    ax.contour(fs, Pas, np.log(np.clip(fmap["Rmax_ratio"], 1, 1e6)), levels=[math.log(INERTIAL_RATIO)],
               colors=[_INK["primary"]], linewidths=1.7)
    pb = float(blake_threshold(R0_fmap, s))
    _halo_line(ax, fs, np.full_like(fs, pb), (0, (4, 2.5)))
    f0 = float(natural_frequency(R0_fmap, s)[0])
    if fs[0] < f0 < fs[-1]:
        ax.axvline(f0, color=_INK["primary"], lw=1.2, ls=(0, (1, 2.2)))
        ax.text(f0 / 1.12, Pas[-1] * 0.62, f"resonance of\nthis bubble\n{_fmt_freq(f0)}", fontsize=8,
                ha="right", va="top", color=_INK["primary"],
                bbox=dict(boxstyle="round,pad=0.25", fc=_INK["surface"], ec="none", alpha=0.85))
    ax.text(fs[1], pb * 1.12, f"Blake threshold {pb/1e3:.0f} kPa", fontsize=8, color=_INK["primary"],
            bbox=dict(boxstyle="round,pad=0.2", fc=_INK["surface"], ec="none", alpha=0.85))
    ax.set_xlim(fs[0], fs[-1])
    ax.set_ylim(Pas[0], Pas[-1])
    ax.set_title(f"c   Frequency vs pressure for an R₀ = {_fmt_radius(R0_fmap)} nucleus", loc="left",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("driving frequency f")
    ax.set_ylabel("acoustic pressure amplitude Pa")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_freq))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pressure))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style_axes(ax, plt)

    # d: radius-time curves
    ax = axs[3]
    series_colors = ["#2a78d6", "#d33739"]
    for (pt, sim, col) in zip(points, sims, series_colors):
        tc = sim.t * sim.f
        x = sim.R / sim.R0
        lab = f"{pt[3]}: Pa = {_fmt_pressure(sim.Pa)} → R_max/R₀ = {sim.Rmax_ratio:.2f}, {sim.regime.split(':')[0]}"
        ax.plot(tc, x, color=col, lw=1.8, label=lab, solid_joinstyle="round")
        first = tc < 1.0
        i = int(np.argmax(np.where(first, x, -1)))
        ax.annotate(pt[3], (tc[i], x[i]), xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9, fontweight="bold", color=_INK["primary"])
    ax.axhline(INERTIAL_RATIO, color=_INK["primary"], lw=1.0, ls=(0, (4, 2.5)))
    xr = max(s_.t[-1] * s_.f for s_ in sims)
    ax.text(xr, INERTIAL_RATIO, "R = 2R₀", fontsize=8, color=_INK["primary"], va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc=_INK["surface"], ec="none"))
    ax.grid(True, axis="y", color=_INK["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xr)
    ax.set_ylim(0, max(s_.Rmax_ratio for s_ in sims) * 1.32)
    ax.set_title(f"d   Bubble radius over time ({_fmt_radius(points[0][0])}, {_fmt_freq(points[0][1])})",
                 loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("time (acoustic cycles)")
    ax.set_ylabel("R / R₀")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=_INK["secondary"])
    _style_axes(ax, plt)

    # stepped legend (colour bins) + line keys
    lax = fig.add_subplot(gs[2, :])
    lax.set_axis_off()
    n = len(REGIME_COLORS_LIGHT)
    x0, w, gap, y0, hgt = 0.0, 0.052, 0.004, 0.25, 0.5
    for i, col in enumerate(REGIME_COLORS_LIGHT):
        lax.add_patch(plt.Rectangle((x0 + i * (w + gap), y0), w, hgt, color=col, transform=lax.transAxes,
                                    clip_on=False))
    for i in range(1, n):
        lax.text(x0 + i * (w + gap) - gap / 2, y0 - 0.12, ["1.01", "1.03", "1.1", "1.3", "2", "3", "5", "10", "20"][i - 1],
                 ha="center", va="top", fontsize=8, color=_INK["secondary"], transform=lax.transAxes)
    mid = x0 + 5 * (w + gap) - gap / 2
    lax.plot([mid, mid], [y0 - 0.05, y0 + hgt + 0.15], color=_INK["primary"], lw=1.7, transform=lax.transAxes,
             clip_on=False)
    lax.text(x0, y0 + hgt + 0.18, "← gentle: stable cavitation", fontsize=9, color=_INK["primary"],
             transform=lax.transAxes, va="bottom")
    lax.text(x0 + n * (w + gap) - gap, y0 + hgt + 0.18, "damaging: inertial cavitation →", fontsize=9,
             color=_INK["primary"], transform=lax.transAxes, va="bottom", ha="right")
    lax.text(x0 + n * (w + gap) / 2 - gap, y0 - 0.62, "largest expansion R_max / R₀", fontsize=8.5,
             color=_INK["secondary"], transform=lax.transAxes, ha="center", va="top")
    handles = [Line2D([], [], color=_INK["primary"], lw=1.7, label="inertial threshold (simulated, R_max = 2R₀)"),
               Line2D([], [], color=_INK["primary"], lw=1.6, ls=(0, (4, 2.5)), label="Blake threshold (static)"),
               Line2D([], [], color=_INK["primary"], lw=1.2, ls=(0, (1, 2.2)), label="linear resonance")]
    lax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.6, 0.5), frameon=False, fontsize=8.5,
               labelcolor=_INK["secondary"], ncol=1, handlelength=3)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"R0s": R0s, "Pas": Pas, "fs": fs, "maps": maps, "fmap": fmap, "sims": sims}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _main(argv=None):  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Acoustic bubble dynamics & cavitation regime maps")
    ap.add_argument("--simulate", nargs=3, type=float, metavar=("R0_um", "f_kHz", "Pa_kPa"),
                    help="run one bubble and print a summary")
    ap.add_argument("--figure", default="bubble_regime_maps.png", help="output PNG for the overview figure")
    ap.add_argument("--freqs", nargs="+", type=float, default=[40e3, 1e6], help="two map frequencies, Hz")
    ap.add_argument("--r0", type=float, default=1.0, help="nucleus radius for the frequency map, µm")
    ap.add_argument("--liquid", choices=sorted(LIQUIDS), default="water20")
    ap.add_argument("--model", choices=["KM", "RP"], default="KM")
    ap.add_argument("--thermal", choices=["prosperetti", "adiabatic", "isothermal"], default="prosperetti")
    ap.add_argument("--p0", type=float, default=P_ATM, help="static pressure, Pa")
    ap.add_argument("--cycles", type=float, default=10.0)
    ap.add_argument("--coarse", action="store_true", help="coarser grids (faster, e.g. without numba)")
    a = ap.parse_args(argv)
    s = Settings(model=a.model, thermal=a.thermal, p0=a.p0, liquid=LIQUIDS[a.liquid])
    if a.simulate:
        R0, f, Pa = a.simulate[0] * 1e-6, a.simulate[1] * 1e3, a.simulate[2] * 1e3
        r = simulate(R0, f, Pa, s, cycles=a.cycles)
        print(r.summary())
        print(f"  resonant radius at this frequency: {resonance_radius(f, s)*1e6:.3g} um; "
              f"natural frequency of this bubble: {natural_frequency(R0, s)[0]/1e3:.4g} kHz")
        return
    if not HAVE_NUMBA:
        print("numba not installed: maps run in pure Python and will be slow (try --coarse, or pip install numba)")
    n = (28, 24, 24) if a.coarse else (72, 60, 60)
    plot_overview(a.figure, tuple(a.freqs[:2]), a.r0 * 1e-6, s, a.cycles, *n)
    print(f"wrote {a.figure}")
    for f in a.freqs[:2]:
        print(f"  f = {f/1e3:g} kHz: resonant radius {resonance_radius(f, s)*1e6:.3g} um")
    for R0 in (0.5e-6, 1e-6, 2e-6, 5e-6):
        print(f"  R0 = {R0*1e6:g} um: Blake threshold {float(blake_threshold(R0, s))/1e3:.1f} kPa; "
              + ", ".join(f"inertial at {inertial_threshold(R0, f, s, a.cycles)/1e3:.0f} kPa ({f/1e3:g} kHz)"
                          for f in a.freqs[:2]))


if __name__ == "__main__":  # pragma: no cover
    _main()
