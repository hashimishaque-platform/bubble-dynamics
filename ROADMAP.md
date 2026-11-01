# Roadmap

Each step is developed on its own branch and merged into `main` once `node checks.js` passes.
A step is ticked here with the date it was merged.

## Thermal model in the page
- [x] Port the effective polytropic exponent and thermal damping (Prosperetti) from `python/` to `bubble.js` as a third gas option, with a check against the Python values at three sizes and frequencies. (2026-10-17)
  The thermal diffusivity of air is now 2.074e-5 m²/s at 20 °C (Çengel and Cimbala, Table A-9) in both models; the Python model used 2.15e-5 without a source.
- [x] Use it in the threshold figure and describe it under Model. (2026-10-20)
  Figure 2 now draws a second line for the other gas model, so the effect of heat exchange is visible.

## Liquids
- [x] Presets from the Python model: water at 40 °C and 60 °C, water with surfactant, and 2-propanol. (2026-11-01)
  Water at 20, 40 and 60 °C from the IAPWS formulations, and 2-propanol at 25 °C from published measurements (20 °C values with a source were not found).
  Water with surfactant is left out: the 35 mN/m in the Python model has no published source, and the value depends on the surfactant and its concentration.

## Regime maps
- [ ] Frequency against pressure map for one bubble size, computed in small steps so the page stays responsive.
- [ ] Bubble size against pressure map at the chosen frequency.

## Rectified diffusion
- [ ] Threshold for rectified diffusion against bubble size (Eller and Flynn 1965, in the form given by Crum 1984), with a check against a published value.
- [ ] Show it in the threshold figure.

## Bjerknes force
- [ ] Primary Bjerknes force in a standing wave from the computed radius, with a check that its sign changes at resonance.

## Python model
- [ ] Tests for the Python model with pytest: natural frequency, Blake threshold and Rayleigh collapse time.
- [ ] A script comparing the Python model and the page at the same settings.

## Export
- [ ] Download the radius-time series as a CSV file.
