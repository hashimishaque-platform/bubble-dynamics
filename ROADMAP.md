# Roadmap

Each step is developed on its own branch and merged into `main` once `node checks.js` passes.
A step is ticked here with the date it was merged.

## Thermal model in the page
- [ ] Port the effective polytropic exponent and thermal damping (Prosperetti) from `python/` to `bubble.js` as a third gas option, with a check against the Python values at three sizes and frequencies.
- [ ] Use it in the threshold figure and describe it under Model.

## Liquids
- [ ] Presets from the Python model: water at 40 °C and 60 °C, water with surfactant, and 2-propanol.

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
