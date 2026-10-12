# Satellite Relative Motion Dynamics

Individual project implementing and benchmarking different relative motion dynamic models for LEO formation proximity operations scenarios.

## Overview

This repository contains an ongoing project focused on relative motion dynamic models simulation for LEO proximity scenario activities.
The objective is to implement and assess the performance of different relative dynamic models, with focus towards guidance and navigation applications.
This work is currently under active development.

## Methodology

The project follows the given methodology:
- Relative motion computation through target and chaser absolute Keplerian motion propagation
- Relative motion computation through target and chaser absolute two-body-problem motion propagation, including J2 and drag perturbations
- Relative motion propagation using Clohessy-Wiltshire relative dynamics
- Relative motion propagation using Tschauner-Hempel relative dynamics
- Relative motion propagation using linear time-varying high-fidelity model described in 2015 Okasha et. al. "Modeling, Dynamics and Control of Spacecraft Relative Motion in a Perturbed Keplerian Orbit"

## Current Status

This project is a work in progress.
At the current stage:
- Core Keplerian and two-body-problem dynamics with J2 perturbations are implemented
- Clohessy-Wiltshire relative dynamics are implemented
- Tschauner-Hempel relative dynamics are implemented
- For circular target orbits (e=0) CW and TH produce equivalent relative motion, consistent with TH reducing to CW at circular limit
- For eccentric target cases (e=!0) TH remains consistent with non-linear Keplerian and 2BP+J2 models, while expectedly CW diverges

Planned next steps include:
- Addition of atmospheric drag disturbance to 2BP model
- Validation of CW implementation against closed form solution
- Validation of TH implementation
- Addition of high-fidelity LTV model

## Preliminaty Outputs

Representative intermediate outputs include:
- Relative motion in target LVLH frame
- Relative distance history

See 'results/results.txt' for figure-by-figure explanations.

## Repository Structure

- 'src/' - Python implementation
- 'results/' - Key result figures (PNG)
- 'figures/' - Source figures (EPS)

## Reproducibility and external dependencies

This project relies on external SPICE kernels for ephemerides and constants, not included in this repository.
To run the simulations, the following NAIF kernels are required:
- 'gm_de440.tpc'
- 'pck00010.tpc'
