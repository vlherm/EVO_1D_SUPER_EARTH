# EVO_1D_SUPER_EARTH

## Overview

This repository contains the source code required to reproduce the results presented in *Magnetic Evolution of Super-Earth Exoplanets with a Basal Magma Ocean* by Lherm et al. (in prep.). The code is adapted from the model developed in [Lherm et al. (2024)](https://doi.org/10.1016/j.pepi.2024.107267), which was originally designed to model the evolution of an Earth-like planet.

## Content

This repository includes the following files:
- [`structure.py`](structure.py): Computes the internal structure of a planet with a given mass and core mass fraction, at multiple thermal states.
- [`thermal.py`](thermal.py): Computes the thermal evolution of a planet using pre-computed internal structures.
- [`functions.py`](functions.py): Contains supporting functions required to compute the thermal evolution.
- [`buoyancy.py`](buoyancy.py): Computes the buoyancy evolution of a planet based on a thermal evolution.
- [`magnetic.py`](magnetic.py): Computes the magnetic evolution of a planet based on a thermal evolution.
- [`sweep.py`](sweep.py): Computes a post-processed data structure across a set of simulations with varying parameters (e.g. mass, core mass fraction, etc.), by combining outputs from multiple structural, thermal, buoyancy, and magnetic evolution models.
- [`figures.py`](figures.py): Computes the figures presented in Lherm et al. (in prep.) using post-processed data structures in the [`Data`](Data/) folder.

This repository also includes the following data files:
- [`sw_M.pkl`](Data/sw_M.dat): Contains a post-processed data structure across a set of simulations with varying planet mass.
- [`sw_CMF.pkl`](Data/sw_CMF.dat): Contains a post-processed data structure across a set of simulations with varying core mass fraction.
- [`sw_n_mu`](Data/sw_n_mu.dat): Contains a post-processed data structure across a set of simulations with varying mass power-law exponent of the mantle viscosity.
- [`sw_m_mu.pkl`](Data/sw_m_mu.dat): Contains a post-processed data structure across a set of simulations with varying core mass fraction exponent of the mantle viscosity.
- [`data_C_2006.pkl`](Data/data_C_2006.dat) contains data from [Christensen & Aubert (2006)](https://doi.org/10.1111/j.1365-246X.2006.03009.x), required for computing magnetic scaling laws in [`magnetic.py`](magnetic.py).
- [`data_A_2009.pkl`](Data/data_A_2009.dat) contains data from [Aubert et al. (2009)](https://doi.org/10.1111/j.1365-246X.2009.04361.x), required for computing magnetic scaling laws in [`magnetic.py`](magnetic.py).

## System Requirements

### Hardware Requirements

The code can be run on a standard machine or on a cluster. Using a cluster is recommended when running a large number of simulations, as it allows for parallel execution and reduces the overall computation time.

The code has been tested on a local machine with 24 CPUs and 64 GB of RAM, and on a cluster with up to 256 CPUs and 512 GB of RAM. 

### Software Requirements

The code requires Python 3 with publicly available packages. Parallel computing is implemented using the `mpi4py` package and OpenMPI.

The code has been tested on a local machine on Linux Ubuntu 22.04 with Python 3.10.12 and OpenMPI 4.1.2, and on a cluster with Python 3.10.12 and OpenMPI 4.0.4. 

## Installation

The code can be installed instantaneously by cloning this repository:

```bash
git clone https://github.com/vlherm/EVO_1D_SUPER_EARTH
cd EVO_1D_SUPER_EARTH
```

## Instructions

For each planetary evolution simulation, the code requires the following steps:

1. **Compute the internal structure** of the planet using `structure.py`. This step returns a .pkl file for each CMB temperature (T_CMB_XXXXX.pkl) within the computed range. Each file contains a class with attributes describing the internal structure (e.g. density, pressure, temperature, and other thermodynamic properties) as a function of radius. Runtime is approximately 54 minutes with 24 CPUs.

```bash
mpiexec -n 24 python3 -m mpi4py.futures structure.py
```

2. **Compute the thermal evolution** of the planet using `thermal.py`. This step requires the pre-computed internal structure files from the previous step, and returns a .pkl file (thermal.pkl) containing a class with attributes describing the thermal evolution of the planet (e.g. temperatures, heat flows, etc.). Runtime is approximately 28 minutes per simulation, but depends on the solver tolerance and minimum time step.

```bash
python3 thermal.py
```

3. **Compute the buoyancy evolution** of the planet using `buoyancy.py`. This step requires the thermal evolution file from the previous step, and returns a .pkl file (buoyancy.pkl) containing a class with attributes describing the local convective stability evolution of the planet (mass anomaly fluxes). Runtime is approximately 2 minutes per simulation. Multiple simulations can be run in parallel to speed up the computation.

```bash
mpiexec -n 1 python3 -m mpi4py.futures buoyancy.py
```

4. **Compute the magnetic evolution** of the planet using `magnetic.py`. This step requires the thermal evolution file from the previous step, and returns .pkl files describing the entropy budget and magnetic field evolution of the planet. The entropy budget files contain a class with the entropy budget terms (e.g. secular cooling, radiogenic heating, latent heat, etc.). The magnetic field files contain a class with magnetic quantities (e.g. magnetic Reynolds number, magnetic field intensities, etc.). Runtime is approximately 15 minutes per simulation. Multiple simulations can be run in parallel to speed up the computation.

```bash
mpiexec -n 1 python3 -m mpi4py.futures magnetic.py
```

Any set of simulations can be post-processed using `sweep.py`, allowing to combine outputs from multiple structural, thermal, buoyancy, and magnetic evolution models with varying parameters (e.g. mass, CMF, etc.), and facilitate data visualization. This script returns a .pkl file (sweep.pkl) containing a class with attributes describing planet evolution (e.g. radii, temperatures, heat flows, mass anomaly fluxes, rates of entropy production, magnetic Reynolds numbers, magnetic field intensities, etc.) as a function of the sweep parameter. 

## Demonstration

The files `structure.py`, `functions.py`, `thermal.py`, `buoyancy.py`, and `magnetic.py` in this repository are configured to run a demonstration simulation for an Earth-like planet. The demonstration simulation can be run by commands provided above. The output files can be used to visualize the evolution of the planet.

## References
- Aubert, J., Labrosse, S. & Poitou, C. Modelling the palaeo-evolution of the geodynamo. *Geophysical Journal International*, 179(3):1414–1428, 2009.
- Christensen, U. R. & Aubert, J. Scaling properties of convection-driven dynamos in rotating spherical shells and application to planetary magnetic fields. *Geophysical Journal International*, 166(1):97–114, 2006.
- Lherm, V., Nakajima, M. & Blackman, E. G. Thermal and magnetic evolution of an Earth-like planet with a basal magma ocean. *Physics of the Earth and Planetary Interiors*, 107267, 2024.
