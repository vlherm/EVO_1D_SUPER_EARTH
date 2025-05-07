# EVO_1D_SUPER_EARTH

This repository contains the source code required to reproduce the results presented *Magnetic Evolution of Super-Earth Exoplanets with a Basal Magma Ocean* by Lherm et al. (in prep.). The code is adapted from the model developed in [Lherm et al. (2024)](https://doi.org/10.1016/j.pepi.2024.107267), which was originally designed to model the evolution of an Earth-like planet.

This repository includes the following files:
- `structure.py`: Computes the internal structure of a planet with a given mass and core mass fraction, at multiple thermal states.
- `thermal.py`: Computes the thermal evolution of a planet using pre-computed internal structures from `structure.py`.
- `functions.py`: Contains supporting functions required to compute the thermal evolution.
- `buoyancy.py`: Computes the buoyancy evolution of a planet based on a thermal evolution obtained from `thermal.py`.
- `magnetic.py`: Computes the magnetic evolution of a planet based on a thermal evolution obtained from `thermal.py`.
- `sweep.py`: Computes a post-processed data structure across a set of simulations with varying parameters, by combining outputs from multiple structural, thermal, buoyancy, and magnetic evolution models.

`data_C_2006.dat` and `data_A_2009.dat` contain data from [Christensen & Aubert (2006)](https://doi.org/10.1111/j.1365-246X.2006.03009.x) and [Aubert et al. (2009)](https://doi.org/10.1111/j.1365-246X.2009.04361.x), required for computing magnetic scaling laws in `magnetic.py`.

### References
- Aubert, J., Labrosse, S. & Poitou, C. Modelling the palaeo-evolution of the geodynamo. *Geophysical Journal International*, 179(3):1414–1428, 2009.
- Christensen, U. R. & Aubert, J. Scaling properties of convection-driven dynamos in rotating spherical shells and application to planetary magnetic fields. *Geophysical Journal International*, 166(1):97–114, 2006.
- Lherm, V., Nakajima, M. & Blackman, E. G. Thermal and magnetic evolution of an Earth-like planet with a basal magma ocean. *Physics of the Earth and Planetary Interiors*, 107267, 2024.
