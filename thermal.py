#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Victor Lherm

All rights reserved.

This code is provided for academic and research purposes only. Any use of this code in published work or presentations must include a citation to the original author. Unauthorized copying, modification, or distribution of this code is prohibited without prior written permission from the author.

Computes the thermal evolution of a planet using pre-computed internal structures (T_CMB_XXXXX.pkl files) from structure.py.

Returns a .pkl file (thermal.pkl) containing a class with attributes describing the thermal evolution of the planet (e.g. temperatures, heat flows, etc.).

Created by Victor Lherm on 2025-05-07 10:00:00.
"""

# %% Modules
# =============================================================================
import numpy as np
import dill as pkl

from scipy import integrate
from scipy.constants import G, mu_0, R, k, N_A, eV

from functions import *

# %% Initial conditions
# =============================================================================
S0, SE0, T_CMB_initial = find_T_CMB_initial(ini_type=ini_type_T_CMB)

# %% Solver
# =============================================================================
t_span = [0, y2s(t_final)]
t_eval = np.linspace(t_span[0], t_span[1], int(N))

Y0 = [T_CMB_initial, Xi_BMO_initial, Xi_core_initial]
sol_BMO = integrate.solve_ivp(
    Q_budget,
    t_span,
    Y0,
    t_eval=t_eval,
    dense_output=True,
    method="RK45",
    rtol=1e-3,
    atol=1e-6,
    max_step=y2s(1e7),
    args=(SE0, Xi_BMO_initial),
)

t = sol_BMO.t
T_CMB = sol_BMO.y[0]
Xi_BMO = sol_BMO.y[1]
Xi_core = sol_BMO.y[2]
dT_CMBdt = np.gradient(T_CMB, t)
dXi_BMOdt = np.gradient(Xi_BMO, t)
dXi_coredt = np.gradient(Xi_core, t)

# %% Structure post-processing
# =============================================================================
S, SE = structure_pp(t, T_CMB, Xi_BMO, SE0, Xi_BMO_initial)

# %% Energy budget post-processing
# =============================================================================
T_BMO, T_ICB, T_center = (np.zeros(len(t)) for i in range(3))
for i in range(len(t)):
    T_BMO[i] = S[i].Ta_BMO_0
    T_ICB[i] = S[i].T_ICB
    T_center[i] = S[i].T_center

R_planet, R_BMO, R_CMB, R_ICB = (np.zeros(len(t)) for i in range(4))
for i in range(len(t)):
    R_planet[i] = S[i].R_planet
    R_BMO[i] = S[i].R_BMO
    R_CMB[i] = S[i].R_CMB
    R_ICB[i] = S[i].R_ICB

Q_S_mantle, Q_R_mantle = (np.zeros(len(t)) for i in range(2))
Q_S_BMO, Q_L_BMO, Q_G_BMO, Q_C_BMO, Q_R_BMO = (np.zeros(len(t)) for i in range(5))
Q_S_core, Q_L_core, Q_G_core, Q_C_core, Q_R_core = (np.zeros(len(t)) for i in range(5))
Q_S_IC, Q_R_IC = (np.zeros(len(t)) for i in range(2))
Q_planet, Q_BMO, Q_CMB, Q_ICB = (np.zeros(len(t)) for i in range(4))
Q_A_BMO, Q_A_core = (np.zeros(len(t)) for i in range(2))
for i in range(len(t)):
    Q_S_mantle[i] = P_S_mantle(S[i]) * dT_CMBdt[i]
    Q_R_mantle[i] = Q_R_mantle_fun(S[i], t[i])

    Q_S_BMO[i] = P_S_BMO(S[i], Xi_BMO[i]) * dT_CMBdt[i]
    Q_L_BMO[i] = P_L_BMO(S[i], Xi_BMO[i]) * dT_CMBdt[i]
    Q_G_BMO[i] = P_G_BMO(S[i], Xi_BMO[i]) * dT_CMBdt[i]
    Q_C_BMO[i] = P_C_BMO(S[i], Xi_BMO[i]) * dT_CMBdt[i]
    Q_R_BMO[i] = Q_R_BMO_fun(S[i], t[i])

    Q_S_core[i] = P_S_core(S[i], Xi_core[i]) * dT_CMBdt[i]
    Q_L_core[i] = P_L_core(S[i], Xi_core[i]) * dT_CMBdt[i]
    Q_G_core[i] = P_G_core(S[i], Xi_core[i]) * dT_CMBdt[i]
    Q_C_core[i] = P_C_core(S[i], Xi_core[i]) * dT_CMBdt[i]
    Q_R_core[i] = Q_R_core_fun(S[i], t[i])

    Q_S_IC[i] = P_S_IC(S[i], Xi_core[i]) * dT_CMBdt[i]
    Q_R_IC[i] = Q_R_IC_fun(S[i], t[i])

    Q_planet[i] = Q_planet_fun(S[i])
    Q_BMO[i] = Q_BMO_fun(S[i], dT_CMBdt[i], t[i])
    Q_CMB[i] = Q_CMB_fun(S[i], Xi_core[i], dT_CMBdt[i], t[i])
    Q_ICB[i] = Q_ICB_fun(S[i], Xi_core[i], dT_CMBdt[i], t[i])

    Q_A_BMO[i] = Q_A_BMO_fun(S[i], k_BMO=8)
    Q_A_core[i] = Q_A_core_fun(S[i], k_core=70)

# %% Temperature post-processing
# =============================================================================
T_mantle, k_mantle = temperature_pp(t, S, Q_BMO, Q_planet)


# %% Save data
# =============================================================================
class Output:
    def __init__(self):
        self.T_CMB_initial = T_CMB_initial

        self.S0 = S0

        self.t_span = t_span
        self.t_eval = t_eval

        self.Y0 = Y0

        self.sol_BMO = sol_BMO

        self.t = t
        self.T_CMB = T_CMB
        self.Xi_BMO = Xi_BMO
        self.Xi_core = Xi_core
        self.dT_CMBdt = dT_CMBdt
        self.dXi_BMOdt = dXi_BMOdt
        self.dXi_coredt = dXi_coredt

        self.S = S

        self.T_BMO = T_BMO
        self.T_ICB = T_ICB
        self.T_center = T_center

        self.R_planet = R_planet
        self.R_BMO = R_BMO
        self.R_CMB = R_CMB
        self.R_ICB = R_ICB

        self.Q_S_mantle = Q_S_mantle
        self.Q_R_mantle = Q_R_mantle

        self.Q_S_BMO = Q_S_BMO
        self.Q_L_BMO = Q_L_BMO
        self.Q_G_BMO = Q_G_BMO
        self.Q_C_BMO = Q_C_BMO
        self.Q_R_BMO = Q_R_BMO

        self.Q_S_core = Q_S_core
        self.Q_L_core = Q_L_core
        self.Q_G_core = Q_G_core
        self.Q_C_core = Q_C_core
        self.Q_R_core = Q_R_core

        self.Q_S_IC = Q_S_IC
        self.Q_R_IC = Q_R_IC

        self.Q_planet = Q_planet
        self.Q_BMO = Q_BMO
        self.Q_CMB = Q_CMB
        self.Q_ICB = Q_ICB

        self.Q_A_BMO = Q_A_BMO
        self.Q_A_core = Q_A_core

        self.T_mantle = T_mantle
        self.k_mantle = k_mantle


Q = Output()

if save_data:
    with open(energy_path + "/thermal.pkl", "wb") as f:
        pkl.dump(Q, f)
