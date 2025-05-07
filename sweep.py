#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Victor Lherm

All rights reserved.

This code is provided for academic and research purposes only. Any use of this code in published work or presentations must include a citation to the original author. Unauthorized copying, modification, or distribution of this code is prohibited without prior written permission from the author.

Computes a post-processed data structure across a set of simulations with varying parameters, by combining outputs from multiple structural, thermal, buoyancy, and magnetic evolution models.

Created by Victor Lherm on 2025-05-07 10:00:00.
"""

# %% Modules
# =============================================================================
import os, glob, time, re, skimage
import numpy as np
import dill as pkl
import importlib as imp
import traceback as tb

from scipy import integrate, interpolate
from scipy.constants import G, mu_0, R, k, N_A, eV

from tqdm import tqdm

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

# %% Parameters
# =============================================================================
if __name__ == "__main__":
    # Main path (Sweep directory)
    main_path = "path_to_models_to_sweep"

    # Subfolder (Sweep subfolder)
    folder = "sweep_sub_folder"

    # Sweep parameter (M, CMF, or any sweep parameter)
    sw_param = "sweep_parameter"

    # Sweep (True if the sweep is in a subfolder)
    sw_flag = "True_if_sweep_in_subfolder"

    # Directories
    if not sw_flag:
        dirs = sorted(
            glob.glob(
                "{:s}/{:s}_*/{:s}".format(main_path, sw_param, folder),
                recursive=False,
            )
        )
    else:
        dirs = sorted(
            glob.glob(
                "{:s}/{:s}/{:s}_*".format(main_path, folder, sw_param),
                recursive=False,
            )
        )

    # Sweep parameter
    if not sw_flag:
        sw_val = np.zeros(len(dirs))
        for i, d in enumerate(dirs):
            sw_val[i] = float(re.search(r"{:s}_([\.\d]+)".format(sw_param), d).group(1))
    else:
        sw_val = np.zeros(len(dirs))
        for i, d in enumerate(dirs):
            sw_val[i] = float(
                re.search(r"{:s}_([-\.\w]+)".format(sw_param), d).group(1)
            )

    # Output directory
    if not sw_flag:
        os.makedirs(
            "{:s}/Sweep/{:s}".format(main_path, folder.split("/")[-1]),
            exist_ok=True,
        )

# %% Output
# =============================================================================
name_output_time = [
    "t",
    "t_1D",
    "t_ICB",
    "t_BMO",
    "R_ICB_1D",
    "R_BMO_1D",
    "T_center_1D",
    "T_ICB_1D",
    "T_CMB_1D",
    "T_BMO_1D",
    "Q_ICB_1D",
    "Q_CMB_1D",
    "Q_BMO_1D",
    "Q_planet_1D",
    "Omega",
    "R_planet",
    "R_BMO",
    "R_CMB",
    "R_ICB",
    "L_BMO",
    "L_OC",
    "M_BMO",
    "M_OC",
    "T_center",
    "T_ICB",
    "T_CMB",
    "T_BMO",
    "Ta_BMO_0_avg",
    "Ta_BMO_0_i_avg",
    "Ta_silicate_0_avg",
    "T_silicate_0_avg",
    "T_mantle_avg",
    "Ta_mantle_avg",
    "P_center",
    "P_ICB",
    "P_CMB",
    "P_BMO",
    "P_BMO_0_avg",
    "P_BMO_0_i_avg",
    "P_silicate_0_avg",
    "P_silicate_0_i_avg",
    "P_mantle_avg",
    "D_BMO",
    "Xi_BMO",
    "Xi_core",
    "k_mantle_avg",
    "k_mantle_max",
    "eta_mantle",
    "eta_mantle_1",
    "C_XI_BMO",
    "C_R_BMO",
    "C_XI_core",
    "C_R_core",
    "Q_ICB",
    "Q_CMB",
    "Q_BMO",
    "Q_planet",
    "Q_S_mantle",
    "Q_R_mantle",
    "Q_A_mantle",
    "Q_R_BMO",
    "Q_S_BMO",
    "Q_L_BMO",
    "Q_G_BMO",
    "Q_A_BMO",
    "Q_A_BMO_S2020",
    "Q_R_core",
    "Q_S_core",
    "Q_L_core",
    "Q_G_core",
    "Q_A_core",
    "Q_S_IC",
    "Q_R_IC",
    "E_R_BMO",
    "E_S_BMO",
    "E_L_BMO",
    "E_G_BMO",
    "E_CMB",
    "E_K_BMO",
    "E_Alpha_BMO",
    "E_R_core",
    "E_S_core",
    "E_L_core",
    "E_G_core",
    "E_ICB",
    "E_K_core",
    "E_Alpha_core",
    "E_PHI_BMO",
    "E_PHI_core",
    "E_R_BMO_S2020",
    "E_S_BMO_S2020",
    "E_L_BMO_S2020",
    "E_G_BMO_S2020",
    "E_CMB_S2020",
    "E_K_BMO_S2020",
    "E_Alpha_BMO_S2020",
    "E_R_core_S2020",
    "E_S_core_S2020",
    "E_L_core_S2020",
    "E_G_core_S2020",
    "E_ICB_S2020",
    "E_K_core_S2020",
    "E_Alpha_core_S2020",
    "E_PHI_BMO_S2020",
    "E_PHI_core_S2020",
    "p_S_BMO",
    "p_S_core",
    "U_S_BMO",
    "U_S_core",
    "Q_conv_BMO",
    "Rm_S_BMO",
    "B_S_BMO",
    "BS_S_BMO",
    "BS_S_BMO_cst",
    "k_BMO",
    "sigma_BMO",
    "Q_conv_core",
    "Rm_S_core",
    "B_S_core",
    "BS_S_core",
    "BS_S_core_cst",
    "k_core",
    "sigma_core",
    "p_S_BMO_S2020",
    "p_S_core_S2020",
    "U_S_BMO_S2020",
    "U_S_core_S2020",
    "Q_conv_BMO_S2020",
    "Rm_S_BMO_S2020",
    "B_S_BMO_S2020",
    "BS_S_BMO_S2020",
    "BS_S_BMO_cst_S2020",
    "k_BMO_S2020",
    "sigma_BMO_S2020",
    "Q_conv_core_S2020",
    "Rm_S_core_S2020",
    "B_S_core_S2020",
    "BS_S_core_S2020",
    "BS_S_core_cst_S2020",
    "k_core_S2020",
    "sigma_core_S2020",
    "dC0dr_TOT_BMO_MID",
    "dC0dr_TOT_BMO_S2020_MID",
    "dC0dr_TOT_core_MID",
    "mask_dC0dr_TOT_BMO",
    "mask_dC0dr_TOT_BMO_S2020",
    "mask_dC0dr_TOT_core",
    "sigma_BMO_min",
    "is_DYN_BMO_S2020",
    "sigma_BMO_N2025",
    "Rm_S_BMO_N2025_interp",
    "is_DYN_BMO_N2025",
    "k_core_min",
]

name_output_radius = [
    "radius",
    "rho",
    "g",
    "P",
    "T",
    "alpha",
    "gamma",
    "KT",
    "Cp",
    "radius_BMO_r_0",
    "Ta_BMO_r_0",
    "radius_BMO_r_0_i",
    "rho_BMO_r_0_i",
    "g_BMO_r_0_i",
    "P_BMO_r_0_i",
    "alpha_BMO_r_0_i",
    "gamma_BMO_r_0_i",
    "KT_BMO_r_0_i",
    "Cp_BMO_r_0_i",
    "radius_mantle",
    "T_mantle",
    "Ta_mantle",
    "Tm_mantle",
    "k_mantle",
    "radius_silicate",
    "radius_core",
    "T_core",
    "Tm_core",
]

name_output_misc = [
    "t_Rm_40",
    "sigma_BMO_Rm_40",
]

name_output = np.concatenate((name_output_time, name_output_radius, name_output_misc))


# %% Function
# =============================================================================
def energy_sw(path):
    print("Processing: {:s}".format(path))
    # General parameters
    # =============================================================================
    # Path
    os.chdir(path)

    # Import functions
    ld = imp.machinery.SourceFileLoader("functions", path + "/functions.py")
    ld.load_module("functions")
    from functions import (
        y2s,
        T_surface,
        D_BMO_fun,
        D_BMO_0,
        CXI_BMO,
        CR_BMO,
        CXI_core,
        CR_core,
        liquidus_mantle,
        liquidus_core,
        Q_A_BMO_fun,
        Omega_planet,
    )

    # Load file
    # =========================================================================
    # Energy
    with open(path + "/thermal.pkl", "rb") as f:
        Q = pkl.load(f)

    # Entropy
    with open(path + "/entropy_nominal.pkl", "rb") as f:
        E = pkl.load(f)

    with open(path + "/entropy_S2020.pkl", "rb") as f:
        E_S2020 = pkl.load(f)

    # Magnetic field
    with open(path + "/magnetic_nominal.pkl", "rb") as f:
        M = pkl.load(f)

    with open(path + "/magnetic_S2020.pkl", "rb") as f:
        M_S2020 = pkl.load(f)

    # Buoyancy
    with open(path + "/buoyancy.pkl", "rb") as f:
        B = pkl.load(f)

    # Parameters
    with open(path + "/param.pkl", "rb") as f:
        param = pkl.load(f)

    # Time
    # =========================================================================
    t = Q.t

    # 1D plots
    # =========================================================================
    t_1D = y2s(4.5e9)
    idx = np.argmin(np.abs(Q.t - t_1D))

    # Timescale
    if np.isnan(Q.R_ICB).all():
        t_ICB = np.nan
    else:
        t_ICB = Q.t[np.argwhere(~np.isnan(Q.R_ICB))[0]]
    if ~np.isnan(Q.R_BMO).any():
        t_BMO = np.nan
    else:
        t_BMO = Q.t[np.argwhere(np.isnan(Q.R_BMO))[0]]

    # Radius
    R_ICB_1D = Q.R_ICB[idx]
    R_BMO_1D = Q.R_BMO[idx]

    # Temperature
    T_center_1D = Q.T_center[idx]
    T_ICB_1D = Q.T_ICB[idx]
    T_CMB_1D = Q.T_CMB[idx]
    T_BMO_1D = Q.T_BMO[idx]

    # Heat flow
    Q_ICB_1D = Q.Q_ICB[idx]
    Q_CMB_1D = Q.Q_CMB[idx]
    Q_BMO_1D = Q.Q_BMO[idx]
    Q_planet_1D = Q.Q_planet[idx]

    # Rotation rate
    # =========================================================================
    Omega = Omega_planet(Q.t)

    # Radial structure
    # =========================================================================
    idx_radius = np.linspace(0, len(Q.t) - 1, 11).astype(int)

    # Planet
    (
        radius,
        rho,
        g,
        P,
        T,
        alpha,
        gamma,
        KT,
        Cp,
    ) = [np.zeros(len(idx_radius), dtype=object) for _ in range(9)]

    for i in range(len(idx_radius)):
        radius[i] = Q.S[idx_radius[i]].radius
        rho[i] = Q.S[idx_radius[i]].rho
        g[i] = Q.S[idx_radius[i]].g
        P[i] = Q.S[idx_radius[i]].P
        T[i] = Q.S[idx_radius[i]].T
        alpha[i] = Q.S[idx_radius[i]].alpha
        gamma[i] = Q.S[idx_radius[i]].gamma
        KT[i] = Q.S[idx_radius[i]].KT
        Cp[i] = Q.S[idx_radius[i]].Cp

    # BMO
    (
        radius_BMO_r_0,
        Ta_BMO_r_0,
    ) = [np.zeros(len(idx_radius), dtype=object) for _ in range(2)]

    for i in range(len(idx_radius)):
        radius_BMO_r_0[i] = Q.S[idx_radius[i]].radius_BMO_r_0
        Ta_BMO_r_0[i] = Q.S[idx_radius[i]].Ta_BMO_r_0

    (
        radius_BMO_r_0_i,
        rho_BMO_r_0_i,
        g_BMO_r_0_i,
        P_BMO_r_0_i,
        alpha_BMO_r_0_i,
        gamma_BMO_r_0_i,
        KT_BMO_r_0_i,
        Cp_BMO_r_0_i,
    ) = [np.zeros(len(idx_radius), dtype=object) for _ in range(8)]

    for i in range(len(idx_radius)):
        radius_BMO_r_0_i[i] = Q.S[idx_radius[i]].radius_BMO_r_0_i
        rho_BMO_r_0_i[i] = Q.S[idx_radius[i]].rho_BMO_r_0_i
        g_BMO_r_0_i[i] = Q.S[idx_radius[i]].g_BMO_r_0_i
        P_BMO_r_0_i[i] = Q.S[idx_radius[i]].P_BMO_r_0_i
        alpha_BMO_r_0_i[i] = Q.S[idx_radius[i]].alpha_BMO_r_0_i
        gamma_BMO_r_0_i[i] = Q.S[idx_radius[i]].gamma_BMO_r_0_i
        KT_BMO_r_0_i[i] = Q.S[idx_radius[i]].KT_BMO_r_0_i
        Cp_BMO_r_0_i[i] = Q.S[idx_radius[i]].Cp_BMO_r_0_i

    # Mantle
    (
        radius_mantle,
        T_mantle,
        Ta_mantle,
        Tm_mantle,
        k_mantle,
    ) = [np.zeros(len(idx_radius), dtype=object) for _ in range(5)]

    for i in range(len(idx_radius)):
        radius_mantle[i] = Q.S[idx_radius[i]].radius_mantle

        T_mantle[i] = Q.T_mantle[idx_radius[i]]

        Ta_mantle[i] = Q.S[idx_radius[i]].Ta_mantle

        Tm_mantle[i] = liquidus_mantle(
            Q.S[idx_radius[i]].P_silicate_0,
            Q.Xi_BMO[idx_radius[i]],
        )

        k_mantle[i] = Q.k_mantle[idx_radius[i]]

    # Silicate
    radius_silicate = np.zeros(len(idx_radius), dtype=object)

    for i in range(len(idx_radius)):
        radius_silicate[i] = Q.S[idx_radius[i]].radius_silicate

    # Core
    (
        radius_core,
        T_core,
        Tm_core,
    ) = [np.zeros(len(idx_radius), dtype=object) for _ in range(3)]

    for i in range(len(idx_radius)):
        radius_core[i] = Q.S[idx_radius[i]].radius_core

        T_core[i] = Q.S[idx_radius[i]].T_core

        Tm_core[i] = liquidus_core(Q.S[idx_radius[i]].P_core)

    # Radius
    # =========================================================================
    R_planet = Q.R_planet

    R_BMO = Q.R_BMO
    R_CMB = Q.R_CMB
    R_ICB = Q.R_ICB

    L_BMO = Q.R_BMO - Q.R_CMB[0]
    L_OC = Q.R_CMB - np.where(np.isnan(Q.R_ICB), 0, Q.R_ICB)

    # Mass
    # =========================================================================
    M_BMO = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        M_BMO[i] = integrate.trapezoid(
            4 * np.pi * Q.S[i].radius_BMO_r_0 ** 2 * Q.S[i].rho_BMO_r_0_i,
            Q.S[i].radius_BMO_r_0,
        )
    M_OC = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        M_OC[i] = integrate.trapezoid(
            4 * np.pi * Q.S[i].radius_OC ** 2 * Q.S[i].rho_OC,
            Q.S[i].radius_OC,
        )

    # Temperature
    # =========================================================================
    T_center = Q.T_center
    T_ICB = Q.T_ICB
    T_CMB = Q.T_CMB
    T_BMO = Q.T_BMO

    Ta_BMO_0_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        Ta_BMO_0_avg[i] = integrate.trapezoid(
            Q.S[i].Ta_BMO_r_0 * Q.S[i].radius_BMO_r_0 ** 2,
            Q.S[i].radius_BMO_r_0,
        ) / integrate.trapezoid(
            Q.S[i].radius_BMO_r_0 ** 2,
            Q.S[i].radius_BMO_r_0,
        )

    Ta_BMO_0_i_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        Ta_BMO_0_i_avg[i] = integrate.trapezoid(
            Q.S[i].Ta_BMO_r_0_i * Q.S[i].radius_BMO_r_0_i ** 2,
            Q.S[i].radius_BMO_r_0_i,
        ) / integrate.trapezoid(
            Q.S[i].radius_BMO_r_0_i ** 2,
            Q.S[i].radius_BMO_r_0_i,
        )

    Ta_silicate_0_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        radius_avg = np.concatenate((Q.S[i].radius_BMO_r_0, Q.S[i].radius_mantle))
        T_avg = np.concatenate((Q.S[i].Ta_BMO_r_0, Q.S[i].Ta_mantle))
        Ta_silicate_0_avg[i] = integrate.trapezoid(
            T_avg * radius_avg**2,
            radius_avg,
        ) / integrate.trapezoid(
            radius_avg**2,
            radius_avg,
        )

    T_silicate_0_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        radius_avg = np.concatenate((Q.S[i].radius_BMO_r_0, Q.S[i].radius_mantle))
        T_avg = np.concatenate((Q.S[i].Ta_BMO_r_0, Q.T_mantle[i]))
        T_silicate_0_avg[i] = integrate.trapezoid(
            T_avg * radius_avg**2,
            radius_avg,
        ) / integrate.trapezoid(
            radius_avg**2,
            radius_avg,
        )

    T_mantle_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        T_mantle_avg[i] = integrate.trapezoid(
            Q.T_mantle[i] * Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        ) / integrate.trapezoid(
            Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        )

    Ta_mantle_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        Ta_mantle_avg[i] = integrate.trapezoid(
            Q.S[i].Ta_mantle * Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        ) / integrate.trapezoid(
            Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        )

    # Pressure
    # =========================================================================
    P_center, P_ICB, P_CMB, P_BMO = [np.zeros(len(Q.S)) for _ in range(4)]
    for i in range(len(Q.S)):
        P_center[i] = Q.S[i].P_center
        P_ICB[i] = Q.S[i].P_ICB
        P_CMB[i] = Q.S[i].P_CMB
        P_BMO[i] = Q.S[i].P_BMO

    P_BMO_0_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        P_BMO_0_avg[i] = integrate.trapezoid(
            Q.S[i].P_BMO_r_0 * Q.S[i].radius_BMO_r_0 ** 2,
            Q.S[i].radius_BMO_r_0,
        ) / integrate.trapezoid(
            Q.S[i].radius_BMO_r_0 ** 2,
            Q.S[i].radius_BMO_r_0,
        )

    P_BMO_0_i_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        P_BMO_0_i_avg[i] = integrate.trapezoid(
            Q.S[i].P_BMO_r_0_i * Q.S[i].radius_BMO_r_0_i ** 2,
            Q.S[i].radius_BMO_r_0_i,
        ) / integrate.trapezoid(
            Q.S[i].radius_BMO_r_0_i ** 2,
            Q.S[i].radius_BMO_r_0_i,
        )

    P_silicate_0_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        P_silicate_0_avg[i] = integrate.trapezoid(
            Q.S[i].P_silicate_0 * Q.S[i].radius_silicate_0 ** 2,
            Q.S[i].radius_silicate_0,
        ) / integrate.trapezoid(
            Q.S[i].radius_silicate_0 ** 2,
            Q.S[i].radius_silicate_0,
        )

    P_silicate_0_i_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        P_silicate_0_i_avg[i] = integrate.trapezoid(
            Q.S[i].P_silicate_0_i * Q.S[i].radius_silicate_0_i ** 2,
            Q.S[i].radius_silicate_0_i,
        ) / integrate.trapezoid(
            Q.S[i].radius_silicate_0_i ** 2,
            Q.S[i].radius_silicate_0_i,
        )

    P_mantle_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        P_mantle_avg[i] = integrate.trapezoid(
            Q.S[i].P_mantle * Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        ) / integrate.trapezoid(
            Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        )

    # Composition
    # =========================================================================
    D_BMO = D_BMO_fun(Q.Xi_BMO, D_BMO_0)
    Xi_BMO = Q.Xi_BMO
    Xi_core = Q.Xi_core

    # Thermal conductivity
    # =========================================================================
    k_mantle_avg = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        k_mantle_avg[i] = integrate.trapezoid(
            Q.k_mantle[i] * Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        ) / integrate.trapezoid(
            Q.S[i].radius_mantle ** 2,
            Q.S[i].radius_mantle,
        )
    k_mantle_max = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        k_mantle_max[i] = np.max(Q.k_mantle[i])

    # Viscosity
    # =========================================================================
    eta_mantle, eta_mantle_1 = [np.zeros(len(Q.S)) for _ in range(2)]
    for i in range(len(Q.S)):
        eta_mantle[i] = Q.S[i].eta_mantle
        eta_mantle_1[i] = Q.S[i].eta_mantle_1

    # Heat flow
    # =========================================================================
    C_XI_BMO = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        C_XI_BMO[i] = CXI_BMO(Q.S[i], Q.Xi_BMO[i])
    C_R_BMO = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        C_R_BMO[i] = CR_BMO(Q.S[i], Q.Xi_BMO[i])
    C_XI_core = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        C_XI_core[i] = CXI_core(Q.S[i], Q.Xi_core[i])
    C_R_core = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        C_R_core[i] = CR_core(Q.S[i], Q.Xi_core[i])

    Q_ICB = Q.Q_ICB
    Q_CMB = Q.Q_CMB
    Q_BMO = Q.Q_BMO
    Q_planet = Q.Q_planet

    Q_S_mantle = Q.Q_S_mantle
    Q_R_mantle = Q.Q_R_mantle

    Q_A_mantle = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        dTadr = (np.nanmin([Q.T_BMO[i], Q.T_CMB[i]]) - T_surface) / (
            Q.R_planet[i] - np.nanmax([Q.R_BMO[i], Q.R_CMB[i]])
        )
        Q_A_mantle[i] = k_mantle_avg[i] * dTadr * 4 * np.pi * Q.R_planet[i] ** 2

    Q_R_BMO = Q.Q_R_BMO
    Q_S_BMO = Q.Q_S_BMO
    Q_L_BMO = Q.Q_L_BMO
    Q_G_BMO = Q.Q_G_BMO
    Q_A_BMO = Q.Q_A_BMO

    Q_A_BMO_S2020 = np.zeros(len(Q.S))
    for i in range(len(Q.S)):
        Q_A_BMO_S2020[i] = Q_A_BMO_fun(Q.S[i], E_S2020.k_BMO[i])

    Q_R_core = Q.Q_R_core
    Q_S_core = Q.Q_S_core
    Q_L_core = Q.Q_L_core
    Q_G_core = Q.Q_G_core
    Q_A_core = Q.Q_A_core

    Q_S_IC = Q.Q_S_IC
    Q_R_IC = Q.Q_R_IC

    # Entropy budget
    # =========================================================================
    # Nominal
    E_R_BMO = E.E_R_BMO
    E_S_BMO = E.E_S_BMO
    E_L_BMO = E.E_L_BMO
    E_G_BMO = E.E_G_BMO
    E_CMB = E.E_CMB
    E_K_BMO = E.E_K_BMO
    E_Alpha_BMO = E.E_Alpha_BMO

    E_R_core = E.E_R_core
    E_S_core = E.E_S_core
    E_L_core = E.E_L_core
    E_G_core = E.E_G_core
    E_ICB = E.E_ICB
    E_K_core = E.E_K_core
    E_Alpha_core = E.E_Alpha_core

    E_PHI_BMO = E.E_phi_BMO
    E_PHI_core = E.E_phi_core

    # Stixrude et al. (2020)
    E_R_BMO_S2020 = E_S2020.E_R_BMO
    E_S_BMO_S2020 = E_S2020.E_S_BMO
    E_L_BMO_S2020 = E_S2020.E_L_BMO
    E_G_BMO_S2020 = E_S2020.E_G_BMO
    E_CMB_S2020 = E_S2020.E_CMB
    E_K_BMO_S2020 = E_S2020.E_K_BMO
    E_Alpha_BMO_S2020 = E_S2020.E_Alpha_BMO

    E_R_core_S2020 = E_S2020.E_R_core
    E_S_core_S2020 = E_S2020.E_S_core
    E_L_core_S2020 = E_S2020.E_L_core
    E_G_core_S2020 = E_S2020.E_G_core
    E_ICB_S2020 = E_S2020.E_ICB
    E_K_core_S2020 = E_S2020.E_K_core
    E_Alpha_core_S2020 = E_S2020.E_Alpha_core

    E_PHI_BMO_S2020 = E_S2020.E_phi_BMO
    E_PHI_core_S2020 = E_S2020.E_phi_core

    # Magnetic field
    # =========================================================================
    # Nominal
    p_S_BMO = M.p_S_BMO
    p_S_core = M.p_S_core

    U_S_BMO = M.U_S_BMO
    U_S_core = M.U_S_core

    Q_conv_BMO = M.Q_conv_BMO
    Rm_S_BMO = M.Rm_S_BMO
    B_S_BMO = M.B_S_BMO
    BS_S_BMO = M.BS_S_BMO
    BS_S_BMO_cst = M.BS_S_BMO_cst

    k_BMO = M.k_BMO
    sigma_BMO = M.sigma_BMO

    Q_conv_core = M.Q_conv_core
    Rm_S_core = M.Rm_S_core
    B_S_core = M.B_S_core
    BS_S_core = M.BS_S_core
    BS_S_core_cst = M.BS_S_core_cst

    k_core = M.k_core
    sigma_core = M.sigma_core

    # Stixrude et al. (2020)
    p_S_BMO_S2020 = M_S2020.p_S_BMO
    p_S_core_S2020 = M_S2020.p_S_core

    U_S_BMO_S2020 = M_S2020.U_S_BMO
    U_S_core_S2020 = M_S2020.U_S_core

    Q_conv_BMO_S2020 = M_S2020.Q_conv_BMO
    Rm_S_BMO_S2020 = M_S2020.Rm_S_BMO
    B_S_BMO_S2020 = M_S2020.B_S_BMO
    BS_S_BMO_S2020 = M_S2020.BS_S_BMO
    BS_S_BMO_cst_S2020 = M_S2020.BS_S_BMO_cst

    k_BMO_S2020 = M_S2020.k_BMO
    sigma_BMO_S2020 = M_S2020.sigma_BMO

    Q_conv_core_S2020 = M_S2020.Q_conv_core
    Rm_S_core_S2020 = M_S2020.Rm_S_core
    B_S_core_S2020 = M_S2020.B_S_core
    BS_S_core_S2020 = M_S2020.BS_S_core
    BS_S_core_cst_S2020 = M_S2020.BS_S_core_cst

    k_core_S2020 = M_S2020.k_core
    sigma_core_S2020 = M_S2020.sigma_core

    # Buoyancy
    # =========================================================================
    dC0dr_TOT_BMO_MID = B.dC0dr_TOT_BMO[
        :,
        int(
            np.size(
                B.dC0dr_TOT_BMO,
                1,
            )
            / 2
        ),
    ]
    dC0dr_TOT_BMO_S2020_MID = B.dC0dr_TOT_BMO_S2020[
        :,
        int(
            np.size(
                B.dC0dr_TOT_BMO_S2020,
                1,
            )
            / 2
        ),
    ]
    dC0dr_TOT_core_MID = B.dC0dr_TOT_core[
        :,
        int(
            np.size(
                B.dC0dr_TOT_core,
                1,
            )
            / 2
        ),
    ]

    mask_dC0dr_TOT_BMO = np.sum(
        (B.dC0dr_TOT_BMO < 0),
        axis=1,
    ) / np.size(B.dC0dr_TOT_BMO, 1)
    mask_dC0dr_TOT_BMO_S2020 = np.sum(
        (B.dC0dr_TOT_BMO_S2020 < 0),
        axis=1,
    ) / np.size(B.dC0dr_TOT_BMO_S2020, 1)
    mask_dC0dr_TOT_core = np.sum(
        (B.dC0dr_TOT_core < 0),
        axis=1,
    ) / np.size(B.dC0dr_TOT_core, 1)

    # Transport properties
    # =========================================================================
    # Electrical conductivity of the BMO
    with open(path + "/magnetic_sigma_BMO_Lo_S2020.pkl", "rb") as f:
        M_dict = pkl.load(f).__dict__

    class Output:
        def __init__(self):
            pass

    M_BMO_Lo = Output()
    for i, n in enumerate(M_dict["0"].keys()):
        shape = tuple(
            x
            for y in (len(M_dict), np.shape(M_dict["0"][n]))
            for x in (y if isinstance(y, tuple) else (y,))
        )
        value = np.zeros(shape)
        for j in range(len(M_dict)):
            value[j] = M_dict[str(j)][n]
        setattr(M_BMO_Lo, n, value)

    try:
        i_BMO_1K = np.nanargmin(np.abs(T_CMB - T_BMO - 1))
        t_BMO_1K = Q.t[i_BMO_1K]
    except:
        i_BMO_1K = -1
        t_BMO_1K = np.nan

    mask_BMO = np.where(Q.t < t_BMO_1K, 1, np.nan)

    contour = skimage.measure.find_contours(
        M_BMO_Lo.Rm_S_BMO * mask_BMO, param.Rm_crit_BMO
    )
    if len(contour) > 0:
        x, y = np.arange(0, len(Q.t)), np.arange(0, len(M_BMO_Lo.sigma_BMO))
        x_m, y_m = np.meshgrid(x, y)

        t_m, sigma_BMO_m = np.meshgrid(Q.t, M_BMO_Lo.sigma_BMO)

        interp_x = interpolate.RegularGridInterpolator(
            (y, x),
            t_m * mask_BMO,
            method="linear",
            bounds_error=False,
        )
        interp_y = interpolate.RegularGridInterpolator(
            (y, x),
            sigma_BMO_m * mask_BMO,
            method="linear",
            bounds_error=False,
        )

        t_Rm_40 = [interp_x(c) for c in contour]
        sigma_BMO_Rm_40 = [interp_y(c) for c in contour]
        sigma_BMO_min = np.min([np.min(c) for c in sigma_BMO_Rm_40])
    else:
        t_Rm_40 = np.nan
        sigma_BMO_Rm_40 = np.nan
        sigma_BMO_min = np.nan

    interp = interpolate.RegularGridInterpolator(
        (M_BMO_Lo.sigma_BMO, Q.t),
        M_BMO_Lo.Rm_S_BMO * mask_BMO,
        method="linear",
        bounds_error=False,
    )

    # Stixrude et al. (2020)
    Rm_S_BMO_S2020_interp = interp(np.stack((sigma_BMO_S2020, Q.t), axis=1))

    is_DYN_BMO_S2020 = np.any(Rm_S_BMO_S2020_interp >= param.Rm_crit_BMO)

    # Nakajima et al. (2025)
    def sigma_BMO_N2025_fun(P, T):
        sigma_0 = 4.9673e5
        DE = 292.370e3
        DV = 4.4706e-8
        sigma = sigma_0 * np.exp(-(DE + DV * P) / (R * T))
        return sigma

    sigma_BMO_N2025 = sigma_BMO_N2025_fun(P_BMO_0_i_avg, Ta_BMO_0_avg)

    Rm_S_BMO_N2025_interp = interp(np.stack((sigma_BMO_N2025, Q.t), axis=1))

    is_DYN_BMO_N2025 = np.any(Rm_S_BMO_N2025_interp >= param.Rm_crit_BMO)

    # Thermal conductivity of the core
    with open(path + "/entropy_k_core_Lo_S2020.pkl", "rb") as f:
        E_dict = pkl.load(f).__dict__

    class Output:
        def __init__(self):
            pass

    E_core_Lo = Output()
    for i, n in enumerate(E_dict["0"].keys()):
        shape = tuple(
            x
            for y in (len(E_dict), np.shape(E_dict["0"][n]))
            for x in (y if isinstance(y, tuple) else (y,))
        )
        value = np.zeros(shape)
        for j in range(len(E_dict)):
            value[j] = E_dict[str(j)][n]
        setattr(E_core_Lo, n, value)

    idx = np.argsort(E_core_Lo.E_phi_core[:, 0])
    k_core_min = np.interp(0, E_core_Lo.E_phi_core[idx, 0], E_core_Lo.k_core[idx, 0])

    # Output
    # =========================================================================
    out = np.zeros(
        len(name_output),
        dtype=object,
    )
    for i, n in enumerate(name_output):
        out[i] = locals()[n]

    return out


# %% MPI Execution
# =============================================================================
# Main function
def fun_MPI(path):
    try:
        out = energy_sw(path)
    except Exception as e:
        print("Error in {:s}\n {:s}".format(path, tb.format_exc()))
        out = np.array(
            [None] * len(name_output),
            dtype=object,
        )
    return out


if __name__ == "__main__":
    tic = time.time()

    # Output
    out_array = np.zeros(len(dirs), dtype=object)

    # Parallel pool
    comm = MPI.COMM_WORLD
    rank = comm.Get_size()

    print("Starting parallel pool on %d cores..." % rank)

    with MPIPoolExecutor() as pool:
        for i, result in enumerate(pool.map(fun_MPI, dirs)):
            out_array[i] = result

    print("Done.")

    toc = time.time() - tic
    print("Elapsed time: %.1f s" % toc)


# %% Save data
# =============================================================================
if __name__ == "__main__":
    # Sort
    sort_idx = np.argsort(sw_val).flatten()

    # Output class
    class Output:
        def __init__(self):
            self.sw_param = sw_param
            self.sw_val = sw_val[sort_idx]

    SW = Output()

    for i, n in enumerate(name_output):
        none = np.zeros(len(out_array), dtype=bool)
        value = np.zeros(len(out_array), dtype=object)
        for j in range(len(out_array)):
            value[j] = out_array[j][i]
            if value[j] is None:
                none[j] = True

        idx = np.argwhere(none == False).flatten()
        if len(idx) > 0:
            for j in range(len(out_array)):
                if none[j]:
                    value[j] = np.zeros_like(value[idx[0]]) * np.nan

        if n in name_output_time:
            value = np.vstack(value).astype(float)

        setattr(SW, n, value[sort_idx])

    # Parameters
    param = np.zeros(len(dirs), dtype=object)
    for i in range(len(dirs)):
        with open(dirs[i] + "/param.pkl", "rb") as f:
            param[i] = pkl.load(f)

    setattr(SW, "param", param[sort_idx])

    # Conversion class
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    if not sw_flag:
        with open(
            "{:s}/Sweep/{:s}/sweep.pkl".format(
                main_path,
                folder.split("/")[-1],
            ),
            "wb",
        ) as f:
            pkl.dump(Struct(**SW.__dict__), f)
    else:
        with open(
            "{:s}/{:s}/sweep.pkl".format(
                main_path,
                folder,
            ),
            "wb",
        ) as f:
            pkl.dump(Struct(**SW.__dict__), f)
