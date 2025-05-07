#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Victor Lherm

All rights reserved.

This code is provided for academic and research purposes only. Any use of this code in published work or presentations must include a citation to the original author. Unauthorized copying, modification, or distribution of this code is prohibited without prior written permission from the author.

Computes the magnetic evolution of a planet based on a thermal evolution obtained from thermal.py.

Created by Victor Lherm on 2025-05-07 10:00:00.
"""

# %% Modules
# =============================================================================
import os, glob, time, re
import numpy as np
import dill as pkl
import importlib as imp
import traceback as tb

from scipy import optimize, integrate
from scipy.constants import G, mu_0, R, k, N_A, eV

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

# %% Paths
# =============================================================================
dirs = sorted(glob.glob("path_to_models/*", recursive=False))

# %% Output
# =============================================================================
name_output = [
    "entropy_nominal",
    "entropy_S2020",
    "entropy_sigma_BMO_Lo",
    "entropy_sigma_BMO_Lo_S2020",
    "entropy_k_core_Lo",
    "entropy_k_core_Lo_S2020",
    "magnetic_nominal",
    "magnetic_S2020",
    "magnetic_sigma_BMO_Lo",
    "magnetic_sigma_BMO_Lo_S2020",
    "magnetic_k_core_Lo",
    "magnetic_k_core_Lo_S2020",
]

# %% Parameters
# =============================================================================
# Sweep parameters
N_sweep = 128

k_BMO_sw = np.logspace(0, 2, N_sweep)
k_core_sw = np.logspace(0, 4, N_sweep)

sigma_BMO_sw = np.logspace(2, 6, N_sweep)
sigma_core_sw = np.logspace(4, 7, N_sweep)

# Universal parameters
Lorenz = np.pi**2 / 3 * (k / eV) ** 2

# Basal magma ocean
alpha_D_BMO = 1e-12

k_BMO = 8
k_BMO_0 = 1.16
rho_0_k_BMO = 2580.7
T_0_k_BMO = 3000
g_k_BMO = 1.75
a_k_BMO = -0.09

sigma_BMO = 3e4

c_B_BMO = 1.17
e_B_BMO = 0.34
c_U_BMO = 1.31
e_U_BMO = 0.42
c_b_dip_BMO = 7.3
f_ohm_BMO = 0.9
Rm_crit_BMO = 40

# Core
alpha_D_core = 1e-12

k_core = 70
sigma_core = 1.0e6

c_B_core = 1.17
e_B_core = 0.34
c_U_core = 1.31
e_U_core = 0.42
c_b_dip_core = 7.3
f_ohm_core = 0.9
Rm_crit_core = 40

# Parameter output
name_param = [
    "N_sweep",
    "k_BMO_sw",
    "k_core_sw",
    "sigma_BMO_sw",
    "sigma_core_sw",
    "Lorenz",
    "alpha_D_BMO",
    "k_BMO",
    "k_BMO_0",
    "rho_0_k_BMO",
    "T_0_k_BMO",
    "g_k_BMO",
    "a_k_BMO",
    "sigma_BMO",
    "c_B_BMO",
    "e_B_BMO",
    "c_U_BMO",
    "e_U_BMO",
    "c_b_dip_BMO",
    "f_ohm_BMO",
    "Rm_crit_BMO",
    "alpha_D_core",
    "k_core",
    "sigma_core",
    "c_B_core",
    "e_B_core",
    "c_U_core",
    "e_U_core",
    "c_b_dip_core",
    "f_ohm_core",
    "Rm_crit_core",
]


# %% Function
# =============================================================================
def dynamo_sw(path):
    print("Processing: %s" % path)
    # General parameters
    # =============================================================================
    # Paths
    simu_path = path

    # Import functions
    os.chdir(simu_path)
    ld = imp.machinery.SourceFileLoader("functions", simu_path + "/functions.py")
    ld.load_module("functions")
    from functions import (
        N,
        Q_R_BMO_fun,
        Q_R_core_fun,
        P_G_BMO,
        P_G_core,
        P_L_core,
        P_L_BMO,
        Q_CMB_fun,
        Q_ICB_fun,
        D_BMO_0,
        D_BMO_fun,
        Drho_BMO,
        Drho_ICB,
        Omega_planet,
    )

    # Load data
    with open(simu_path + "/thermal.pkl", "rb") as f:
        Q = pkl.load(f)

    # Conductivity definitions
    # =============================================================================
    # Electrical conductivity (Stixrude et al., 2020)
    def sigma_BMO_S2020_fun(P, T):
        sigma_0_el, E_s_el, V_s_el = 1.754e9, 108.6e3, 0.0611e-6
        sigma_el = sigma_0_el / T * np.exp(-(E_s_el + P * V_s_el) / (R * T))

        sigma_0_ion, E_s_ion, V_s_ion = 1.0811e9, 131.0e3, 0.437e-6
        sigma_ion = sigma_0_ion / T * np.exp(-(E_s_ion + P * V_s_ion) / (R * T))

        sigma = sigma_el + sigma_ion
        return sigma

    # Conversions
    def k_BMO_fun(sigma, rho, T):
        k_ph = k_BMO_0 * (rho / rho_0_k_BMO) ** g_k_BMO * (T_0_k_BMO / T) ** a_k_BMO
        k_el = Lorenz * T * sigma
        k = k_ph + k_el
        return k

    def sigma_BMO_fun(k, rho, T):
        k_ph = k_BMO_0 * (rho / rho_0_k_BMO) ** g_k_BMO * (T_0_k_BMO / T) ** a_k_BMO
        k_el = k - k_ph
        sigma = k_el / (Lorenz * T)
        return sigma

    def k_core_fun(sigma, T):
        k = Lorenz * T * sigma
        return k

    def sigma_core_fun(k, T):
        sigma = k / (Lorenz * T)
        return sigma

    # Averages
    # =============================================================================
    def rho_BMO_AVG_fun(S):
        rho_BMO_AVG = integrate.trapezoid(
            S.radius_BMO_r_0**2 * S.rho_BMO_r_0_i,
            S.radius_BMO_r_0,
        ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
        return rho_BMO_AVG

    def rho_OC_AVG_fun(S):
        rho_OC_AVG = integrate.trapezoid(
            S.radius_OC**2 * S.rho_OC,
            S.radius_OC,
        ) / integrate.trapezoid(S.radius_OC**2, S.radius_OC)
        return rho_OC_AVG

    def T_BMO_AVG_fun(S):
        T_BMO_AVG = integrate.trapezoid(
            S.radius_BMO_r_0**2 * S.Ta_BMO_r_0,
            S.radius_BMO_r_0,
        ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
        return T_BMO_AVG

    def T_OC_AVG_fun(S):
        T_OC_AVG = integrate.trapezoid(
            S.radius_OC**2 * S.T_OC,
            S.radius_OC,
        ) / integrate.trapezoid(S.radius_OC**2, S.radius_OC)
        return T_OC_AVG

    def P_BMO_AVG_fun(S):
        P_BMO_AVG = integrate.trapezoid(
            S.radius_BMO_r_0**2 * S.P_BMO_r_0_i,
            S.radius_BMO_r_0,
        ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
        return P_BMO_AVG

    def P_OC_AVG_fun(S):
        P_OC_AVG = integrate.trapezoid(
            S.radius_OC**2 * S.P_OC,
            S.radius_OC,
        ) / integrate.trapezoid(S.radius_OC**2, S.radius_OC)
        return P_OC_AVG
    
    # Dynamo scaling laws
    # =============================================================================
    data = np.loadtxt("data_A_2009.dat", delimiter="\t")
    (
        Ro_A2009,
        Lo_A2009,
        p_A2009,
        f_dip_A2009,
        f_ohm_A2009,
        f_i_A2009,
    ) = [data[:, i] for i in range(6)]

    data = np.loadtxt("data_C_2006.dat", delimiter="\t")
    (
        Ra_star_C2006,
        Nu_C2006,
        E_C2006,
        Pr_C2006,
        Ro_C2006,
        Lo_C2006,
        f_dip_C2006,
        f_ohm_C2006,
    ) = [data[:, i] for i in range(8)]

    chi_C2006 = 0.35
    gamma_C2006 = (
        (3 / 2)
        * (1 - chi_C2006) ** 2
        / (1 - chi_C2006**3)
        * (
            1 * ((3 / 5) * (1 - chi_C2006**5) / (1 - chi_C2006**3) - chi_C2006**2)
            + 1 * (1 - (3 / 5) * (1 - chi_C2006**5) / (1 - chi_C2006**3))
        )
    )
    Ek_C2006 = E_C2006 / Pr_C2006
    Nu_star_C2006 = (Nu_C2006 - 1) * Ek_C2006
    Ra_Q_star_C2006 = Ra_star_C2006 * Nu_star_C2006
    Ra_Q_C2006 = (chi_C2006 / (1 - chi_C2006) ** 2) * Ra_Q_star_C2006
    p_C2006 = gamma_C2006 * Ra_Q_C2006

    scaling_name = ["MLT", "CIA", "MAC"]
    alpha_Ro = [1 / 3, 2 / 5, 1 / 2]
    alpha_Lo = [1 / 3, 3 / 10, 1 / 4]

    # BMO
    mask = (Lo_A2009 > 0) & (f_dip_A2009 > 0.35)
    x_1 = p_A2009[mask]
    y_1 = Ro_A2009[mask]
    mask = (Lo_C2006 > 0) & (f_dip_C2006 > 0.35)
    x_2 = p_C2006[mask]
    y_2 = Ro_C2006[mask]
    p_fit = np.concatenate((x_1, x_2))
    Ro_fit = np.concatenate((y_1, y_2))

    c_U_BMO_fit = np.zeros(len(scaling_name))
    for i, alpha in enumerate(alpha_Ro):
        fun = lambda x: x + np.log(p_fit) * alpha - np.log(Ro_fit)
        lsq = optimize.least_squares(fun, np.log(c_U_BMO))
        c_U_BMO_fit[i] = np.exp(lsq.x)

    mask = (Lo_A2009 > 0) & (f_dip_A2009 > 0.35)
    x_1 = p_A2009[mask]
    y_1 = Lo_A2009[mask] / f_ohm_A2009[mask] ** 0.5
    x_1 = np.delete(x_1, -3)
    y_1 = np.delete(y_1, -3)
    mask = (Lo_C2006 > 0) & (f_dip_C2006 > 0.35)
    x_2 = p_C2006[mask]
    y_2 = Lo_C2006[mask] / f_ohm_C2006[mask] ** 0.5
    p_fit = np.concatenate((x_1, x_2))
    Lo_fit = np.concatenate((y_1, y_2))

    c_B_BMO_fit = np.zeros(len(scaling_name))
    for i, alpha in enumerate(alpha_Lo):
        fun = lambda x: x + np.log(p_fit) * alpha - np.log(Lo_fit)
        lsq = optimize.least_squares(fun, np.log(c_B_BMO))
        c_B_BMO_fit[i] = np.exp(lsq.x)

    # Core
    mask = (Lo_A2009 > 0) & (f_dip_A2009 > 0.35)
    x_1 = p_A2009[mask]
    y_1 = Ro_A2009[mask]
    mask = (Lo_C2006 > 0) & (f_dip_C2006 > 0.35)
    x_2 = p_C2006[mask]
    y_2 = Ro_C2006[mask]
    p_fit = np.concatenate((x_1, x_2))
    Ro_fit = np.concatenate((y_1, y_2))

    c_U_core_fit = np.zeros(len(scaling_name))
    for i, alpha in enumerate(alpha_Ro):
        fun = lambda x: x + np.log(p_fit) * alpha - np.log(Ro_fit)
        lsq = optimize.least_squares(fun, np.log(c_U_core))
        c_U_core_fit[i] = np.exp(lsq.x)

    mask = (Lo_A2009 > 0) & (f_dip_A2009 > 0.35)
    x_1 = p_A2009[mask]
    y_1 = Lo_A2009[mask] / f_ohm_A2009[mask] ** 0.5
    x_1 = np.delete(x_1, -3)
    y_1 = np.delete(y_1, -3)
    mask = (Lo_C2006 > 0) & (f_dip_C2006 > 0.35)
    x_2 = p_C2006[mask]
    y_2 = Lo_C2006[mask] / f_ohm_C2006[mask] ** 0.5
    p_fit = np.concatenate((x_1, x_2))
    Lo_fit = np.concatenate((y_1, y_2))

    c_B_core_fit = np.zeros(len(scaling_name))
    for i, alpha in enumerate(alpha_Lo):
        fun = lambda x: x + np.log(p_fit) * alpha - np.log(Lo_fit)
        lsq = optimize.least_squares(fun, np.log(c_B_core))
        c_B_core_fit[i] = np.exp(lsq.x)

    # Entropy budget
    # =============================================================================
    # Basal magma ocean
    # =============================================================================
    # BMO secular cooling (E_S_BMO)
    def E_S_BMO_fun(S, dT_CMBdt):
        E = -integrate.trapezoid(
            4
            * np.pi
            * S.radius_BMO_r_0**2
            * S.rho_BMO_r_0_i
            * S.Cp_BMO_r_0_i
            * (S.Ta_BMO_r_0 / S.T_CMB)
            * (1 / S.Ta_BMO_0 - 1 / S.Ta_BMO_r_0)
            * dT_CMBdt,
            S.radius_BMO_r_0,
        )
        return E

    # BMO radiogenic heating (E_R_BMO)
    def E_R_BMO_fun(S, t):
        M_BMO = integrate.trapezoid(
            4 * np.pi * S.radius_BMO_r_0**2 * S.rho_BMO_r_0_i, S.radius_BMO_r_0
        )
        E = integrate.trapezoid(
            4
            * np.pi
            * S.radius_BMO_r_0**2
            * S.rho_BMO_r_0_i
            * (1 / S.Ta_BMO_0 - 1 / S.Ta_BMO_r_0),
            S.radius_BMO_r_0,
        ) * (Q_R_BMO_fun(S, t) / M_BMO)
        return E

    # BMO latent heat (E_L_BMO)
    def E_L_BMO_fun():
        E = 0
        return E

    # BMO compositional/gravity heating (E_G_BMO)
    def E_G_BMO_fun(S, Xi_BMO, dT_CMBdt):
        E = P_G_BMO(S, Xi_BMO) * dT_CMBdt / S.Ta_BMO_0
        return E

    # CMB heat flow (E_CMB)
    def E_CMB_fun(S, Xi_core, dT_CMBdt, t):
        E = Q_CMB_fun(S, Xi_core, dT_CMBdt, t) * (1 / S.Ta_BMO_0 - 1 / S.T_CMB)
        return E

    # BMO thermal diffusion (E_K_BMO)
    def E_K_BMO_fun(S, k_BMO):
        E = integrate.trapezoid(
            4
            * np.pi
            * S.radius_BMO_r_0**2
            * k_BMO
            * (np.gradient(S.Ta_BMO_r_0, S.radius_BMO_r_0) / S.Ta_BMO_r_0) ** 2,
            S.radius_BMO_r_0,
        )
        return E

    # BMO molecular diffusion (E_ALPHA_BMO)
    def E_Alpha_BMO_fun(S, Xi_BMO):
        D_BMO = D_BMO_fun(Xi_BMO, D_BMO_0)
        DXi_BMO = Xi_BMO * (1 - D_BMO)
        alpha_c_BMO = Drho_BMO / (S.rho_mantle[0] * DXi_BMO)
        E = integrate.trapezoid(
            alpha_c_BMO**2
            * alpha_D_BMO
            * 4
            * np.pi
            * S.radius_BMO_r_0**2
            * S.g_BMO_r_0_i**2
            / S.Ta_BMO_r_0,
            S.radius_BMO_r_0,
        )
        return E

    # Core
    # =============================================================================
    # Core secular cooling (E_S_core)
    def E_S_core_fun(S, dT_CMBdt):
        E = -integrate.trapezoid(
            4
            * np.pi
            * S.radius_OC**2
            * S.rho_OC
            * S.Cp_OC
            * (1 / S.T_CMB)
            * (S.T_OC / S.T_CMB - 1)
            * dT_CMBdt,
            S.radius_OC,
        )
        return E

    # Core radiogenic heating (E_R_core)
    def E_R_core_fun(S, t):
        M_OC = integrate.trapezoid(4 * np.pi * S.radius_OC**2 * S.rho_OC, S.radius_OC)
        E = integrate.trapezoid(
            4 * np.pi * S.radius_OC**2 * S.rho_OC * (1 / S.T_CMB - 1 / S.T_OC),
            S.radius_OC,
        ) * (Q_R_core_fun(S, t) / M_OC)
        return E

    # Core latent heat (E_L_core)
    def E_L_core_fun(S, Xi_core, dT_CMBdt):
        E = P_L_core(S, Xi_core) * dT_CMBdt * (1 / S.T_CMB - 1 / S.T_ICB)
        return E

    # Core compositional/gravity heating (E_G_core)
    def E_G_core_fun(S, Xi_core, dT_CMBdt):
        E = P_G_core(S, Xi_core) * dT_CMBdt / S.T_CMB
        return E

    # Inner core heat flow (E_IC)
    def E_IC_fun(S, Xi_core, dT_CMBdt, t):
        E = Q_ICB_fun(S, Xi_core, dT_CMBdt, t) * (1 / S.T_CMB - 1 / S.T_ICB)
        return E

    # Core thermal diffusion (E_K_core)
    def E_K_core_fun(S, k_core):
        E = integrate.trapezoid(
            4
            * np.pi
            * S.radius_OC**2
            * k_core
            * (np.gradient(S.T_OC, S.radius_OC) / S.T_OC) ** 2,
            S.radius_OC,
        )
        return E

    # Core molecular diffusion (E_ALPHA_core)
    def E_Alpha_core_fun(S, Xi_core):
        DXi_core = Xi_core
        alpha_c_core = Drho_ICB / (S.rho_ICB_bot * DXi_core)
        E = integrate.trapezoid(
            alpha_c_core**2
            * alpha_D_core
            * 4
            * np.pi
            * S.radius_OC**2
            * S.g_OC**2
            / S.T_OC,
            S.radius_OC,
        )
        return E

    # Post-processing
    # =============================================================================
    # Entropy
    # =============================================================================
    def entropy_pp(Q, k_BMO, k_core):
        # BMO
        if isinstance(k_BMO, (int, float)):
            k_BMO = np.ones(len(Q.t)) * k_BMO

        E_S_BMO, E_R_BMO, E_L_BMO, E_G_BMO, E_CMB, E_K_BMO, E_Alpha_BMO = (
            np.zeros(len(Q.t)) for i in range(7)
        )
        E_phi_BMO = np.zeros(len(Q.t))
        for i in range(len(Q.t)):
            E_S_BMO[i] = E_S_BMO_fun(Q.S[i], Q.dT_CMBdt[i])
            E_R_BMO[i] = E_R_BMO_fun(Q.S[i], Q.t[i])
            E_L_BMO[i] = E_L_BMO_fun()
            E_G_BMO[i] = E_G_BMO_fun(Q.S[i], Q.Xi_BMO[i], Q.dT_CMBdt[i])
            E_CMB[i] = E_CMB_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i], Q.t[i])
            E_K_BMO[i] = E_K_BMO_fun(Q.S[i], k_BMO[i])
            E_Alpha_BMO[i] = E_Alpha_BMO_fun(Q.S[i], Q.Xi_BMO[i])

            E_phi_BMO[i] = np.nansum(
                [
                    E_S_BMO[i],
                    E_R_BMO[i],
                    E_L_BMO[i],
                    E_G_BMO[i],
                    E_CMB[i],
                    -E_K_BMO[i],
                    -E_Alpha_BMO[i],
                ],
                axis=0,
            )

        # Core
        if isinstance(k_core, (int, float)):
            k_core = np.ones(len(Q.t)) * k_core

        E_S_core, E_R_core, E_L_core, E_G_core, E_ICB, E_K_core, E_Alpha_core = (
            np.zeros(len(Q.t)) for i in range(7)
        )
        E_phi_core = np.zeros(len(Q.t))
        for i in range(len(Q.t)):
            E_S_core[i] = E_S_core_fun(Q.S[i], Q.dT_CMBdt[i])
            E_R_core[i] = E_R_core_fun(Q.S[i], Q.t[i])
            E_L_core[i] = E_L_core_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i])
            E_G_core[i] = E_G_core_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i])
            E_ICB[i] = E_IC_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i], Q.t[i])
            E_K_core[i] = E_K_core_fun(Q.S[i], k_core[i])
            E_Alpha_core[i] = E_Alpha_core_fun(Q.S[i], Q.Xi_core[i])

            E_phi_core[i] = np.nansum(
                [
                    E_S_core[i],
                    E_R_core[i],
                    E_L_core[i],
                    E_G_core[i],
                    E_ICB[i],
                    -E_K_core[i],
                    -E_Alpha_core[i],
                ],
                axis=0,
            )

            class Output:
                def __init__(self):
                    self.E_S_BMO = E_S_BMO
                    self.E_R_BMO = E_R_BMO
                    self.E_L_BMO = E_L_BMO
                    self.E_G_BMO = E_G_BMO
                    self.E_CMB = E_CMB
                    self.E_K_BMO = E_K_BMO
                    self.E_Alpha_BMO = E_Alpha_BMO
                    self.E_phi_BMO = E_phi_BMO
                    self.k_BMO = k_BMO

                    self.E_S_core = E_S_core
                    self.E_R_core = E_R_core
                    self.E_L_core = E_L_core
                    self.E_G_core = E_G_core
                    self.E_ICB = E_ICB
                    self.E_K_core = E_K_core
                    self.E_Alpha_core = E_Alpha_core
                    self.E_phi_core = E_phi_core
                    self.k_core = k_core

        return Output()

    # Magnetic field
    # =============================================================================
    def magnetic_pp(Q, E, k_BMO, k_core, sigma_BMO, sigma_core):
        # Thermal diffusivity
        # =============================================================================
        if isinstance(k_BMO, (int, float)):
            k_BMO = np.ones(len(Q.t)) * k_BMO
        if isinstance(k_core, (int, float)):
            k_core = np.ones(len(Q.t)) * k_core

        # Planet
        # =============================================================================
        Omega = np.zeros(len(Q.t))
        for i in range(len(Q.t)):
            Omega[i] = Omega_planet(Q.t[i])

        # BMO
        # =============================================================================
        (
            L_BMO,
            V_BMO,
            M_BMO,
            rho_BMO_AVG,
            alpha_BMO_AVG,
            Cp_BMO_AVG,
            g_BMO,
            psi_R_BMO_BMO,
            psi_R_BMO_CMB,
            psi_R_BMO_BMO_AVG,
            H_T_BMO,
            T_D_BMO,
        ) = (np.zeros(len(Q.t)) for i in range(12))

        for i in range(len(Q.t)):
            L_BMO[i] = Q.S[i].R_BMO - Q.S[0].R_CMB

            V_BMO[i] = integrate.trapezoid(
                4 * np.pi * Q.S[i].radius_BMO_r_0 ** 2, Q.S[i].radius_BMO_r_0
            )

            M_BMO[i] = integrate.trapezoid(
                4 * np.pi * Q.S[i].radius_BMO_r_0 ** 2 * Q.S[i].rho_BMO_r_0_i,
                Q.S[i].radius_BMO_r_0,
            )

            rho_BMO_AVG[i] = integrate.trapezoid(
                Q.S[i].radius_BMO_r_0 ** 2 * Q.S[i].rho_BMO_r_0_i,
                Q.S[i].radius_BMO_r_0,
            ) / integrate.trapezoid(Q.S[i].radius_BMO_r_0 ** 2, Q.S[i].radius_BMO_r_0)

            alpha_BMO_AVG[i] = integrate.trapezoid(
                Q.S[i].radius_BMO_r_0 ** 2 * Q.S[i].alpha_BMO_r_0_i,
                Q.S[i].radius_BMO_r_0,
            ) / integrate.trapezoid(Q.S[i].radius_BMO_r_0 ** 2, Q.S[i].radius_BMO_r_0)

            Cp_BMO_AVG[i] = integrate.trapezoid(
                Q.S[i].radius_BMO_r_0 ** 2 * Q.S[i].Cp_BMO_r_0_i,
                Q.S[i].radius_BMO_r_0,
            ) / integrate.trapezoid(Q.S[i].radius_BMO_r_0 ** 2, Q.S[i].radius_BMO_r_0)

            g_BMO[i] = Q.S[i].g_BMO_0_i

            psi_R_BMO_BMO[i] = Q.S[i].psi_R_BMO_BMO_r_0_i[-1]
            psi_R_BMO_CMB[i] = Q.S[i].psi_R_BMO_BMO_r_0_i[0]
            psi_R_BMO_BMO_AVG[i] = integrate.trapezoid(
                Q.S[i].radius_BMO_r_0 ** 2
                * Q.S[i].rho_BMO_r_0_i
                * Q.S[i].psi_R_BMO_BMO_r_0_i,
                Q.S[i].radius_BMO_r_0,
            ) / integrate.trapezoid(
                Q.S[i].radius_BMO_r_0 ** 2 * Q.S[i].rho_BMO_r_0_i, Q.S[i].radius_BMO_r_0
            )

            H_T_BMO[i] = integrate.trapezoid(
                Q.S[i].radius_BMO_r_0 ** 2
                * (Q.S[i].Cp_BMO_r_0_i / (Q.S[i].alpha_BMO_r_0_i * Q.S[i].g_BMO_r_0_i)),
                Q.S[i].radius_BMO_r_0,
            ) / integrate.trapezoid(Q.S[i].radius_BMO_r_0 ** 2, Q.S[i].radius_BMO_r_0)

            T_D_BMO[i] = 1 / (
                integrate.trapezoid(
                    Q.S[i].radius_BMO_r_0 ** 2
                    * Q.S[i].rho_BMO_r_0_i
                    / Q.S[i].Ta_BMO_r_0,
                    Q.S[i].radius_BMO_r_0,
                )
                / integrate.trapezoid(
                    Q.S[i].radius_BMO_r_0 ** 2 * Q.S[i].rho_BMO_r_0_i,
                    Q.S[i].radius_BMO_r_0,
                )
            )

        # Dissipation
        phi_o_BMO = T_D_BMO * np.nansum(
            [
                E.E_S_BMO,
                E.E_R_BMO,
                E.E_G_BMO,
                -E.E_K_BMO,
                -E.E_Alpha_BMO,
            ],
            axis=0,
        )
        phi_i_BMO = T_D_BMO * np.nansum(
            [
                E.E_CMB,
            ],
            axis=0,
        )
        phi_BMO = phi_o_BMO + phi_i_BMO

        p_S_BMO = phi_BMO / (M_BMO * Omega**3 * L_BMO**2)

        F_o_S_BMO = phi_o_BMO / (psi_R_BMO_BMO - psi_R_BMO_BMO_AVG)
        F_i_S_BMO = phi_i_BMO / (psi_R_BMO_BMO_AVG - psi_R_BMO_CMB)
        F_S_BMO = np.nansum([F_o_S_BMO, F_i_S_BMO], axis=0)
        fi_S_BMO = F_i_S_BMO / F_S_BMO

        U_S_BMO = (
            c_U_BMO * np.where(p_S_BMO < 0, np.nan, p_S_BMO) ** e_U_BMO * Omega * L_BMO
        )

        Rm_S_BMO = U_S_BMO * L_BMO * mu_0 * sigma_BMO

        B_S_BMO = (
            c_B_BMO
            * f_ohm_BMO**0.5
            * np.where(p_S_BMO < 0, np.nan, p_S_BMO) ** e_B_BMO
            * (rho_BMO_AVG * mu_0) ** 0.5
            * Omega
            * L_BMO
        )

        b_dip_S_BMO = (
            c_b_dip_BMO
            * L_BMO
            / Q.R_BMO
            * (1 + np.where(fi_S_BMO < 0, np.nan, fi_S_BMO))
        )
        TDM_S_BMO = (
            (4 * np.pi * Q.R_BMO**3) / (np.sqrt(2) * mu_0) * (B_S_BMO / b_dip_S_BMO)
        )
        BS_S_BMO = mu_0 / (4 * np.pi * Q.R_planet**3) * TDM_S_BMO

        BS_S_BMO_cst = (1 / 7) * (Q.R_BMO / Q.R_planet) ** 3 * B_S_BMO

        # Analytical scaling
        (
            U_fit_BMO,
            Rm_fit_BMO,
            B_fit_BMO,
            b_dip_fit_BMO,
            TDM_fit_BMO,
            BS_fit_BMO,
            BS_fit_BMO_cst,
        ) = (np.zeros((len(alpha_Ro), len(Q.t))) for i in range(7))

        for i in range(len(scaling_name)):
            U_fit_BMO[i] = (
                c_U_BMO_fit[i]
                * np.where(p_S_BMO < 0, np.nan, p_S_BMO) ** alpha_Ro[i]
                * Omega
                * L_BMO
            )
            Rm_fit_BMO[i] = U_fit_BMO[i] * L_BMO * mu_0 * sigma_BMO
            B_fit_BMO[i] = (
                c_B_BMO_fit[i]
                * f_ohm_BMO**0.5
                * np.where(p_S_BMO < 0, np.nan, p_S_BMO) ** alpha_Lo[i]
                * (rho_BMO_AVG * mu_0) ** 0.5
                * Omega
                * L_BMO
            )
            b_dip_fit_BMO[i] = (
                c_b_dip_BMO
                * L_BMO
                / Q.R_BMO
                * (1 + np.where(fi_S_BMO < 0, np.nan, fi_S_BMO))
            )
            TDM_fit_BMO[i] = (
                (4 * np.pi * Q.R_BMO**3)
                / (np.sqrt(2) * mu_0)
                * (B_fit_BMO[i] / b_dip_fit_BMO[i])
            )
            BS_fit_BMO[i] = mu_0 / (4 * np.pi * Q.R_planet**3) * TDM_fit_BMO[i]

            BS_fit_BMO_cst[i] = (1 / 7) * (Q.R_BMO / Q.R_planet) ** 3 * B_fit_BMO[i]

        # Core
        # =============================================================================
        (
            L_core,
            V_core,
            M_core,
            rho_OC_AVG,
            alpha_OC_AVG,
            Cp_OC_AVG,
            g_CMB,
            psi_R_CMB_CMB,
            psi_R_CMB_ICB,
            psi_R_CMB_core,
            H_T_core,
            T_D_core,
        ) = (np.zeros(len(Q.t)) for i in range(12))

        for i in range(len(Q.t)):
            L_core[i] = np.nansum([Q.S[i].R_CMB, -Q.S[i].R_ICB])

            V_core[i] = integrate.trapezoid(
                4 * np.pi * Q.S[i].radius_OC ** 2, Q.S[i].radius_OC
            )

            M_core[i] = integrate.trapezoid(
                4 * np.pi * Q.S[i].radius_OC ** 2 * Q.S[i].rho_OC, Q.S[i].radius_OC
            )

            rho_OC_AVG[i] = integrate.trapezoid(
                Q.S[i].radius_OC ** 2 * Q.S[i].rho_OC, Q.S[i].radius_OC
            ) / integrate.trapezoid(Q.S[i].radius_OC ** 2, Q.S[i].radius_OC)

            alpha_OC_AVG[i] = integrate.trapezoid(
                Q.S[i].radius_OC ** 2 * Q.S[i].alpha_OC, Q.S[i].radius_OC
            ) / integrate.trapezoid(Q.S[i].radius_OC ** 2, Q.S[i].radius_OC)

            Cp_OC_AVG[i] = integrate.trapezoid(
                Q.S[i].radius_OC ** 2 * Q.S[i].Cp_OC, Q.S[i].radius_OC
            ) / integrate.trapezoid(Q.S[i].radius_OC ** 2, Q.S[i].radius_OC)

            g_CMB[i] = Q.S[i].g_CMB

            psi_R_CMB_CMB[i] = Q.S[i].psi_R_CMB_OC[-1]
            psi_R_CMB_ICB[i] = Q.S[i].psi_R_CMB_OC[0]
            psi_R_CMB_core[i] = integrate.trapezoid(
                Q.S[i].radius_OC ** 2 * Q.S[i].rho_OC * Q.S[i].psi_R_CMB_OC,
                Q.S[i].radius_OC,
            ) / integrate.trapezoid(
                Q.S[i].radius_OC ** 2 * Q.S[i].rho_OC, Q.S[i].radius_OC
            )

            H_T_core[i] = integrate.trapezoid(
                Q.S[i].radius_OC ** 2
                * (Q.S[i].Cp_OC / (Q.S[i].alpha_OC * Q.S[i].g_OC)),
                Q.S[i].radius_OC,
            ) / integrate.trapezoid(Q.S[i].radius_OC ** 2, Q.S[i].radius_OC)

            T_D_core[i] = 1 / (
                integrate.trapezoid(
                    Q.S[i].radius_OC ** 2 * Q.S[i].rho_OC / Q.S[i].T_OC,
                    Q.S[i].radius_OC,
                )
                / integrate.trapezoid(
                    Q.S[i].radius_OC ** 2 * Q.S[i].rho_OC, Q.S[i].radius_OC
                )
            )

        # Dissipation
        phi_o_core = T_D_core * np.nansum(
            [
                E.E_S_core,
                E.E_R_core,
                -E.E_K_core,
                -E.E_Alpha_core,
            ],
            axis=0,
        )
        phi_i_core = T_D_core * np.nansum(
            [
                E.E_L_core,
                E.E_G_core,
                E.E_ICB,
            ],
            axis=0,
        )
        phi_core = phi_o_core + phi_i_core

        p_S_core = phi_core / (M_core * Omega**3 * L_core**2)

        F_o_S_core = phi_o_core / (psi_R_CMB_CMB - psi_R_CMB_core)
        F_i_S_core = phi_i_core / (psi_R_CMB_core - psi_R_CMB_ICB)
        F_S_core = np.nansum([F_o_S_core, F_i_S_core], axis=0)
        fi_S_core = F_i_S_core / F_S_core

        U_S_core = (
            c_U_core
            * np.where(p_S_core < 0, np.nan, p_S_core) ** e_U_core
            * Omega
            * L_core
        )

        Rm_S_core = U_S_core * L_core * mu_0 * sigma_core

        B_S_core = (
            c_B_core
            * f_ohm_core**0.5
            * np.where(p_S_core < 0, np.nan, p_S_core) ** e_B_core
            * (rho_OC_AVG * mu_0) ** 0.5
            * Omega
            * L_core
        )

        b_dip_S_core = (
            c_b_dip_core
            * L_core
            / Q.R_CMB
            * (1 + np.where(fi_S_core < 0, np.nan, fi_S_core))
        )
        TDM_S_core = (
            (4 * np.pi * Q.R_CMB**3) / (np.sqrt(2) * mu_0) * (B_S_core / b_dip_S_core)
        )
        BS_S_core = mu_0 / (4 * np.pi * Q.R_planet**3) * TDM_S_core

        BS_S_core_cst = (1 / 7) * (Q.R_CMB / Q.R_planet) ** 3 * B_S_core

        # Analytical scaling
        (
            U_fit_core,
            Rm_fit_core,
            B_fit_core,
            b_dip_fit_core,
            TDM_fit_core,
            BS_fit_core,
            BS_fit_core_cst,
        ) = (np.zeros((len(alpha_Ro), len(Q.t))) for i in range(7))

        for i in range(len(scaling_name)):
            U_fit_core[i] = (
                c_U_core_fit[i]
                * np.where(p_S_core < 0, np.nan, p_S_core) ** alpha_Ro[i]
                * Omega
                * L_core
            )
            Rm_fit_core[i] = U_fit_core[i] * L_core * mu_0 * sigma_core
            B_fit_core[i] = (
                c_B_core_fit[i]
                * f_ohm_core**0.5
                * np.where(p_S_core < 0, np.nan, p_S_core) ** alpha_Lo[i]
                * (rho_OC_AVG * mu_0) ** 0.5
                * Omega
                * L_core
            )
            b_dip_fit_core[i] = (
                c_b_dip_core
                * L_core
                / Q.R_CMB
                * (1 + np.where(fi_S_core < 0, np.nan, fi_S_core))
            )
            TDM_fit_core[i] = (
                (4 * np.pi * Q.R_CMB**3)
                / (np.sqrt(2) * mu_0)
                * (B_fit_core[i] / b_dip_fit_core[i])
            )
            BS_fit_core[i] = mu_0 / (4 * np.pi * Q.R_planet**3) * TDM_fit_core[i]

            BS_fit_core_cst[i] = (1 / 7) * (Q.R_CMB / Q.R_planet) ** 3 * B_fit_core[i]

        class Output:
            def __init__(self):
                self.p_S_BMO = p_S_BMO
                self.fi_S_BMO = fi_S_BMO
                self.b_dip_S_BMO = b_dip_S_BMO
                self.TDM_S_BMO = TDM_S_BMO

                self.U_S_BMO = U_S_BMO
                self.Rm_S_BMO = Rm_S_BMO
                self.B_S_BMO = B_S_BMO
                self.BS_S_BMO = BS_S_BMO
                self.BS_S_BMO_cst = BS_S_BMO_cst

                self.U_fit_BMO = U_fit_BMO
                self.Rm_fit_BMO = Rm_fit_BMO
                self.B_fit_BMO = B_fit_BMO
                self.BS_fit_BMO = BS_fit_BMO
                self.BS_fit_BMO_cst = BS_fit_BMO_cst

                self.Q_conv_BMO = Q_conv_BMO

                self.k_BMO = k_BMO
                self.sigma_BMO = sigma_BMO

                self.p_S_core = p_S_core
                self.fi_S_core = fi_S_core
                self.b_dip_S_core = b_dip_S_core
                self.TDM_S_core = TDM_S_core

                self.U_S_core = U_S_core
                self.Rm_S_core = Rm_S_core
                self.B_S_core = B_S_core
                self.BS_S_core = BS_S_core
                self.BS_S_core_cst = BS_S_core_cst

                self.U_fit_core = U_fit_core
                self.Rm_fit_core = Rm_fit_core
                self.B_fit_core = B_fit_core
                self.BS_fit_core = BS_fit_core
                self.BS_fit_core_cst = BS_fit_core_cst

                self.Q_conv_core = Q_conv_core

                self.k_core = k_core
                self.sigma_core = sigma_core

        return Output()

    # Sweeps
    # =============================================================================
    # Nominal
    def lorenz_sweep(
        param_name,
        k_BMO=k_BMO,
        k_core=k_core,
        sigma_BMO=sigma_BMO,
        sigma_core=sigma_core,
    ):
        if param_name == "k_BMO":
            param_sw = k_BMO_sw
        elif param_name == "k_core":
            param_sw = k_core_sw
        elif param_name == "sigma_BMO":
            param_sw = sigma_BMO_sw
        elif param_name == "sigma_core":
            param_sw = sigma_core_sw

        E, M = (np.zeros(len(param_sw), dtype=object) for i in range(2))
        for i, p in enumerate(param_sw):
            (T_BMO_AVG, T_OC_AVG, rho_BMO_AVG, rho_OC_AVG) = (
                np.zeros(len(Q.t)) for i in range(4)
            )

            for j in range(len(Q.t)):
                T_BMO_AVG[j] = T_BMO_AVG_fun(Q.S[j])
                T_OC_AVG[j] = T_OC_AVG_fun(Q.S[j])
                rho_BMO_AVG[j] = rho_BMO_AVG_fun(Q.S[j])
                rho_OC_AVG[j] = rho_OC_AVG_fun(Q.S[j])

            if param_name == "k_BMO":
                E[i] = entropy_pp(Q, p, k_core)
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    p,
                    k_core,
                    sigma_BMO_fun(p, rho_BMO_AVG, T_BMO_AVG),
                    sigma_core,
                )
            elif param_name == "k_core":
                E[i] = entropy_pp(Q, k_BMO, p)
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    k_BMO,
                    p,
                    sigma_BMO,
                    sigma_core_fun(p, T_OC_AVG),
                )
            elif param_name == "sigma_BMO":
                E[i] = entropy_pp(Q, k_BMO_fun(p, rho_BMO_AVG, T_BMO_AVG), k_core)
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    k_BMO_fun(p, rho_BMO_AVG, T_BMO_AVG),
                    k_core,
                    p,
                    sigma_core,
                )
            elif param_name == "sigma_core":
                E[i] = entropy_pp(Q, k_BMO, k_core_fun(p, T_OC_AVG))
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    k_BMO,
                    k_core_fun(p, T_OC_AVG),
                    sigma_BMO,
                    p,
                )
        return E, M

    # Stixrude et al. (2020)
    def lorenz_sweep_S2020(
        param_name,
        k_core=k_core,
    ):
        if param_name == "k_BMO":
            param_sw = k_BMO_sw
        elif param_name == "k_core":
            param_sw = k_core_sw
        elif param_name == "sigma_BMO":
            param_sw = sigma_BMO_sw
        elif param_name == "sigma_core":
            param_sw = sigma_core_sw

        E, M = (np.zeros(len(param_sw), dtype=object) for i in range(2))
        for i, p in enumerate(param_sw):
            (
                rho_BMO_AVG,
                rho_OC_AVG,
                T_BMO_AVG,
                T_OC_AVG,
                P_BMO_AVG,
                P_OC_AVG,
            ) = (np.zeros(len(Q.t)) for i in range(6))

            for j in range(len(Q.t)):
                rho_BMO_AVG[j] = rho_BMO_AVG_fun(Q.S[j])
                rho_OC_AVG[j] = rho_OC_AVG_fun(Q.S[j])
                T_BMO_AVG[j] = T_BMO_AVG_fun(Q.S[j])
                T_OC_AVG[j] = T_OC_AVG_fun(Q.S[j])
                P_BMO_AVG[j] = P_BMO_AVG_fun(Q.S[j])
                P_OC_AVG[j] = P_OC_AVG_fun(Q.S[j])

            if param_name == "k_BMO":
                E[i] = entropy_pp(Q, p, k_core)
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    p,
                    k_core,
                    sigma_BMO_fun(
                        p,
                        rho_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    sigma_core_fun(
                        p,
                        T_OC_AVG,
                    ),
                )
            elif param_name == "k_core":
                E[i] = entropy_pp(
                    Q,
                    k_BMO_fun(
                        sigma_BMO_S2020_fun(P_BMO_AVG, T_BMO_AVG),
                        rho_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    p,
                )
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    k_BMO_fun(
                        sigma_BMO_S2020_fun(P_BMO_AVG, T_BMO_AVG),
                        rho_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    p,
                    sigma_BMO_S2020_fun(
                        P_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    sigma_core_fun(
                        p,
                        T_OC_AVG,
                    ),
                )
            elif param_name == "sigma_BMO":
                E[i] = entropy_pp(
                    Q,
                    k_BMO_fun(
                        p,
                        rho_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    k_core,
                )
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    k_BMO_fun(
                        p,
                        rho_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    k_core,
                    p,
                    sigma_core_fun(
                        p,
                        T_OC_AVG,
                    ),
                )
            elif param_name == "sigma_core":
                E[i] = entropy_pp(
                    Q,
                    k_BMO_fun(
                        sigma_BMO_S2020_fun(P_BMO_AVG, T_BMO_AVG),
                        rho_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    k_core_fun(
                        p,
                        T_OC_AVG,
                    ),
                )
                M[i] = magnetic_pp(
                    Q,
                    E[i],
                    k_BMO_fun(
                        sigma_BMO_S2020_fun(P_BMO_AVG, T_BMO_AVG),
                        rho_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    k_core_fun(
                        p,
                        T_OC_AVG,
                    ),
                    sigma_BMO_S2020_fun(
                        P_BMO_AVG,
                        T_BMO_AVG,
                    ),
                    p,
                )
        return E, M

    # Entropy outputs
    # =============================================================================
    # Nominal
    # =============================================================================
    E = entropy_pp(Q, k_BMO, k_core)
    entropy_nominal = E

    # Stixrude et al. (2020)
    # =============================================================================
    P_BMO_AVG, T_BMO_AVG, rho_BMO_AVG = [np.zeros(len(Q.t)) for _ in range(3)]

    for j in range(len(Q.t)):
        P_BMO_AVG[j] = P_BMO_AVG_fun(Q.S[j])
        T_BMO_AVG[j] = T_BMO_AVG_fun(Q.S[j])
        rho_BMO_AVG[j] = rho_BMO_AVG_fun(Q.S[j])

    E = entropy_pp(
        Q,
        k_BMO_fun(sigma_BMO_S2020_fun(P_BMO_AVG, T_BMO_AVG), rho_BMO_AVG, T_BMO_AVG),
        k_core,
    )
    entropy_S2020 = E

    # Magnetic field outputs
    # =============================================================================
    # Nominal
    # =============================================================================
    M = magnetic_pp(Q, entropy_nominal, k_BMO, k_core, sigma_BMO, sigma_core)
    magnetic_nominal = M

    # Stixrude et al. (2020)
    # =============================================================================
    P_BMO_AVG, T_BMO_AVG, rho_BMO_AVG, T_OC_AVG = [np.zeros(len(Q.t)) for _ in range(4)]

    for j in range(len(Q.t)):
        P_BMO_AVG[j] = P_BMO_AVG_fun(Q.S[j])
        T_BMO_AVG[j] = T_BMO_AVG_fun(Q.S[j])
        rho_BMO_AVG[j] = rho_BMO_AVG_fun(Q.S[j])
        T_OC_AVG[j] = T_OC_AVG_fun(Q.S[j])

    M = magnetic_pp(
        Q,
        entropy_S2020,
        k_BMO_fun(sigma_BMO_S2020_fun(P_BMO_AVG, T_BMO_AVG), rho_BMO_AVG, T_BMO_AVG),
        k_core,
        sigma_BMO_S2020_fun(P_BMO_AVG, T_BMO_AVG),
        sigma_core_fun(k_core, T_OC_AVG),
    )
    magnetic_S2020 = M

    # Lorenz number outputs
    # =============================================================================
    # Thermal conductivity
    # =============================================================================
    # Nominal
    # =============================================================================
    E, M = lorenz_sweep("k_BMO")
    entropy_k_BMO_Lo = E
    magnetic_k_BMO_Lo = M

    (
        E_phi_BMO,
        Q_conv_BMO,
        Rm_S_BMO,
        B_S_BMO,
        BS_S_BMO,
        BS_S_BMO_cst,
        k_BMO_sw_Lo,
        sigma_BMO_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_BMO[i] = E[i].E_phi_BMO
        Q_conv_BMO[i] = M[i].Q_conv_BMO
        Rm_S_BMO[i] = M[i].Rm_S_BMO
        B_S_BMO[i] = M[i].B_S_BMO
        BS_S_BMO[i] = M[i].BS_S_BMO
        BS_S_BMO_cst[i] = M[i].BS_S_BMO_cst
        k_BMO_sw_Lo[i] = M[i].k_BMO
        sigma_BMO_sw_Lo[i] = M[i].sigma_BMO

    E, M = lorenz_sweep("k_core")
    entropy_k_core_Lo = E
    magnetic_k_core_Lo = M

    (
        E_phi_core,
        Q_conv_core,
        Rm_S_core,
        B_S_core,
        BS_S_core,
        BS_S_core_cst,
        k_core_sw_Lo,
        sigma_core_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_core[i] = E[i].E_phi_core
        Q_conv_core[i] = M[i].Q_conv_core
        Rm_S_core[i] = M[i].Rm_S_core
        B_S_core[i] = M[i].B_S_core
        BS_S_core[i] = M[i].BS_S_core
        BS_S_core_cst[i] = M[i].BS_S_core_cst
        k_core_sw_Lo[i] = M[i].k_core
        sigma_core_sw_Lo[i] = M[i].sigma_core

    # Stixrude et al. (2020)
    # =============================================================================
    E, M = lorenz_sweep_S2020("k_BMO")
    entropy_k_BMO_Lo_S2020 = E
    magnetic_k_BMO_Lo_S2020 = M

    (
        E_phi_BMO,
        Q_conv_BMO,
        Rm_S_BMO,
        B_S_BMO,
        BS_S_BMO,
        BS_S_BMO_cst,
        k_BMO_sw_Lo,
        sigma_BMO_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_BMO[i] = E[i].E_phi_BMO
        Q_conv_BMO[i] = M[i].Q_conv_BMO
        Rm_S_BMO[i] = M[i].Rm_S_BMO
        B_S_BMO[i] = M[i].B_S_BMO
        BS_S_BMO[i] = M[i].BS_S_BMO
        BS_S_BMO_cst[i] = M[i].BS_S_BMO_cst
        k_BMO_sw_Lo[i] = M[i].k_BMO
        sigma_BMO_sw_Lo[i] = M[i].sigma_BMO

    E, M = lorenz_sweep_S2020("k_core")
    entropy_k_core_Lo_S2020 = E
    magnetic_k_core_Lo_S2020 = M

    (
        E_phi_core,
        Q_conv_core,
        Rm_S_core,
        B_S_core,
        BS_S_core,
        BS_S_core_cst,
        k_core_sw_Lo,
        sigma_core_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_core[i] = E[i].E_phi_core
        Q_conv_core[i] = M[i].Q_conv_core
        Rm_S_core[i] = M[i].Rm_S_core
        B_S_core[i] = M[i].B_S_core
        BS_S_core[i] = M[i].BS_S_core
        BS_S_core_cst[i] = M[i].BS_S_core_cst
        k_core_sw_Lo[i] = M[i].k_core
        sigma_core_sw_Lo[i] = M[i].sigma_core

    # Electrical conductivity
    # =============================================================================
    # Nominal
    # =============================================================================
    E, M = lorenz_sweep("sigma_BMO")
    entropy_sigma_BMO_Lo = E
    magnetic_sigma_BMO_Lo = M

    (
        E_phi_BMO,
        Q_conv_BMO,
        Rm_S_BMO,
        B_S_BMO,
        BS_S_BMO,
        BS_S_BMO_cst,
        k_BMO_sw_Lo,
        sigma_BMO_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_BMO[i] = E[i].E_phi_BMO
        Q_conv_BMO[i] = M[i].Q_conv_BMO
        Rm_S_BMO[i] = M[i].Rm_S_BMO
        B_S_BMO[i] = M[i].B_S_BMO
        BS_S_BMO[i] = M[i].BS_S_BMO
        BS_S_BMO_cst[i] = M[i].BS_S_BMO_cst
        k_BMO_sw_Lo[i] = M[i].k_BMO
        sigma_BMO_sw_Lo[i] = M[i].sigma_BMO

    E, M = lorenz_sweep("sigma_core")
    entropy_sigma_core_Lo = E
    magnetic_sigma_core_Lo = M

    (
        E_phi_core,
        Q_conv_core,
        Rm_S_core,
        B_S_core,
        BS_S_core,
        BS_S_core_cst,
        k_core_sw_Lo,
        sigma_core_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_core[i] = E[i].E_phi_core
        Q_conv_core[i] = M[i].Q_conv_core
        Rm_S_core[i] = M[i].Rm_S_core
        B_S_core[i] = M[i].B_S_core
        BS_S_core[i] = M[i].BS_S_core
        BS_S_core_cst[i] = M[i].BS_S_core_cst
        k_core_sw_Lo[i] = M[i].k_core
        sigma_core_sw_Lo[i] = M[i].sigma_core

    # Stixrude et al. (2020)
    # =============================================================================
    E, M = lorenz_sweep_S2020("sigma_BMO")
    entropy_sigma_BMO_Lo_S2020 = E
    magnetic_sigma_BMO_Lo_S2020 = M

    (
        E_phi_BMO,
        Q_conv_BMO,
        Rm_S_BMO,
        B_S_BMO,
        BS_S_BMO,
        BS_S_BMO_cst,
        k_BMO_sw_Lo,
        sigma_BMO_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_BMO[i] = E[i].E_phi_BMO
        Q_conv_BMO[i] = M[i].Q_conv_BMO
        Rm_S_BMO[i] = M[i].Rm_S_BMO
        B_S_BMO[i] = M[i].B_S_BMO
        BS_S_BMO[i] = M[i].BS_S_BMO
        BS_S_BMO_cst[i] = M[i].BS_S_BMO_cst
        k_BMO_sw_Lo[i] = M[i].k_BMO
        sigma_BMO_sw_Lo[i] = M[i].sigma_BMO

    E, M = lorenz_sweep_S2020("sigma_core")
    entropy_sigma_core_Lo_S2020 = E
    magnetic_sigma_core_Lo_S2020 = M

    (
        E_phi_core,
        Q_conv_core,
        Rm_S_core,
        B_S_core,
        BS_S_core,
        BS_S_core_cst,
        k_core_sw_Lo,
        sigma_core_sw_Lo,
    ) = (np.zeros((len(M), len(Q.t))) for i in range(12))
    for i in range(len(M)):
        E_phi_core[i] = E[i].E_phi_core
        Q_conv_core[i] = M[i].Q_conv_core
        Rm_S_core[i] = M[i].Rm_S_core
        B_S_core[i] = M[i].B_S_core
        BS_S_core[i] = M[i].BS_S_core
        BS_S_core_cst[i] = M[i].BS_S_core_cst
        k_core_sw_Lo[i] = M[i].k_core
        sigma_core_sw_Lo[i] = M[i].sigma_core

    # Output
    # =============================================================================
    out = np.zeros(len(name_output), dtype=object)
    for i, n in enumerate(name_output):
        out[i] = locals()[n]

    return out


# %% MPI Execution
# =============================================================================
# Main function
def fun_MPI(path):
    try:
        out = dynamo_sw(path)

        out_MPI = np.zeros(len(name_output), dtype=object)
        for i, n in enumerate(name_output):
            if isinstance(out[i], np.ndarray):
                out_dict = dict()
                for j in range(len(out[i])):
                    out_dict[str(j)] = out[i][j].__dict__
                out_MPI[i] = out_dict
            else:
                out_MPI[i] = out[i].__dict__

    except Exception as e:
        print("Error in %s\n %s" % (path, tb.format_exc()))
        out_MPI = np.zeros(len(name_output), dtype=object)
        for i, n in enumerate(name_output):
            out_MPI[i] = dict()
    return out_MPI


if __name__ == "__main__":
    tic = time.time()

    # Conversion class
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    # Parameters
    class Output:
        def __init__(self):
            pass

    param = Output()
    for i, n in enumerate(name_param):
        setattr(param, n, locals()[n])

    # Parallel pool
    comm = MPI.COMM_WORLD
    rank = comm.Get_size()

    print("Starting parallel pool on %d cores..." % rank)

    with MPIPoolExecutor() as pool:
        for i, result in enumerate(pool.map(fun_MPI, dirs)):
            for r in range(len(result)):
                # Parameters
                with open(
                    dirs[i] + "/param.pkl",
                    "wb",
                ) as f:
                    pkl.dump(Struct(**param.__dict__), f)
                # Results
                with open(
                    dirs[i] + "/{:s}.pkl".format(name_output[r]),
                    "wb",
                ) as f:
                    pkl.dump(Struct(**result[r]), f)

    print("Done.")

    toc = time.time() - tic
    print("Elapsed time: %.1f s" % toc)
