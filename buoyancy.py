#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Victor Lherm

All rights reserved.

This code is provided for academic and research purposes only. Any use of this code in published work or presentations must include a citation to the original author. Unauthorized copying, modification, or distribution of this code is prohibited without prior written permission from the author.

Computes the buoyancy evolution of a planet based on a thermal evolution obtained from thermal.py.

Created by Victor Lherm on 2025-05-07 10:00:00.
"""

# %% Modules
# =============================================================================
import os, glob, time
import numpy as np
import dill as pkl
import importlib as imp
import traceback as tb

from scipy import integrate
from scipy.constants import G, mu_0, R, k, N_A, eV

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

# %% Paths
# =============================================================================
dirs = sorted(glob.glob("path_to_models/*", recursive=False))

# %% Parameters
# =============================================================================
# Numerical flags
verbose = True  # Verbose flag
save_data = True  # Save data flag

# Universal parameters
Lorenz = np.pi**2 / 3 * (k / eV) ** 2

# Basal magma ocean
k_BMO = 8  # Thermal conductivity [W/m/K]

# Core
k_core = 70  # Thermal conductivity [W/m/K]


# Conductivities (Stixrude et al., 2020)
def sigma_BMO_S2020_fun(P, T):
    sigma_0_el, E_s_el, V_s_el = 1.754e9, 108.6e3, 0.0611e-6
    sigma_el = sigma_0_el / T * np.exp(-(E_s_el + P * V_s_el) / (R * T))

    sigma_0_ion, E_s_ion, V_s_ion = 1.0811e9, 131.0e3, 0.437e-6
    sigma_ion = sigma_0_ion / T * np.exp(-(E_s_ion + P * V_s_ion) / (R * T))

    sigma = sigma_el + sigma_ion
    return sigma


def k_BMO_fun(sigma, rho, T):
    k_BMO_0 = 1.16
    rho_0_k_BMO = 2580.7
    T_0_k_BMO = 3000
    g_k_BMO = 1.75
    a_k_BMO = -0.09

    k_ph = k_BMO_0 * (rho / rho_0_k_BMO) ** g_k_BMO * (T_0_k_BMO / T) ** a_k_BMO
    k_el = Lorenz * T * sigma
    k = k_ph + k_el
    return k


# Thermal conductivity sweep
N_sweep = 64
k_BMO_sw = np.linspace(0, 50, N_sweep)
k_core_sw = np.linspace(0, 300, N_sweep)


# %% Function
# =============================================================================
def buoyancy(path):
    print("Processing: %s" % path)
    try:
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
            s2y,
            h_mantle,
            h_core,
            Q_CMB_fun,
            Q_ICB_fun,
            CR_BMO,
            CXI_BMO,
            CR_core,
            CXI_core,
            P_G_BMO,
            P_G_core,
            P_L_BMO,
            P_L_core,
            D_BMO_fun,
            D_BMO_0,
            Drho_BMO,
            Drho_ICB,
            t_earth,
        )

        # Load data
        with open(simu_path + "/thermal.pkl", "rb") as f:
            Q = pkl.load(f)

        # Codensity profile
        # =============================================================================
        # Basal Magma Ocean
        # =============================================================================
        # Averages
        def rho_BMO_AVG_fun(S):
            rho_BMO_AVG = integrate.trapezoid(
                S.radius_BMO_r_0**2 * S.rho_BMO_r_0_i,
                S.radius_BMO_r_0,
            ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
            return rho_BMO_AVG

        def Cp_BMO_AVG_fun(S):
            Cp_BMO_AVG = integrate.trapezoid(
                S.radius_BMO_r_0**2 * S.Cp_BMO_r_0_i,
                S.radius_BMO_r_0,
            ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
            return Cp_BMO_AVG

        def alpha_BMO_AVG_fun(S):
            alpha_BMO_AVG = integrate.trapezoid(
                S.radius_BMO_r_0**2 * S.alpha_BMO_r_0_i,
                S.radius_BMO_r_0,
            ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
            return alpha_BMO_AVG

        def P_BMO_AVG_fun(S):
            P_BMO_AVG = integrate.trapezoid(
                S.radius_BMO_r_0**2 * S.P_BMO_r_0_i,
                S.radius_BMO_r_0,
            ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
            return P_BMO_AVG

        def T_BMO_AVG_fun(S):
            T_BMO_AVG = integrate.trapezoid(
                S.radius_BMO_r_0**2 * S.Ta_BMO_r_0,
                S.radius_BMO_r_0,
            ) / integrate.trapezoid(S.radius_BMO_r_0**2, S.radius_BMO_r_0)
            return T_BMO_AVG

        # BMO secular cooling (dC0dr_S_BMO)
        def dC0dr_S_BMO_fun(S, Xi_BMO, dT_CMBdt):
            I = integrate.cumulative_trapezoid(
                -4
                * np.pi
                * S.radius_BMO_r_0**2
                * S.rho_BMO_r_0_i
                * S.Cp_BMO_r_0_i
                * (S.Ta_BMO_r_0 / S.T_CMB)
                * dT_CMBdt,
                S.radius_BMO_r_0,
                initial=0,
            )

            dC0dr = (
                -1
                / (4 * np.pi)
                / S.radius_BMO_r_0**2
                * (alpha_BMO_AVG_fun(S) / Cp_BMO_AVG_fun(S))
                * I
            )
            return dC0dr

        # BMO latent heat (dC0dr_L_BMO)
        def dC0dr_L_BMO_fun(S, Xi_BMO, dT_CMBdt):
            dC0dr = np.zeros((len(S.radius_BMO_r_0)))
            return dC0dr

        # BMO radioactive heating (dC0dr_R_BMO)
        def dC0dr_R_BMO_fun(S, t):
            I = integrate.cumulative_trapezoid(
                4
                * np.pi
                * S.radius_BMO_r_0**2
                * S.rho_BMO_r_0_i
                * h_mantle(t, t_earth),
                S.radius_BMO_r_0,
                initial=0,
            )

            dC0dr = (
                -1
                / (4 * np.pi)
                / S.radius_BMO_r_0**2
                * (alpha_BMO_AVG_fun(S) / Cp_BMO_AVG_fun(S))
                * I
            )
            return dC0dr

        # BMO composition (dC0dr_C_BMO)
        def dC0dr_C_BMO_fun(S, Xi_BMO, dT_CMBdt):
            D_BMO = D_BMO_fun(Xi_BMO, D_BMO_0)
            DXi_BMO = Xi_BMO * (1 - D_BMO)
            alpha_c_BMO = Drho_BMO / (S.rho_mantle[0] * DXi_BMO)
            I = integrate.cumulative_trapezoid(
                4
                * np.pi
                * S.radius_BMO_r_0**2
                * rho_BMO_AVG_fun(S)
                * alpha_c_BMO
                * CR_BMO(S, Xi_BMO)
                * CXI_BMO(S, Xi_BMO)
                * dT_CMBdt,
                S.radius_BMO_r_0,
                initial=0,
            )

            dC0dr = -1 / (4 * np.pi) / S.radius_BMO_r**2 * I
            return dC0dr

        # BMO dissipation (dC0dr_PHI_BMO)
        def dC0dr_G_BMO_fun(S, Xi_BMO, dT_CMBdt):
            V_BMO = integrate.trapezoid(
                4 * np.pi * S.radius_BMO_r_0**2, S.radius_BMO_r_0
            )
            I = integrate.cumulative_trapezoid(
                4 * np.pi * S.radius_BMO_r_0**2 * P_G_BMO(S, Xi_BMO) * dT_CMBdt / V_BMO,
                S.radius_BMO_r_0,
                initial=0,
            )

            dC0dr = (
                -1
                / (4 * np.pi)
                / S.radius_BMO_r_0**2
                * (alpha_BMO_AVG_fun(S) / Cp_BMO_AVG_fun(S))
                * I
            )
            return dC0dr

        # CMB heat flow (dC0dr_CMB)
        def dC0dr_CMB_fun(S, Xi_core, dT_CMBdt, t):
            dC0dr = (
                -1
                / (4 * np.pi)
                / S.radius_BMO_r_0**2
                * (alpha_BMO_AVG_fun(S) / Cp_BMO_AVG_fun(S))
                * Q_CMB_fun(S, Xi_core, dT_CMBdt, t)
            )
            return dC0dr

        # BMO adiabatic heat flow (dC0dr_A_BMO)
        def dC0dr_A_BMO_fun(S, k_BMO):
            kappa = k_BMO / (rho_BMO_AVG_fun(S) * Cp_BMO_AVG_fun(S))
            dC0dr = (
                -rho_BMO_AVG_fun(S)
                * alpha_BMO_AVG_fun(S)
                * kappa
                * np.gradient(S.Ta_BMO_r_0, S.radius_BMO_r_0)
            )
            return dC0dr

        # Total codensity gradient (dC0dr_TOT_BMO)
        def dC0dr_TOT_BMO_fun(S, Xi_BMO, dT_CMBdt, t, k_BMO):
            dC0dr = np.nansum(
                [
                    dC0dr_R_BMO_fun(S, t),
                    dC0dr_S_BMO_fun(S, Xi_BMO, dT_CMBdt),
                    dC0dr_L_BMO_fun(S, Xi_BMO, dT_CMBdt),
                    dC0dr_C_BMO_fun(S, Xi_BMO, dT_CMBdt),
                    dC0dr_G_BMO_fun(S, Xi_BMO, dT_CMBdt),
                    dC0dr_CMB_fun(S, Xi_BMO, dT_CMBdt, t),
                    dC0dr_A_BMO_fun(S, k_BMO),
                ],
                axis=0,
            )
            return dC0dr

        # Core
        # =============================================================================
        # Averages
        def rho_core_AVG(S):
            rho_core_AVG = integrate.trapezoid(
                S.radius_OC**2 * S.rho_OC,
                S.radius_OC,
            ) / integrate.trapezoid(S.radius_OC**2, S.radius_OC)
            return rho_core_AVG

        def Cp_core_AVG(S):
            Cp_core_AVG = integrate.trapezoid(
                S.radius_OC**2 * S.Cp_OC,
                S.radius_OC,
            ) / integrate.trapezoid(S.radius_OC**2, S.radius_OC)
            return Cp_core_AVG

        def alpha_core_AVG(S):
            alpha_core_AVG = integrate.trapezoid(
                S.radius_OC**2 * S.alpha_OC,
                S.radius_OC,
            ) / integrate.trapezoid(S.radius_OC**2, S.radius_OC)
            return alpha_core_AVG

        # Core secular cooling (dC0dr_S_core)
        def dC0dr_S_core_fun(S, Xi_core, dT_CMBdt):
            radius_OC = np.where(S.radius_OC == 0, np.nan, S.radius_OC)

            I = integrate.cumulative_trapezoid(
                -4
                * np.pi
                * S.radius_OC**2
                * S.rho_OC
                * S.Cp_OC
                * (S.T_OC / S.T_CMB)
                * dT_CMBdt,
                S.radius_OC,
                initial=0,
            )

            dC0dr = (
                -1
                / (4 * np.pi)
                / radius_OC**2
                * (alpha_core_AVG(S) / Cp_core_AVG(S))
                * I
            )
            return dC0dr

        # Core latent heat (dC0dr_L_core)
        def dC0dr_L_core_fun(S, Xi_core, dT_CMBdt):
            radius_OC = np.where(S.radius_OC == 0, np.nan, S.radius_OC)

            dC0dr = (
                -1
                / (4 * np.pi)
                / radius_OC**2
                * (alpha_core_AVG(S) / Cp_core_AVG(S))
                * P_L_core(S, Xi_core)
                * dT_CMBdt
            )
            return dC0dr

        # Core radioactive heating (dC0dr_R_core)
        def dC0dr_R_core_fun(S, t):
            radius_OC = np.where(S.radius_OC == 0, np.nan, S.radius_OC)

            I = integrate.cumulative_trapezoid(
                4 * np.pi * S.radius_OC**2 * S.rho_OC * h_core(t, t_earth),
                S.radius_OC,
                initial=0,
            )

            dC0dr = (
                -1
                / (4 * np.pi)
                / radius_OC**2
                * (alpha_core_AVG(S) / Cp_core_AVG(S))
                * I
            )
            return dC0dr

        # Core composition (dC0dr_C_core)
        def dC0dr_C_core_fun(S, Xi_core, dT_CMBdt):
            radius_OC = np.where(S.radius_OC == 0, np.nan, S.radius_OC)

            DXi_core = Xi_core
            alpha_c_core = Drho_ICB / (S.rho_ICB_bot * DXi_core)
            I = integrate.cumulative_trapezoid(
                4
                * np.pi
                * S.radius_OC**2
                * rho_core_AVG(S)
                * alpha_c_core
                * CR_core(S, Xi_core)
                * CXI_core(S, Xi_core)
                * dT_CMBdt,
                S.radius_OC,
                initial=0,
            )

            dC0dr = 1 / (4 * np.pi) / radius_OC**2 * (I - I[-1])
            return dC0dr

        # Core dissipation (dC0dr_PHI_core)
        def dC0dr_G_core_fun(S, Xi_core, dT_CMBdt):
            radius_OC = np.where(S.radius_OC == 0, np.nan, S.radius_OC)

            V_OC = 4 / 3 * np.pi * (S.R_CMB**3 - S.R_ICB**3)
            I = integrate.cumulative_trapezoid(
                4 * np.pi * S.radius_OC**2 * P_G_core(S, Xi_core) * dT_CMBdt / V_OC,
                S.radius_OC,
                initial=0,
            )

            dC0dr = (
                -1
                / (4 * np.pi)
                / radius_OC**2
                * (alpha_core_AVG(S) / Cp_core_AVG(S))
                * I
            )
            return dC0dr

        # Inner core heat flow (dC0dr_ICB)
        def dC0dr_ICB_fun(S, Xi_core, dT_CMBdt, t):
            radius_OC = np.where(S.radius_OC == 0, np.nan, S.radius_OC)

            dC0dr = (
                -1
                / (4 * np.pi)
                / radius_OC**2
                * (alpha_core_AVG(S) / Cp_core_AVG(S))
                * Q_ICB_fun(S, Xi_core, dT_CMBdt, t)
            )
            return dC0dr

        # Core adiabatic heat flow (dC0dr_A_core)
        def dC0dr_A_core_fun(S, k_core):
            kappa = k_core / (rho_core_AVG(S) * Cp_core_AVG(S))
            dC0dr = (
                -rho_core_AVG(S)
                * alpha_core_AVG(S)
                * kappa
                * np.gradient(S.T_OC, S.radius_OC)
            )
            dC0dr = np.where(S.radius_OC == 0, np.nan, dC0dr)
            return dC0dr

        # Total codensity gradient (dC0dr_TOT_core)
        def dC0dr_TOT_core_fun(S, Xi_core, dT_CMBdt, t, k_core):
            dC0dr = np.nansum(
                [
                    dC0dr_R_core_fun(S, t),
                    dC0dr_S_core_fun(S, Xi_core, dT_CMBdt),
                    dC0dr_L_core_fun(S, Xi_core, dT_CMBdt),
                    dC0dr_C_core_fun(S, Xi_core, dT_CMBdt),
                    dC0dr_G_core_fun(S, Xi_core, dT_CMBdt),
                    dC0dr_ICB_fun(S, Xi_core, dT_CMBdt, t),
                    dC0dr_A_core_fun(S, k_core),
                ],
                axis=0,
            )
            return dC0dr

        # Post-processing
        # =============================================================================
        # BMO
        radius_BMO_r = np.zeros((len(Q.t), N))

        (
            dC0dr_R_BMO,
            dC0dr_S_BMO,
            dC0dr_L_BMO,
            dC0dr_C_BMO,
            dC0dr_G_BMO,
            dC0dr_CMB,
            dC0dr_A_BMO,
        ) = [np.zeros((len(Q.t), N)) for i in range(7)]

        dC0dr_TOT_BMO = np.zeros((len(Q.t), N))

        sigma_BMO_S2020, k_BMO_S2020 = [np.zeros(len(Q.t)) for _ in range(2)]
        dC0dr_A_BMO_S2020 = np.zeros((len(Q.t), N))
        dC0dr_TOT_BMO_S2020 = np.zeros((len(Q.t), N))

        for i in range(len(Q.t)):
            radius_BMO_r[i] = Q.S[i].radius_BMO_r

            dC0dr_R_BMO[i] = dC0dr_R_BMO_fun(Q.S[i], Q.t[i])
            dC0dr_S_BMO[i] = dC0dr_S_BMO_fun(Q.S[i], Q.Xi_BMO, Q.dT_CMBdt[i])
            dC0dr_L_BMO[i] = dC0dr_L_BMO_fun(Q.S[i], Q.Xi_BMO[i], Q.dT_CMBdt[i])
            dC0dr_C_BMO[i] = dC0dr_C_BMO_fun(Q.S[i], Q.Xi_BMO[i], Q.dT_CMBdt[i])
            dC0dr_G_BMO[i] = dC0dr_G_BMO_fun(Q.S[i], Q.Xi_BMO[i], Q.dT_CMBdt[i])
            dC0dr_CMB[i] = dC0dr_CMB_fun(Q.S[i], Q.Xi_BMO[i], Q.dT_CMBdt[i], Q.t[i])
            dC0dr_A_BMO[i] = dC0dr_A_BMO_fun(Q.S[i], k_BMO)
            dC0dr_TOT_BMO[i] = dC0dr_TOT_BMO_fun(
                Q.S[i],
                Q.Xi_BMO[i],
                Q.dT_CMBdt[i],
                Q.t[i],
                k_BMO,
            )

            sigma_BMO_S2020[i] = sigma_BMO_S2020_fun(
                P_BMO_AVG_fun(Q.S[i]),
                T_BMO_AVG_fun(Q.S[i]),
            )
            k_BMO_S2020[i] = k_BMO_fun(
                sigma_BMO_S2020[i],
                rho_BMO_AVG_fun(Q.S[i]),
                T_BMO_AVG_fun(Q.S[i]),
            )

            dC0dr_A_BMO_S2020[i] = dC0dr_A_BMO_fun(
                Q.S[i],
                k_BMO_S2020[i],
            )
            dC0dr_TOT_BMO_S2020[i] = dC0dr_TOT_BMO_fun(
                Q.S[i],
                Q.Xi_BMO[i],
                Q.dT_CMBdt[i],
                Q.t[i],
                k_BMO_S2020[i],
            )

        mask_dC0dr_TOT_BMO = np.zeros((len(Q.t), N_sweep))
        for j in range(N_sweep):
            for i in range(len(Q.t)):
                dC0dr_TOT_BMO_sw = dC0dr_TOT_BMO_fun(
                    Q.S[i],
                    Q.Xi_BMO[i],
                    Q.dT_CMBdt[i],
                    Q.t[i],
                    k_BMO_sw[j],
                )
                mask_dC0dr_TOT_BMO[i, j] = np.sum(dC0dr_TOT_BMO_sw < 0) / N

        # Core
        radius_core_r = np.zeros((len(Q.t), N))

        (
            dC0dr_R_core,
            dC0dr_S_core,
            dC0dr_L_core,
            dC0dr_C_core,
            dC0dr_G_core,
            dC0dr_ICB,
            dC0dr_A_core,
        ) = [np.zeros((len(Q.t), N)) for i in range(7)]

        dC0dr_TOT_core = np.zeros((len(Q.t), N))

        for i in range(len(Q.t)):
            radius_core_r[i] = Q.S[i].radius_OC

            dC0dr_R_core[i] = dC0dr_R_core_fun(Q.S[i], Q.t[i])
            dC0dr_S_core[i] = dC0dr_S_core_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i])
            dC0dr_L_core[i] = dC0dr_L_core_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i])
            dC0dr_C_core[i] = dC0dr_C_core_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i])
            dC0dr_G_core[i] = dC0dr_G_core_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i])
            dC0dr_ICB[i] = dC0dr_ICB_fun(Q.S[i], Q.Xi_core[i], Q.dT_CMBdt[i], Q.t[i])
            dC0dr_A_core[i] = dC0dr_A_core_fun(Q.S[i], k_core)
            dC0dr_TOT_core[i] = dC0dr_TOT_core_fun(
                Q.S[i],
                Q.Xi_core[i],
                Q.dT_CMBdt[i],
                Q.t[i],
                k_core,
            )

        mask_dC0dr_TOT_core = np.zeros((len(Q.t), N_sweep))
        for j in range(N_sweep):
            for i in range(len(Q.t)):
                dC0dr_TOT_core_sw = dC0dr_TOT_core_fun(
                    Q.S[i],
                    Q.Xi_core[i],
                    Q.dT_CMBdt[i],
                    Q.t[i],
                    k_core_sw[j],
                )
                mask_dC0dr_TOT_core[i, j] = np.sum(dC0dr_TOT_core_sw < 0) / N

        # Save data
        # =============================================================================
        class Output:
            def __init__(self):
                self.t = Q.t

                self.k_BMO = k_BMO
                self.k_BMO_sw = k_BMO_sw

                self.radius_BMO_r = radius_BMO_r
                self.dC0dr_R_BMO = dC0dr_R_BMO
                self.dC0dr_S_BMO = dC0dr_S_BMO
                self.dC0dr_L_BMO = dC0dr_L_BMO
                self.dC0dr_C_BMO = dC0dr_C_BMO
                self.dC0dr_G_BMO = dC0dr_G_BMO
                self.dC0dr_CMB = dC0dr_CMB
                self.dC0dr_A_BMO = dC0dr_A_BMO
                self.dC0dr_TOT_BMO = dC0dr_TOT_BMO
                self.mask_dC0dr_TOT_BMO = mask_dC0dr_TOT_BMO

                self.sigma_BMO_S2020 = sigma_BMO_S2020
                self.k_BMO_S2020 = k_BMO_S2020

                self.dC0dr_A_BMO_S2020 = dC0dr_A_BMO_S2020
                self.dC0dr_TOT_BMO_S2020 = dC0dr_TOT_BMO_S2020

                self.k_core = k_core
                self.k_core_sw = k_core_sw

                self.radius_core_r = radius_core_r
                self.dC0dr_R_core = dC0dr_R_core
                self.dC0dr_S_core = dC0dr_S_core
                self.dC0dr_L_core = dC0dr_L_core
                self.dC0dr_C_core = dC0dr_C_core
                self.dC0dr_G_core = dC0dr_G_core
                self.dC0dr_ICB = dC0dr_ICB
                self.dC0dr_A_core = dC0dr_A_core
                self.dC0dr_TOT_core = dC0dr_TOT_core
                self.mask_dC0dr_TOT_core = mask_dC0dr_TOT_core

        B = Output()

    except Exception as e:
        print("Error in %s\n %s" % (path, tb.format_exc()))

        class Output:
            def __init__(self):
                pass

        B = Output()

    return B


# %% MPI Execution
# =============================================================================
# Main function
def fun_MPI(path):
    S = buoyancy(path)
    return S.__dict__


if __name__ == "__main__":
    tic = time.time()

    # Conversion class
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    # Parallel pool
    comm = MPI.COMM_WORLD
    rank = comm.Get_size()

    print("Starting parallel pool on %d cores..." % rank)

    with MPIPoolExecutor() as pool:
        S = np.zeros(len(dirs), dtype=object)
        for i, result in enumerate(pool.map(fun_MPI, dirs)):
            with open(dirs[i] + "/buoyancy.pkl", "wb") as f:
                pkl.dump(Struct(**result), f)

    print("Done.")

    toc = time.time() - tic
    print("Elapsed time: %.1f s" % toc)
