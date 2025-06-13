#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Victor Lherm

All rights reserved.

This code is provided for academic and research purposes only. Any use of this code in published work or presentations must include a citation to the original author. Unauthorized copying, modification, or distribution of this code is prohibited without prior written permission from the author.

Contains supporting functions required to compute the thermal evolution.

Created by Victor Lherm on 2025-05-07 10:00:00.
"""

# %% Modules
# =============================================================================
import glob
import numpy as np
import dill as pkl

from scipy import integrate
from scipy.constants import G, mu_0, R, k, N_A, eV


# %% Time functions
# =============================================================================
def y2s(years):
    seconds = years * 31556952
    return seconds


def s2y(seconds):
    years = seconds / 31556952
    return years


# Molar & Mass fractions
# =============================================================================
def wt2mol(w1, M1, M2):
    x1 = w1 / (w1 + (1 - w1) * M1 / M2)
    return x1


def mol2wt(x1, M1, M2):
    w1 = x1 / (x1 + (1 - x1) * M2 / M1)
    return w1


# %% General parameters
# =============================================================================
# Paths
energy_path = "Demo"  # Simulation path
structure_path = energy_path

# Flags
verbose = True  # Verbose flag
save_data = True  # Save data flag

# Numerical parameters
N = 1024  # Number of grid points
zero = 1e-6  # Numerical zero

# Initial T_CMB
ini_type_T_CMB = "adiabat"  # Initial T_CMB type (adiabat, radius or temperature)
T_CMB_0 = 0  # Initial T_CMB value (if ini_type_T_CMB = temperature) [K]
L_BMO_0 = 0  # Initial BMO thickness (if ini_type_T_CMB = radius) [m]

# Earth
M_earth = 5.972e24  # Earth mass [kg]
CMF_earth = 0.32  # Earth core mass fraction
R_earth = 6371e3  # Earth radius [m]
t_earth = y2s(4.5e9)  # Earth age [s]

# Planet
M_planet = M_earth  # Planet mass [kg]
CMF = 0.32  # Core mass fraction

T_surface = 300  # Surface temperature [K]
P_surface = 1.01325e5  # Surface pressure [Pa]

t_final = 10e9  # Final time [yr]

# Mantle
c_U238_mantle = 20e-9  # Uranium-238 concentration [ppb]
c_U235_mantle = 20e-9  # Uranium-235 concentration [ppb]
c_Th232_mantle = 80e-9  # Thorium-232 concentration [ppb]
c_K40_mantle = 240e-6  # Potassium-40 concentration [ppb]

f_R_mantle = 0.58  # Mantle fraction of the BSE radiogenic heat production

beta_mantle = 1 / 3  # Nu-Ra scaling law exponent
Ra_c = 660  # Critical Rayleigh number

k_0 = 4.7  # Thermal conductivity scale value (phonon transport) [W/m/K]
a_k = 0.3  # Thermal conductivity power-law exponent (phonon transport)
k_trunc = 15  # Thermal conductivity truncation value (phonon transport) [W/m/K]
k_rad = 2  # Thermal conductivity (photon transport) [W/m/K]
T1_rad, T2_rad, T3_rad, T4_rad = (
    573,
    4000,
    4500,
    7000,
)  # Corner temperature taper (photon transport) [K]
E_el_k = 3 * eV  # Activation energy (electron transport) [J]
k_el_p = 487.54  # Prefactor (electron transport) [W/m/K]

eta_earth = 1.1e15  # Reference dynamic viscosity [Pa.s]
A_eta = 3e5  # Viscosity activation energy [J/mol]
psi_eta = 0  # Viscosity exponent (M)
theta_eta = 0  # Viscosity exponent (CMF)
f_eta = 10  # Viscosity jump from upper to mid-mantle

# Basal magma ocean
Drho_BMO = 250  # Density jump at the BMO [kg/m3]

DS_0_mantle = 300  # Entropy of fusion of the mantle [J/kg/K]

M_FeO = 71.84e-3  # Molar mass of FeO [kg/mol]
M_MgSiO3 = 100.39e-3  # Molar mass of MgSiO3 [kg/mol]

Xi_BMO_initial = 0.16  # Initial FeO mass fraction [wt%]
xi_BMO_initial = wt2mol(Xi_BMO_initial, M_FeO, M_MgSiO3)  # [mol%]

D_BMO_0 = 0.85  # Initial FeO partition coefficient

# Core
Drho_ICB = 580  # Density jump at the ICB [kg/m3]

DS_0_core = 127  # Entropy of fusion of the core [J/kg/K]

Xi_core_initial = 0.056  # Initial light element mass fraction [wt%]

c_K40_core = 20e-6  # Potassium-40 concentration at present time [ppm]

# Radiogenic elements properties
a_U238 = 0.9927  # Uranium-238 natural abundance
a_U235 = 0.0072  # Uranium-235 natural abundance
a_Th232 = 0.9998  # Thorium-232 natural abundance
a_K40 = 0.000117  # Potassium-40 natural abundance
e_U238 = 7.65e-12  # Uranium-238 energy per atom [J]
e_U235 = 7.11e-12  # Uranium-235 energy per atom [J]
e_Th232 = 6.48e-12  # Thorium-232 energy per atom [J]
e_K40 = 0.110e-12  # Potassium-40 energy per atom [J]
t12_U238 = y2s(4.46e9)  # Uranium-238 half-life [s]
t12_U235 = y2s(7.04e8)  # Uranium-235 half-life [s]
t12_Th232 = y2s(1.40e10)  # Thorium-232 half-life [s]
t12_K40 = y2s(1.26e9)  # Potassium-40 half-life [s]
M_U238 = 238.05e-3  # Uranium-238 molar mass [kg/mol]
M_U235 = 235.04e-3  # Uranium-235 molar mass [kg/mol]
M_Th232 = 232.04e-3  # Thorium-232 molar mass [kg/mol]
M_K40 = 39.96e-3  # Potassium-40 molar mass [kg/mol]


# %% Planetary functions
# =============================================================================
# Rotation rate
def Omega_planet(t):
    if isinstance(t, (int, float)):
        t = np.array([t])
    omega = np.zeros(np.size(t))
    for i in range(np.size(t)):
        lod = 5.0 + t[i] * (23.934 - 5.0) / y2s(4.5e9)
        omega[i] = 2 * np.pi / (lod * 3600)
    if len(t) == 1:
        omega = omega[0]
    return omega


# %% Solid mantle structure
# =============================================================================
# Phase transitions
def DS_0_solid(layer):
    if layer == "mantle":
        DS0 = DS_0_mantle
    elif layer == "core":
        DS0 = DS_0_core
    return DS0


# Entropy of fusion
def entropy_fusion(rho, layer):
    if layer == "core":
        DS = DS_0_solid(layer) * np.ones(len(rho))
    elif layer == "mantle":
        DS = DS_0_solid(layer) * np.ones(len(rho))
    return DS


# Gravitationnal potential
def gravi_potential(radius, g, R0):
    r0 = np.linspace(0, R0, N)
    g0 = np.interp(r0, radius, g)
    psi_0 = -integrate.trapezoid(g0, r0)
    psi = psi_0 + integrate.cumulative_trapezoid(g, radius, initial=0)
    return psi


# Transport coefficients
def k_solid(P, T, gamma, alpha, K0, K0_prime):
    k_lat_star = (
        k_0
        * (T_surface / T) ** a_k
        * np.exp(-(4 * gamma + (1 / 3)) * alpha * (T - T_surface))
        * (1 + K0_prime / K0 * P)
    )
    k_lat = (k_lat_star**-2 + k_trunc**-2) ** -0.5
    if isinstance(T, (int, float)):
        k_r = 0
    else:
        k_r = np.zeros(len(T))
    k_r = np.where(
        (T > T1_rad) & (T < T2_rad),
        k_rad / 2 * (np.sin(np.pi * (T - T1_rad) / (T2_rad - T1_rad) - np.pi / 2) + 1),
        k_r,
    )
    k_r = np.where(
        (T > T3_rad) & (T < T4_rad),
        k_rad / 2 * (np.sin(np.pi * (T - T3_rad) / (T4_rad - T3_rad) + np.pi / 2) + 1),
        k_r,
    )
    k_r = np.where((T > T2_rad) & (T < T3_rad), k_rad, k_r)
    k_el = k_el_p * np.exp(-E_el_k / (k * T))
    k_tot = k_lat + k_r + k_el
    return k_tot


def eta_solid(T, M, CMF):
    eta = (
        eta_earth
        * np.exp(A_eta / (R * T))
        * (M / M_earth) ** psi_eta
        * (CMF / CMF_earth) ** theta_eta
    )
    return eta


# %% BMO structure
# =============================================================================
# Thermodynamic parameters
# =============================================================================
# Universal
def n_BMO(phase):
    if phase == "MgSiO3":
        n = 5
    elif phase == "FeO":
        n = 2
    return n


def molar_mass_BMO(phase):
    if phase == "MgSiO3":
        M = M_MgSiO3
    elif phase == "FeO":
        M = M_FeO
    return M


def T0_BMO(phase):
    if phase == "MgSiO3":
        T0 = 3000
    elif phase == "FeO":
        T0 = 1650
    return T0


def U0_BMO(phase):
    if phase == "MgSiO3":
        U0 = -9935.731e3
    elif phase == "FeO":
        U0 = -442.217e3
    return U0


def V0_BMO(phase):
    if phase == "MgSiO3":
        V0 = 38.99e-6
    elif phase == "FeO":
        V0 = 15.97e-6
    return V0


def rho_0_BMO(phase):
    if phase == "MgSiO3":
        rho0 = 2574.7
    elif phase == "FeO":
        rho0 = 4498.7
    return rho0


def K0_BMO(phase):
    if phase == "MgSiO3":
        K0 = 13.20e9
    elif phase == "FeO":
        K0 = 40.72e9
    return K0


def K0_prime_BMO(phase):
    if phase == "MgSiO3":
        K0_prime = 8.238
    elif phase == "FeO":
        K0_prime = 5.33
    return K0_prime


def gamma_0_BMO(phase):
    if phase == "MgSiO3":
        gamma0 = 0.3675
    elif phase == "FeO":
        gamma0 = 2.499
    return gamma0


def gamma_inf_BMO(phase):
    if phase == "MgSiO3":
        gamma_inf = 0.50
    elif phase == "FeO":
        gamma_inf = 0
    return gamma_inf


def beta_BMO(phase):
    if phase == "MgSiO3":
        beta = 1.0
    elif phase == "FeO":
        beta = -0.021
    return beta


# MgSiO3
def rho_0_gamma_BMO(phase):
    if phase == "MgSiO3":
        rho0 = 2743.4
    return rho0


def rho_e_BMO(phase):
    if phase == "MgSiO3":
        rho_e = 5195
    return rho_e


def m_BMO(phase):
    if phase == "MgSiO3":
        m = 0.6
    return m


def b_n_BMO(n, phase):
    if phase == "MgSiO3":
        if n == 0:
            b_n = 473.838e3
        elif n == 1:
            b_n = 296.982e3
        elif n == 2:
            b_n = 633.281e3
        elif n == 3:
            b_n = -1455.481e3
        elif n == 4:
            b_n = -1991.940e3
    return b_n


def gamma_01_BMO(phase):
    if phase == "MgSiO3":
        gamma_01 = 0.65
    return gamma_01


def s_BMO(phase):
    if phase == "MgSiO3":
        s = 1707e3
    return s


def zeta_0_BMO(phase):
    if phase == "MgSiO3":
        zeta_0 = 1.88849e-3
    return zeta_0


def zeta_exp_BMO(phase):
    if phase == "MgSiO3":
        zeta_exp = 0.67774
    return zeta_exp


def T_e_0_BMO(phase):
    if phase == "MgSiO3":
        T_e_0 = 2466.6
    return T_e_0


def T_e_exp_BMO(phase):
    if phase == "MgSiO3":
        T_e_exp = -0.4578
    return T_e_exp


# FeO
def e_0_BMO(phase):
    if phase == "FeO":
        e0 = -49.72e-6
    return e0


def gamma_e_BMO(phase):
    if phase == "FeO":
        gamma_e = 4.96
    return gamma_e


def theta_0_BMO(phase):
    if phase == "FeO":
        theta_0 = 1247.8
    return theta_0


def a_s_BMO(phase):
    if phase == "FeO":
        a_s = 18.81
    return a_s


# Variable definition
# =============================================================================
# Density
def x_BMO(rho, phase):
    x = rho_0_BMO(phase) / rho
    return x


def X_BMO(rho, phase):
    X = (rho_0_BMO(phase) / rho) ** (1 / 3)
    return X


# Eta
def eta_BMO(phase):
    eta = 3 / 2 * (K0_prime_BMO(phase) - 1)
    return eta


# Gruneisen parameter (Altshuler form)
def grun_BMO(rho, phase):
    if phase == "MgSiO3":
        gamma = (
            gamma_inf_BMO(phase)
            + (
                (gamma_0_BMO(phase) - gamma_inf_BMO(phase))
                * (rho_0_gamma_BMO(phase) / rho) ** beta_BMO(phase)
            )
            + (
                gamma_01_BMO(phase)
                * np.exp(-((rho - rho_e_BMO(phase)) ** 2) / s_BMO(phase) ** 2)
            )
        )
    elif phase == "FeO":
        gamma = gamma_inf_BMO(phase) + (
            (gamma_0_BMO(phase) - gamma_inf_BMO(phase))
            * x_BMO(rho, phase) ** beta_BMO(phase)
        )
    return gamma


# MgSiO3 (Wolf et al., 2018 & Fratanduono et al., 2018)
# =============================================================================
# Thermal coefficient
def b_MgSiO3(rho, phase):
    b = 0
    for n in range(5):
        b += b_n_BMO(n, phase) * (x_BMO(rho, phase) - 1) ** n
    return b


# Thermal deviations
def f_T_MgSiO3(T, phase):
    f_T = (T / T0_BMO(phase)) ** m_BMO(phase) - 1
    return f_T


def f_T_prime_MgSiO3(T, phase):
    f_T_prime = m_BMO(phase) / T0_BMO(phase) * (T / T0_BMO(phase)) ** (m_BMO(phase) - 1)
    return f_T_prime


# Reference adiabatic profile
def T0_S_MgSiO3_fit(rho, phase):
    if isinstance(rho, (int, float)):
        rho = np.array([rho])

    rho_0 = rho_0_BMO(phase)

    T0_S = np.zeros(len(rho))
    for i, j in enumerate(rho):
        z = np.linspace(rho_0, j, N)
        T0_S[i] = T0_BMO(phase) * np.exp(integrate.trapezoid(grun_BMO(z, phase) / z, z))

    if len(rho) == 1:
        T0_S = T0_S[0]
    return T0_S


phase = "MgSiO3"
rho_min, rho_max = rho_0_BMO(phase), 12e3
rho_fit = np.linspace(rho_min, rho_max, N)
T0_S_coefs = np.polyfit(rho_fit, T0_S_MgSiO3_fit(rho_fit, phase), 3)


def T0_S_MgSiO3(rho, coefs=T0_S_coefs):
    T0_S = np.polyval(coefs, rho)
    return T0_S


# Free energy
def F_0_MgSiO3(rho, phase):
    F_0 = (
        9
        * K0_BMO(phase)
        * V0_BMO(phase)
        * eta_BMO(phase) ** -2
        * (
            1
            - (1 - eta_BMO(phase) * (1 - X_BMO(rho, phase)))
            * np.exp(eta_BMO(phase) * (1 - X_BMO(rho, phase)))
        )
    )
    return F_0


def DE_MgSiO3(rho, T, phase):
    DE = (3 / 2) * n_BMO(phase) * R * (T - T0_BMO(phase)) + (
        b_MgSiO3(rho, phase) * (f_T_MgSiO3(T, phase) - f_T_MgSiO3(T0_BMO(phase), phase))
    )
    return DE


def DS_MgSiO3(rho, T, phase):
    DS = (3 / 2) * n_BMO(phase) * R * np.log(T / T0_S_MgSiO3(rho)) + (
        b_MgSiO3(rho, phase)
        / (m_BMO(phase) - 1)
        * (f_T_prime_MgSiO3(T, phase) - f_T_prime_MgSiO3(T0_S_MgSiO3(rho), phase))
    )
    return DS


def F_EL_MgSiO3(rho, T, phase):
    zeta = zeta_0_BMO(phase) * x_BMO(rho, phase) ** zeta_exp_BMO(phase)
    T_e = T_e_0_BMO(phase) * x_BMO(rho, phase) ** T_e_exp_BMO(phase)

    F_e = np.where(
        T >= T_e, -zeta * ((1 / 2) * (T**2 - T_e**2) - T * T_e * np.log(T / T_e)), 0
    )
    return F_e


def F_MgSiO3(rho, T, phase):
    F = (
        U0_BMO(phase)
        + F_0_MgSiO3(rho, phase)
        + DE_MgSiO3(rho, T, phase)
        + (
            DS_MgSiO3(rho, T0_BMO(phase), phase) * T0_BMO(phase)
            - DS_MgSiO3(rho, T, phase) * T
        )
        + F_EL_MgSiO3(rho, T, phase)
    )
    return F


# FeO (Morard et al., 2022)
# =============================================================================
# Density dependence of the Gruneisen parameter
def q_FeO(rho, phase):
    q = (
        beta_BMO(phase)
        * x_BMO(rho, phase) ** beta_BMO(phase)
        * (gamma_0_BMO(phase) - gamma_inf_BMO(phase))
    ) / grun_BMO(rho, phase)
    return q


# Debye (Einstein) temperature
def theta_FeO(rho, phase):
    theta = (
        theta_0_BMO(phase)
        * x_BMO(rho, phase) ** -gamma_inf_BMO(phase)
        * np.exp(
            (gamma_0_BMO(phase) - gamma_inf_BMO(phase))
            / beta_BMO(phase)
            * (1 - x_BMO(rho, phase) ** beta_BMO(phase))
        )
    )
    return theta


# Electronic contribution to the free energy
def e_FeO(rho, phase):
    e = e_0_BMO(phase) * x_BMO(rho, phase) ** gamma_e_BMO(phase)
    return e


# Free energy
def E_0_FeO(rho, phase):
    E_0 = (
        9
        * K0_BMO(phase)
        * V0_BMO(phase)
        * eta_BMO(phase) ** -2
        * (
            1
            - (1 - eta_BMO(phase) * (1 - X_BMO(rho, phase)))
            * np.exp(eta_BMO(phase) * (1 - X_BMO(rho, phase)))
        )
    )
    return E_0


def F_th_FeO(rho, T, phase):
    F_th = 3 * n_BMO(phase) * R * T * np.log(1 - np.exp(-theta_FeO(rho, phase) / T))
    return F_th


def F_e_FeO(rho, T, phase):
    F_e = -(3 / 2) * n_BMO(phase) * R * e_FeO(rho, phase) * T**2
    return F_e


def F_FeO(rho, T, phase):
    F = (
        U0_BMO(phase)
        + E_0_FeO(rho, phase)
        + F_th_FeO(rho, T, phase)
        - F_th_FeO(rho, T0_BMO(phase), phase)
        + F_e_FeO(rho, T, phase)
        - F_e_FeO(rho, T0_BMO(phase), phase)
        - a_s_BMO(phase) * R * (T - T0_BMO(phase))
    )
    return F


# Mixing model
# =============================================================================
# Molar mass
def molar_mass_mix(Xi_BMO):
    M = 1 / ((1 - Xi_BMO) / molar_mass_BMO("MgSiO3") + Xi_BMO / molar_mass_BMO("FeO"))
    return M


# Free energy
def F_mix(rho, T, Xi_BMO):
    xi_BMO = wt2mol(Xi_BMO, molar_mass_BMO("FeO"), molar_mass_BMO("MgSiO3"))
    F = F_MgSiO3(rho, T, "MgSiO3") * (1 - xi_BMO) + F_FeO(rho, T, "FeO") * xi_BMO
    return F


# Pressure
def P_mix(rho, T, Xi_BMO, drho=1):
    fm = F_mix(rho - drho, T, Xi_BMO)
    fp = F_mix(rho + drho, T, Xi_BMO)

    dFdrho = (fp - fm) / (2 * drho)

    P = rho**2 / molar_mass_mix(Xi_BMO) * dFdrho
    return P


# Heat capacity at constant volume
def CV_mix(rho, T, Xi_BMO, dT=1):
    fm = F_mix(rho, T - dT, Xi_BMO)
    f0 = F_mix(rho, T, Xi_BMO)
    fp = F_mix(rho, T + dT, Xi_BMO)

    d2FdT2 = (fp + fm - 2 * f0) / dT**2

    CV = -T * d2FdT2
    return CV


# Isothermal bulk modulus
def KT_mix(rho, T, Xi_BMO, drho=1):
    fm = F_mix(rho - drho, T, Xi_BMO)
    f0 = F_mix(rho, T, Xi_BMO)
    fp = F_mix(rho + drho, T, Xi_BMO)

    dFdrho = (fp - fm) / (2 * drho)
    d2Fdrho2 = (fp + fm - 2 * f0) / drho**2

    KT = rho / molar_mass_mix(Xi_BMO) * (2 * rho * dFdrho + rho**2 * d2Fdrho2)
    return KT


# Thermal expansion coefficient
def alpha_mix(rho, T, Xi_BMO, drho=1, dT=1):
    fm = F_mix(rho - drho, T - dT, Xi_BMO)
    fp = F_mix(rho + drho, T - dT, Xi_BMO)
    dFdrho_m = (fp - fm) / (2 * drho)

    fm = F_mix(rho - drho, T + dT, Xi_BMO)
    fp = F_mix(rho + drho, T + dT, Xi_BMO)
    dFdrho_p = (fp - fm) / (2 * drho)

    dFdTdrho = (dFdrho_p - dFdrho_m) / (2 * dT)

    alpha = 1 / KT_mix(rho, T, Xi_BMO) * rho**2 / molar_mass_mix(Xi_BMO) * dFdTdrho
    return alpha


# Thermal Gruneisen parameter
def grun_th_mix(rho, T, Xi_BMO):
    grun_th = (
        alpha_mix(rho, T, Xi_BMO)
        * KT_mix(rho, T, Xi_BMO)
        / CV_mix(rho, T, Xi_BMO)
        * (molar_mass_mix(Xi_BMO) / rho)
    )
    return grun_th


# Isentropic bulk modulus
def KS_mix(rho, T, Xi_BMO):
    KS = KT_mix(rho, T, Xi_BMO) * (
        1 + alpha_mix(rho, T, Xi_BMO) * grun_th_mix(rho, T, Xi_BMO) * T
    )
    return KS


# Heat capacity at constant pressure
def CP_mix(rho, T, Xi_BMO):
    CP = CV_mix(rho, T, Xi_BMO) * (
        1 + alpha_mix(rho, T, Xi_BMO) * grun_th_mix(rho, T, Xi_BMO) * T
    )
    return CP


# %% Melting curves
# =============================================================================
def liquidus_core(P):
    T = (6500 * (P / 340e9) ** 0.515) / (1 - np.log(0.87))
    return T


def liquidus_mantle(P, Xi_BMO):
    # Fei et al. (2021) - MgSiO3 (low)
    TA = 6000 * (P / 140e9) ** 0.26

    # Morard et al. (2022) - FeO
    TB = 1650 * (P / 14.89e9 + 1) ** (1 / 2.77)

    # Linear phase diagram
    T = TA + (TB - TA) * Xi_BMO
    return T


# %% Initial conditions
# =============================================================================
def find_T_CMB_initial(
    ini_type, Xi_BMO_0=Xi_BMO_initial, L_BMO_0=L_BMO_0, T_CMB_0=T_CMB_0
):
    # Adiabat
    if ini_type == "adiabat":
        files = sorted(glob.glob(structure_path + "/Output/T_CMB_*.pkl"))
        i = int(len(files) / 2)
        while True:
            with open(files[i], "rb") as f:
                SE0 = pkl.load(f)
            S0 = BMO_processing(SE0, SE0.T_CMB, Xi_BMO_0, SE0, Xi_BMO_0)
            if S0.R_BMO == S0.R_planet:
                files = files[:i]
            else:
                files = files[i:]
            i = int(len(files) / 2)
            if len(files) == 1:
                T_CMB_initial = int(files[0].split("_")[-1].split(".")[0])
                break

    # Radius
    if ini_type == "radius":
        files = sorted(glob.glob(structure_path + "/Output/T_CMB_*.pkl"))
        i = int(len(files) / 2)
        while True:
            with open(files[i], "rb") as f:
                SE0 = pkl.load(f)
            S0 = BMO_processing(SE0, SE0.T_CMB, Xi_BMO_0, SE0, Xi_BMO_0)
            if S0.R_BMO > (S0.R_CMB + L_BMO_0):
                files = files[:i]
            else:
                files = files[i:]
            i = int(len(files) / 2)
            if len(files) == 1:
                T_CMB_initial = int(files[0].split("_")[-1].split(".")[0])
                break

    # Temperature
    if ini_type == "temperature":
        T_CMB_initial = T_CMB_0

    # Initial structure
    T_CMB_file = structure_path + "/Output/T_CMB_%05d.pkl" % np.round(T_CMB_initial)
    with open(T_CMB_file, "rb") as f:
        SE0 = pkl.load(f)

    S0 = BMO_processing(SE0, SE0.T_CMB, Xi_BMO_initial, SE0, Xi_BMO_initial)

    return S0, SE0, T_CMB_initial


# %% Processing
# =============================================================================
def BMO_processing(S, T_CMB, Xi_BMO, SE_0, Xi_BMO_0):
    # Structure loading
    layer = S.layer
    phase = S.phase
    radius_lim = S.radius_lim

    R_CMB = S.R_CMB
    R_ICB = S.R_ICB

    radius = S.radius
    rho = S.rho
    m = S.m
    g = S.g
    P = S.P
    T = S.T
    alpha = S.alpha
    gamma = S.gamma
    KT = S.KT
    KS = S.KS
    KT_prime = S.KT_prime
    Cp = S.Cp
    DS = S.DS

    # Planet
    R_planet = radius[-1]

    # BSE
    radius_silicate = np.linspace(R_CMB, R_planet, N)
    rho_silicate = np.interp(radius_silicate, radius, rho)
    m_silicate = np.interp(radius_silicate, radius, m)
    g_silicate = np.interp(radius_silicate, radius, g)
    P_silicate = np.interp(radius_silicate, radius, P)
    T_silicate = np.interp(radius_silicate, radius, T)
    alpha_silicate = np.interp(radius_silicate, radius, alpha)
    gamma_silicate = np.interp(radius_silicate, radius, gamma)
    KT_silicate = np.interp(radius_silicate, radius, KT)
    KT_prime_silicate = np.interp(radius_silicate, radius, KT_prime)
    KS_silicate = np.interp(radius_silicate, radius, KS)
    Cp_silicate = np.interp(radius_silicate, radius, Cp)
    DS_silicate = np.interp(radius_silicate, radius, DS)

    radius_silicate_0 = np.linspace(SE_0.R_CMB, SE_0.radius[-1], N)
    rho_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.rho)
    m_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.m)
    g_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.g)
    P_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.P)
    T_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.T)
    alpha_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.alpha)
    gamma_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.gamma)
    KT_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.KT)
    KT_prime_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.KT_prime)
    KS_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.KS)
    Cp_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.Cp)
    DS_silicate_0 = np.interp(radius_silicate_0, SE_0.radius, SE_0.DS)

    radius_silicate_0_i = np.linspace(SE_0.R_CMB, SE_0.radius[-1], N)
    rho_silicate_0_i = np.interp(radius_silicate_0_i, radius, rho)
    m_silicate_0_i = np.interp(radius_silicate_0_i, radius, m)
    g_silicate_0_i = np.interp(radius_silicate_0_i, radius, g)
    P_silicate_0_i = np.interp(radius_silicate_0_i, radius, P)
    T_silicate_0_i = np.interp(radius_silicate_0_i, radius, T)
    alpha_silicate_0_i = np.interp(radius_silicate_0_i, radius, alpha)
    gamma_silicate_0_i = np.interp(radius_silicate_0_i, radius, gamma)
    KT_silicate_0_i = np.interp(radius_silicate_0_i, radius, KT)
    KT_prime_silicate_0_i = np.interp(radius_silicate_0_i, radius, KT_prime)
    KS_silicate_0_i = np.interp(radius_silicate_0_i, radius, KS)
    Cp_silicate_0_i = np.interp(radius_silicate_0_i, radius, Cp)
    DS_silicate_0_i = np.interp(radius_silicate_0_i, radius, DS)

    # Core
    radius_core = np.linspace(0, R_CMB - zero, N)
    rho_core = np.interp(radius_core, radius, rho)
    m_core = np.interp(radius_core, radius, m)
    g_core = np.interp(radius_core, radius, g)
    P_core = np.interp(radius_core, radius, P)
    T_core = np.interp(radius_core, radius, T)
    alpha_core = np.interp(radius_core, radius, alpha)
    gamma_core = np.interp(radius_core, radius, gamma)
    KT_core = np.interp(radius_core, radius, KT)
    KT_prime_core = np.interp(radius_core, radius, KT_prime)
    KS_core = np.interp(radius_core, radius, KS)
    Cp_core = np.interp(radius_core, radius, Cp)
    DS_core = np.interp(radius_core, radius, DS)

    # Outer core
    if np.isnan(R_ICB):
        radius_OC = np.linspace(0, R_CMB - zero, N)
    elif R_ICB == R_CMB:
        radius_OC = np.array([np.nan] * N)
    else:
        radius_OC = np.linspace(R_ICB, R_CMB - zero, N)
    rho_OC = np.interp(radius_OC, radius, rho)
    m_OC = np.interp(radius_OC, radius, m)
    g_OC = np.interp(radius_OC, radius, g)
    P_OC = np.interp(radius_OC, radius, P)
    T_OC = np.interp(radius_OC, radius, T)
    alpha_OC = np.interp(radius_OC, radius, alpha)
    gamma_OC = np.interp(radius_OC, radius, gamma)
    KT_OC = np.interp(radius_OC, radius, KT)
    KT_prime_OC = np.interp(radius_OC, radius, KT_prime)
    KS_OC = np.interp(radius_OC, radius, KS)
    Cp_OC = np.interp(radius_OC, radius, Cp)
    DS_OC = np.interp(radius_OC, radius, DS)

    # Inner core
    if np.isnan(R_ICB):
        radius_IC = np.array([np.nan] * N)
    elif R_ICB == R_CMB:
        radius_IC = np.linspace(0, R_CMB - zero, N)
    else:
        radius_IC = np.linspace(0, R_ICB - zero, N)
    rho_IC = np.interp(radius_IC, radius, rho)
    m_IC = np.interp(radius_IC, radius, m)
    g_IC = np.interp(radius_IC, radius, g)
    P_IC = np.interp(radius_IC, radius, P)
    T_IC = np.interp(radius_IC, radius, T)
    alpha_IC = np.interp(radius_IC, radius, alpha)
    gamma_IC = np.interp(radius_IC, radius, gamma)
    KT_IC = np.interp(radius_IC, radius, KT)
    KT_prime_IC = np.interp(radius_IC, radius, KT_prime)
    KS_IC = np.interp(radius_IC, radius, KS)
    Cp_IC = np.interp(radius_IC, radius, Cp)
    DS_IC = np.interp(radius_IC, radius, DS)

    # BMO
    fit_radius_rho = np.polyfit(radius_silicate, rho_silicate, 1)
    rho_silicate_lin = np.polyval(fit_radius_rho, radius_silicate)
    fit_radius_rho_0 = np.polyfit(radius_silicate_0, rho_silicate_0, 1)
    rho_silicate_lin_0 = np.polyval(fit_radius_rho_0, radius_silicate_0)

    grun_th_BMO_r = grun_th_mix(rho_silicate_lin, T_CMB, Xi_BMO)
    grun_th_BMO_r_0 = grun_th_mix(rho_silicate_lin_0, SE_0.T_CMB, Xi_BMO_0)

    Ta_silicate = T_CMB * np.exp(
        integrate.cumulative_trapezoid(
            grun_th_BMO_r / rho_silicate_lin,
            rho_silicate_lin,
            initial=0,
        )
    )
    Ta_silicate_0 = T_CMB * np.exp(
        integrate.cumulative_trapezoid(
            grun_th_BMO_r_0 / rho_silicate_lin_0,
            rho_silicate_lin_0,
            initial=0,
        )
    )

    Ta = Ta_silicate_0
    Tm = liquidus_mantle(P_silicate_0, Xi_BMO)
    idx = np.argwhere(np.diff(np.sign(Ta - Tm))).flatten()
    if len(idx) == 0:
        if Ta[0] < Tm[0]:
            R_BMO = np.nan
        else:
            R_BMO = R_planet
    else:
        idx = idx[0]
        if Ta[idx] < Tm[idx]:
            R_BMO = np.nan
        else:
            x = radius_silicate_0[idx : idx + 2]
            y = Ta[idx : idx + 2] - Tm[idx : idx + 2]
            a = (y[1] - y[0]) / (x[1] - x[0])
            b = y[0] - a * x[0]
            R_BMO = -b / a

            T_BMO = np.interp(R_BMO, radius_silicate_0, Ta_silicate_0)
            if T_CMB - T_BMO < 2e-2:
                R_BMO = np.nan

    if np.isnan(R_BMO):
        radius_BMO_r = np.array([np.nan] * N)
    else:
        radius_BMO_r = np.linspace(R_CMB, R_BMO, N)
    rho_BMO_r = np.interp(radius_BMO_r, radius_silicate, rho_silicate_lin)
    m_BMO_r = np.interp(radius_BMO_r, radius_silicate, m_silicate)
    g_BMO_r = np.interp(radius_BMO_r, radius_silicate, g_silicate)
    P_BMO_r = np.interp(radius_BMO_r, radius_silicate, P_silicate)

    Ta_BMO_r = np.interp(radius_BMO_r, radius_silicate, Ta_silicate)
    Ta_BMO = Ta_BMO_r[-1]

    alpha_BMO_r = alpha_mix(rho_BMO_r, Ta_BMO_r, Xi_BMO)
    gamma_BMO_r = grun_th_mix(rho_BMO_r, T_CMB, Xi_BMO)
    KT_BMO_r = KT_mix(rho_BMO_r, Ta_BMO_r, Xi_BMO)
    KS_BMO_r = KS_mix(rho_BMO_r, Ta_BMO_r, Xi_BMO)
    Cp_BMO_r = CP_mix(rho_BMO_r, Ta_BMO_r, Xi_BMO) / molar_mass_mix(Xi_BMO)
    DS_BMO_r = entropy_fusion(rho_BMO_r, "mantle")

    if np.isnan(R_BMO):
        radius_BMO_r_0 = np.array([np.nan] * N)
    else:
        radius_BMO_r_0 = np.linspace(SE_0.R_CMB, R_BMO, N)
    rho_BMO_r_0 = np.interp(radius_BMO_r_0, radius_silicate_0, rho_silicate_lin_0)
    m_BMO_r_0 = np.interp(radius_BMO_r_0, radius_silicate_0, m_silicate_0)
    g_BMO_r_0 = np.interp(radius_BMO_r_0, radius_silicate_0, g_silicate_0)
    P_BMO_r_0 = np.interp(radius_BMO_r_0, radius_silicate_0, P_silicate_0)

    Ta_BMO_r_0 = np.interp(radius_BMO_r_0, radius_silicate_0, Ta_silicate_0)
    Ta_BMO_0 = Ta_BMO_r_0[-1]

    alpha_BMO_r_0 = alpha_mix(rho_BMO_r_0, Ta_BMO_r_0, Xi_BMO_0)
    gamma_BMO_r_0 = grun_th_mix(rho_BMO_r_0, SE_0.T_CMB, Xi_BMO_0)
    KT_BMO_r_0 = KT_mix(rho_BMO_r_0, Ta_BMO_r_0, Xi_BMO_0)
    KS_BMO_r_0 = KS_mix(rho_BMO_r_0, Ta_BMO_r_0, Xi_BMO_0)
    Cp_BMO_r_0 = CP_mix(rho_BMO_r_0, Ta_BMO_r_0, Xi_BMO_0) / molar_mass_mix(Xi_BMO_0)
    DS_BMO_r_0 = entropy_fusion(rho_BMO_r_0, "mantle")

    if np.isnan(R_BMO):
        radius_BMO_r_0_i = np.array([np.nan] * N)
    else:
        radius_BMO_r_0_i = np.linspace(SE_0.R_CMB, R_BMO, N)
    rho_BMO_r_0_i = np.interp(radius_BMO_r_0_i, radius_silicate, rho_silicate_lin)
    m_BMO_r_0_i = np.interp(radius_BMO_r_0_i, radius_silicate, m_silicate)
    g_BMO_r_0_i = np.interp(radius_BMO_r_0_i, radius_silicate, g_silicate)
    P_BMO_r_0_i = np.interp(radius_BMO_r_0_i, radius_silicate, P_silicate)

    Ta_BMO_r_0_i = np.interp(radius_BMO_r_0_i, radius_silicate, Ta_silicate)
    Ta_BMO_0_i = Ta_BMO_r_0_i[-1]

    alpha_BMO_r_0_i = alpha_mix(rho_BMO_r_0_i, Ta_BMO_r_0_i, Xi_BMO)
    gamma_BMO_r_0_i = grun_th_mix(rho_BMO_r_0_i, T_CMB, Xi_BMO)
    KT_BMO_r_0_i = KT_mix(rho_BMO_r_0_i, Ta_BMO_r_0_i, Xi_BMO)
    KS_BMO_r_0_i = KS_mix(rho_BMO_r_0_i, Ta_BMO_r_0_i, Xi_BMO)
    Cp_BMO_r_0_i = CP_mix(rho_BMO_r_0_i, Ta_BMO_r_0_i, Xi_BMO) / molar_mass_mix(Xi_BMO)
    DS_BMO_r_0_i = entropy_fusion(rho_BMO_r_0_i, "mantle")

    # Mantle
    if np.isnan(R_BMO):
        radius_mantle = np.linspace(R_CMB, R_planet, N)
    else:
        radius_mantle = np.linspace(R_BMO, R_planet, N)
    rho_mantle = np.interp(radius_mantle, radius, rho)
    m_mantle = np.interp(radius_mantle, radius, m)
    g_mantle = np.interp(radius_mantle, radius, g)
    P_mantle = np.interp(radius_mantle, radius, P)
    alpha_mantle = np.interp(radius_mantle, radius, alpha)
    gamma_mantle = np.interp(radius_mantle, radius, gamma)
    KT_mantle = np.interp(radius_mantle, radius, KT)
    KT_prime_mantle = np.interp(radius_mantle, radius, KT_prime)
    KS_mantle = np.interp(radius_mantle, radius, KS)
    Cp_mantle = np.interp(radius_mantle, radius, Cp)
    DS_mantle = np.interp(radius_mantle, radius, DS)

    if R_BMO == R_planet:
        k_1 = np.nan
        eta_mantle = np.nan
        eta_mantle_1 = np.nan
        DT1 = np.nan
        delta_1 = np.nan
        Q_planet = np.nan
        Ta_mantle = np.array([np.nan] * N)
    else:
        idx = np.argwhere(radius_mantle >= (radius_mantle[0] + R_planet) / 2).T[0]
        eta_1 = np.exp(
            integrate.trapezoid(
                -rho_mantle[idx] * g_mantle[idx] * gamma_mantle[idx] / KS_mantle[idx],
                radius_mantle[idx],
            )
        )
        DT1 = (np.nanmin([T_CMB, Ta_BMO_0]) + T_surface) / 2 * eta_1 - T_surface

        T_mantle_ad = (T_surface + DT1) * np.exp(
            np.flip(
                integrate.cumulative_trapezoid(
                    np.flip(-rho_mantle * g_mantle * gamma_mantle / KS_mantle),
                    np.flip(radius_mantle),
                    initial=0,
                )
            )
        )

        k_surface = k_solid(
            P_mantle[-1],
            T_surface + DT1 / 2,
            gamma_mantle[-1],
            alpha_mantle[-1],
            KT_mantle[-1],
            KT_prime_mantle[-1],
        )

        kappa_mantle_p = k_surface / (rho_mantle[-1] * Cp_mantle[-1])

        T_mantle_ad_AVG = integrate.trapezoid(
            radius_mantle**2 * T_mantle_ad, radius_mantle
        ) / integrate.trapezoid(radius_mantle**2, radius_mantle)

        eta_mantle = eta_solid(T_mantle_ad_AVG, M_planet, CMF)
        eta_mantle_1 = eta_mantle / f_eta

        delta_1 = (R_planet - np.nanmax([R_BMO, R_CMB])) * (
            Ra_c
            * (kappa_mantle_p * eta_mantle_1)
            / (
                rho_mantle[-1]
                * alpha_mantle[-1]
                * g_mantle[-1]
                * DT1
                * (R_planet - np.nanmax([R_BMO, R_CMB])) ** 3
            )
        ) ** beta_mantle

        radius_1 = np.linspace(R_planet - delta_1, R_planet, N)

        P_1 = np.interp(radius_1, radius_mantle, P_mantle)
        gamma_1 = np.interp(radius_1, radius_mantle, gamma_mantle)
        alpha_1 = np.interp(radius_1, radius_mantle, alpha_mantle)
        KT_1 = np.interp(radius_1, radius_mantle, KT_mantle)
        KT_prime_1 = np.interp(radius_1, radius_mantle, KT_prime_mantle)

        P_1_AVG = integrate.trapezoid(
            radius_1**2 * P_1,
            radius_1,
        ) / integrate.trapezoid(radius_1**2, radius_1)
        T_1_AVG = integrate.trapezoid(
            radius_1**2 * (T_surface - DT1 / delta_1 * (radius_1 - R_planet)),
            radius_1,
        ) / integrate.trapezoid(radius_1**2, radius_1)
        gamma_1_AVG = integrate.trapezoid(
            radius_1**2 * gamma_1,
            radius_1,
        ) / integrate.trapezoid(radius_1**2, radius_1)
        alpha_1_AVG = integrate.trapezoid(
            radius_1**2 * alpha_1,
            radius_1,
        ) / integrate.trapezoid(radius_1**2, radius_1)
        KT_1_AVG = integrate.trapezoid(
            radius_1**2 * KT_1,
            radius_1,
        ) / integrate.trapezoid(radius_1**2, radius_1)
        KT_prime_1_AVG = integrate.trapezoid(
            radius_1**2 * KT_prime_1,
            radius_1,
        ) / integrate.trapezoid(radius_1**2, radius_1)

        k_1 = k_solid(
            P_1_AVG, T_1_AVG, gamma_1_AVG, alpha_1_AVG, KT_1_AVG, KT_prime_1_AVG
        )

        Q_planet = 4 * np.pi * R_planet**2 * k_1 * DT1 / delta_1

        T_delta_1 = T_surface + Q_planet / (4 * np.pi * R_planet * k_1) * (
            1 - (R_planet - delta_1) / R_planet
        )

        r_top = np.linspace(R_planet - delta_1, R_planet, N)
        rho_top = np.interp(r_top, radius_mantle, rho_mantle)
        g_top = np.interp(r_top, radius_mantle, g_mantle)
        gamma_top = np.interp(r_top, radius_mantle, gamma_mantle)
        KS_top = np.interp(r_top, radius_mantle, KS_mantle)
        Ta_mantle_top = T_delta_1 * np.exp(
            integrate.cumulative_trapezoid(
                -rho_top * g_top * gamma_top / KS_top,
                r_top,
                initial=0,
            )
        )
        r_bot = np.linspace(np.nanmax([R_BMO, R_CMB]), R_planet - delta_1, N)
        rho_bot = np.interp(r_bot, radius_mantle, rho_mantle)
        g_bot = np.interp(r_bot, radius_mantle, g_mantle)
        gamma_bot = np.interp(r_bot, radius_mantle, gamma_mantle)
        KS_bot = np.interp(r_bot, radius_mantle, KS_mantle)
        Ta_mantle_bot = T_delta_1 * np.exp(
            np.flip(
                integrate.cumulative_trapezoid(
                    np.flip(-rho_bot * g_bot * gamma_bot / KS_bot),
                    np.flip(r_bot),
                    initial=0,
                )
            )
        )

        Ta_mantle = np.interp(
            radius_mantle,
            np.concatenate((r_bot, r_top)),
            np.concatenate((Ta_mantle_bot, Ta_mantle_top)),
        )

    # Gravitational potential
    psi_R_BMO = gravi_potential(radius, g, R_BMO)
    psi_R_BMO_silicate = np.interp(radius_silicate, radius, psi_R_BMO)
    psi_R_BMO_core = np.interp(radius_core, radius, psi_R_BMO)
    psi_R_BMO_OC = np.interp(radius_OC, radius, psi_R_BMO)
    psi_R_BMO_IC = np.interp(radius_IC, radius, psi_R_BMO)
    psi_R_BMO_BMO_r = np.interp(radius_BMO_r, radius, psi_R_BMO)
    psi_R_BMO_BMO_r_0 = np.interp(radius_BMO_r_0, radius, psi_R_BMO)
    psi_R_BMO_BMO_r_0_i = np.interp(radius_BMO_r_0_i, radius, psi_R_BMO)
    psi_R_BMO_mantle = np.interp(radius_mantle, radius, psi_R_BMO)
    psi_R_BMO_BMO = psi_R_BMO_BMO_r[-1]
    psi_R_BMO_BMO_0 = psi_R_BMO_BMO_r_0[-1]
    psi_R_BMO_BMO_0_i = psi_R_BMO_BMO_r_0_i[-1]
    psi_R_BMO_CMB = psi_R_BMO_silicate[0]
    psi_R_BMO_ICB = psi_R_BMO_IC[-1]

    psi_R_CMB = gravi_potential(radius, g, R_CMB)
    psi_R_CMB_silicate = np.interp(radius_silicate, radius, psi_R_CMB)
    psi_R_CMB_core = np.interp(radius_core, radius, psi_R_CMB)
    psi_R_CMB_OC = np.interp(radius_OC, radius, psi_R_CMB)
    psi_R_CMB_IC = np.interp(radius_IC, radius, psi_R_CMB)
    psi_R_CMB_BMO_r = np.interp(radius_BMO_r, radius, psi_R_CMB)
    psi_R_CMB_BMO_r_0 = np.interp(radius_BMO_r_0, radius, psi_R_CMB)
    psi_R_CMB_BMO_r_0_i = np.interp(radius_BMO_r_0_i, radius, psi_R_CMB)
    psi_R_CMB_mantle = np.interp(radius_mantle, radius, psi_R_CMB)
    psi_R_CMB_BMO = psi_R_CMB_BMO_r[-1]
    psi_R_CMB_BMO_0 = psi_R_CMB_BMO_r_0[-1]
    psi_R_CMB_BMO_0_i = psi_R_CMB_BMO_r_0_i[-1]
    psi_R_CMB_CMB = psi_R_CMB_silicate[0]
    psi_R_CMB_ICB = psi_R_CMB_IC[-1]

    # BMO Interface
    R_BMO = R_BMO

    rho_BMO = rho_BMO_r[-1]
    m_BMO = m_BMO_r[-1]
    g_BMO = g_BMO_r[-1]
    P_BMO = P_BMO_r[-1]
    alpha_BMO = alpha_BMO_r[-1]
    gamma_BMO = gamma_BMO_r[-1]
    KT_BMO = KT_BMO_r[-1]
    KS_BMO = KS_BMO_r[-1]
    Cp_BMO = Cp_BMO_r[-1]
    DS_BMO = DS_BMO_r[-1]

    rho_BMO_0 = rho_BMO_r_0[-1]
    m_BMO_0 = m_BMO_r_0[-1]
    g_BMO_0 = g_BMO_r_0[-1]
    P_BMO_0 = P_BMO_r_0[-1]
    alpha_BMO_0 = alpha_BMO_r_0[-1]
    gamma_BMO_0 = gamma_BMO_r_0[-1]
    KT_BMO_0 = KT_BMO_r_0[-1]
    KS_BMO_0 = KS_BMO_r_0[-1]
    Cp_BMO_0 = Cp_BMO_r_0[-1]
    DS_BMO_0 = DS_BMO_r_0[-1]

    rho_BMO_0_i = rho_BMO_r_0_i[-1]
    m_BMO_0_i = m_BMO_r_0_i[-1]
    g_BMO_0_i = g_BMO_r_0_i[-1]
    P_BMO_0_i = P_BMO_r_0_i[-1]
    alpha_BMO_0_i = alpha_BMO_r_0_i[-1]
    gamma_BMO_0_i = gamma_BMO_r_0_i[-1]
    KT_BMO_0_i = KT_BMO_r_0_i[-1]
    KS_BMO_0_i = KS_BMO_r_0_i[-1]
    Cp_BMO_0_i = Cp_BMO_r_0_i[-1]
    DS_BMO_0_i = DS_BMO_r_0_i[-1]

    # CMB interface
    R_CMB = R_CMB
    rho_CMB_top = rho_silicate[0]
    rho_CMB_bot = rho_core[-1]
    m_CMB = m_silicate[0]
    g_CMB = g_silicate[0]
    P_CMB = P_silicate[0]
    T_CMB = T_CMB
    alpha_CMB_top = alpha_silicate[0]
    alpha_CMB_bot = alpha_core[-1]
    gamma_CMB_top = gamma_silicate[0]
    gamma_CMB_bot = gamma_core[-1]
    KT_CMB = KT_silicate[0]
    KT_prime_CMB_top = KT_prime_silicate[0]
    KT_prime_CMB_bot = KT_prime_core[-1]
    KS_CMB_top = KS_silicate[0]
    KS_CMB_bot = KS_core[-1]
    Cp_CMB_top = Cp_silicate[0]
    Cp_CMB_bot = Cp_core[-1]
    DS_CMB_top = DS_silicate[0]
    DS_CMB_bot = DS_core[-1]

    # ICB interface
    R_ICB = R_ICB
    rho_ICB_top = rho_OC[0]
    rho_ICB_bot = rho_IC[-1]
    m_ICB = m_IC[-1]
    g_ICB = g_IC[-1]
    P_ICB = P_IC[-1]
    T_ICB = T_IC[-1]
    alpha_ICB_top = alpha_OC[0]
    alpha_ICB_bot = alpha_IC[-1]
    gamma_ICB_top = gamma_OC[0]
    gamma_ICB_bot = gamma_IC[-1]
    KT_ICB = KT_IC[-1]
    KT_prime_ICB_top = KT_prime_OC[0]
    KT_prime_ICB_bot = KT_prime_IC[-1]
    KS_ICB_top = KS_OC[0]
    KS_ICB_bot = KS_IC[-1]
    Cp_ICB_top = Cp_OC[0]
    Cp_ICB_bot = Cp_IC[-1]
    DS_ICB_top = DS_OC[0]
    DS_ICB_bot = DS_IC[-1]

    if R_ICB == R_CMB:
        T_ICB = T_CMB

    # Center
    R_center = radius[0]
    rho_center = rho[0]
    m_center = m[0]
    g_center = g[0]
    P_center = P[0]
    T_center = T[0]
    alpha_center = alpha[0]
    gamma_center = gamma[0]
    KT_center = KT[0]
    KT_prime_center = KT_prime[0]
    KS_center = KS[0]
    Cp_center = Cp[0]
    DS_center = DS[0]

    # Output
    class Output:
        def __init__(self):
            self.layer = layer
            self.phase = phase
            self.radius_lim = radius_lim

            self.radius = radius
            self.rho = rho
            self.m = m
            self.g = g
            self.P = P
            self.T = T
            self.alpha = alpha
            self.gamma = gamma
            self.KT = KT
            self.KT_prime = KT_prime
            self.KS = KS
            self.Cp = Cp
            self.DS = DS

            # Planet
            self.R_planet = R_planet

            # Silicates
            self.radius_silicate = radius_silicate
            self.rho_silicate = rho_silicate
            self.m_silicate = m_silicate
            self.g_silicate = g_silicate
            self.P_silicate = P_silicate
            self.T_silicate = T_silicate
            self.alpha_silicate = alpha_silicate
            self.gamma_silicate = gamma_silicate
            self.KT_silicate = KT_silicate
            self.KT_prime_silicate = KT_prime_silicate
            self.KS_silicate = KS_silicate
            self.Cp_silicate = Cp_silicate
            self.DS_silicate = DS_silicate

            self.radius_silicate_0 = radius_silicate_0
            self.rho_silicate_0 = rho_silicate_0
            self.m_silicate_0 = m_silicate_0
            self.g_silicate_0 = g_silicate_0
            self.P_silicate_0 = P_silicate_0
            self.T_silicate_0 = T_silicate_0
            self.alpha_silicate_0 = alpha_silicate_0
            self.gamma_silicate_0 = gamma_silicate_0
            self.KT_silicate_0 = KT_silicate_0
            self.KT_prime_silicate_0 = KT_prime_silicate_0
            self.KS_silicate_0 = KS_silicate_0
            self.Cp_silicate_0 = Cp_silicate_0
            self.DS_silicate_0 = DS_silicate_0

            self.radius_silicate_0_i = radius_silicate_0_i
            self.rho_silicate_0_i = rho_silicate_0_i
            self.m_silicate_0_i = m_silicate_0_i
            self.g_silicate_0_i = g_silicate_0_i
            self.P_silicate_0_i = P_silicate_0_i
            self.T_silicate_0_i = T_silicate_0_i
            self.alpha_silicate_0_i = alpha_silicate_0_i
            self.gamma_silicate_0_i = gamma_silicate_0_i
            self.KT_silicate_0_i = KT_silicate_0_i
            self.KT_prime_silicate_0_i = KT_prime_silicate_0_i
            self.KS_silicate_0_i = KS_silicate_0_i
            self.Cp_silicate_0_i = Cp_silicate_0_i
            self.DS_silicate_0_i = DS_silicate_0_i

            # Core
            self.radius_core = radius_core
            self.rho_core = rho_core
            self.m_core = m_core
            self.g_core = g_core
            self.P_core = P_core
            self.T_core = T_core
            self.alpha_core = alpha_core
            self.gamma_core = gamma_core
            self.KT_core = KT_core
            self.KT_prime_core = KT_prime_core
            self.KS_core = KS_core
            self.Cp_core = Cp_core
            self.DS_core = DS_core

            # Outer core
            self.radius_OC = radius_OC
            self.rho_OC = rho_OC
            self.m_OC = m_OC
            self.g_OC = g_OC
            self.P_OC = P_OC
            self.T_OC = T_OC
            self.alpha_OC = alpha_OC
            self.gamma_OC = gamma_OC
            self.KT_OC = KT_OC
            self.KT_prime_OC = KT_prime_OC
            self.KS_OC = KS_OC
            self.Cp_OC = Cp_OC
            self.DS_OC = DS_OC

            # Inner core
            self.radius_IC = radius_IC
            self.rho_IC = rho_IC
            self.m_IC = m_IC
            self.g_IC = g_IC
            self.P_IC = P_IC
            self.T_IC = T_IC
            self.alpha_IC = alpha_IC
            self.gamma_IC = gamma_IC
            self.KT_IC = KT_IC
            self.KT_prime_IC = KT_prime_IC
            self.KS_IC = KS_IC
            self.Cp_IC = Cp_IC
            self.DS_IC = DS_IC

            # BMO
            self.radius_BMO_r = radius_BMO_r
            self.rho_BMO_r = rho_BMO_r
            self.m_BMO_r = m_BMO_r
            self.g_BMO_r = g_BMO_r
            self.P_BMO_r = P_BMO_r
            self.alpha_BMO_r = alpha_BMO_r
            self.gamma_BMO_r = gamma_BMO_r
            self.KT_BMO_r = KT_BMO_r
            self.KS_BMO_r = KS_BMO_r
            self.Cp_BMO_r = Cp_BMO_r
            self.DS_BMO_r = DS_BMO_r

            self.Ta_BMO_r = Ta_BMO_r
            self.Ta_BMO = Ta_BMO

            self.radius_BMO_r_0 = radius_BMO_r_0
            self.rho_BMO_r_0 = rho_BMO_r_0
            self.m_BMO_r_0 = m_BMO_r_0
            self.g_BMO_r_0 = g_BMO_r_0
            self.P_BMO_r_0 = P_BMO_r_0
            self.alpha_BMO_r_0 = alpha_BMO_r_0
            self.gamma_BMO_r_0 = gamma_BMO_r_0
            self.KT_BMO_r_0 = KT_BMO_r_0
            self.KS_BMO_r_0 = KS_BMO_r_0
            self.Cp_BMO_r_0 = Cp_BMO_r_0
            self.DS_BMO_r_0 = DS_BMO_r_0

            self.Ta_BMO_r_0 = Ta_BMO_r_0
            self.Ta_BMO_0 = Ta_BMO_0

            self.radius_BMO_r_0_i = radius_BMO_r_0_i
            self.rho_BMO_r_0_i = rho_BMO_r_0_i
            self.m_BMO_r_0_i = m_BMO_r_0_i
            self.g_BMO_r_0_i = g_BMO_r_0_i
            self.P_BMO_r_0_i = P_BMO_r_0_i
            self.alpha_BMO_r_0_i = alpha_BMO_r_0_i
            self.gamma_BMO_r_0_i = gamma_BMO_r_0_i
            self.KT_BMO_r_0_i = KT_BMO_r_0_i
            self.KS_BMO_r_0_i = KS_BMO_r_0_i
            self.Cp_BMO_r_0_i = Cp_BMO_r_0_i
            self.DS_BMO_r_0_i = DS_BMO_r_0_i

            self.Ta_BMO_r_0_i = Ta_BMO_r_0_i
            self.Ta_BMO_0_i = Ta_BMO_0_i

            # Mantle
            self.radius_mantle = radius_mantle
            self.rho_mantle = rho_mantle
            self.m_mantle = m_mantle
            self.g_mantle = g_mantle
            self.P_mantle = P_mantle
            self.alpha_mantle = alpha_mantle
            self.gamma_mantle = gamma_mantle
            self.KT_mantle = KT_mantle
            self.KT_prime_mantle = KT_prime_mantle
            self.KS_mantle = KS_mantle
            self.Cp_mantle = Cp_mantle
            self.DS_mantle = DS_mantle

            # Mantle temperature profile
            self.k_1 = k_1
            self.eta_mantle = eta_mantle
            self.eta_mantle_1 = eta_mantle_1
            self.DT1 = DT1
            self.delta_1 = delta_1
            self.Q_planet = Q_planet
            self.Ta_mantle = Ta_mantle

            # Gravitational potential
            self.psi_R_BMO = psi_R_BMO
            self.psi_R_BMO_silicate = psi_R_BMO_silicate
            self.psi_R_BMO_core = psi_R_BMO_core
            self.psi_R_BMO_OC = psi_R_BMO_OC
            self.psi_R_BMO_IC = psi_R_BMO_IC
            self.psi_R_BMO_BMO_r = psi_R_BMO_BMO_r
            self.psi_R_BMO_BMO_r_0 = psi_R_BMO_BMO_r_0
            self.psi_R_BMO_BMO_r_0_i = psi_R_BMO_BMO_r_0_i
            self.psi_R_BMO_mantle = psi_R_BMO_mantle
            self.psi_R_BMO_BMO = psi_R_BMO_BMO
            self.psi_R_BMO_BMO_0 = psi_R_BMO_BMO_0
            self.psi_R_BMO_BMO_0_i = psi_R_BMO_BMO_0_i
            self.psi_R_BMO_CMB = psi_R_BMO_CMB
            self.psi_R_BMO_ICB = psi_R_BMO_ICB
            self.psi_R_CMB = psi_R_CMB
            self.psi_R_CMB_silicate = psi_R_CMB_silicate
            self.psi_R_CMB_core = psi_R_CMB_core
            self.psi_R_CMB_OC = psi_R_CMB_OC
            self.psi_R_CMB_IC = psi_R_CMB_IC
            self.psi_R_CMB_BMO_r = psi_R_CMB_BMO_r
            self.psi_R_CMB_BMO_r_0 = psi_R_CMB_BMO_r_0
            self.psi_R_CMB_BMO_r_0_i = psi_R_CMB_BMO_r_0_i
            self.psi_R_CMB_mantle = psi_R_CMB_mantle
            self.psi_R_CMB_BMO = psi_R_CMB_BMO
            self.psi_R_CMB_BMO_0 = psi_R_CMB_BMO_0
            self.psi_R_CMB_BMO_0_i = psi_R_CMB_BMO_0_i
            self.psi_R_CMB_CMB = psi_R_CMB_CMB
            self.psi_R_CMB_ICB = psi_R_CMB_ICB

            # BMO interface
            self.R_BMO = R_BMO

            self.rho_BMO = rho_BMO
            self.m_BMO = m_BMO
            self.g_BMO = g_BMO
            self.P_BMO = P_BMO
            self.alpha_BMO = alpha_BMO
            self.gamma_BMO = gamma_BMO
            self.KT_BMO = KT_BMO
            self.KS_BMO = KS_BMO
            self.Cp_BMO = Cp_BMO
            self.DS_BMO = DS_BMO

            self.rho_BMO_0 = rho_BMO_0
            self.m_BMO_0 = m_BMO_0
            self.g_BMO_0 = g_BMO_0
            self.P_BMO_0 = P_BMO_0
            self.alpha_BMO_0 = alpha_BMO_0
            self.gamma_BMO_0 = gamma_BMO_0
            self.KT_BMO_0 = KT_BMO_0
            self.KS_BMO_0 = KS_BMO_0
            self.Cp_BMO_0 = Cp_BMO_0
            self.DS_BMO_0 = DS_BMO_0

            self.rho_BMO_0_i = rho_BMO_0_i
            self.m_BMO_0_i = m_BMO_0_i
            self.g_BMO_0_i = g_BMO_0_i
            self.P_BMO_0_i = P_BMO_0_i
            self.alpha_BMO_0_i = alpha_BMO_0_i
            self.gamma_BMO_0_i = gamma_BMO_0_i
            self.KT_BMO_0_i = KT_BMO_0_i
            self.KS_BMO_0_i = KS_BMO_0_i
            self.Cp_BMO_0_i = Cp_BMO_0_i
            self.DS_BMO_0_i = DS_BMO_0_i

            # CMB interface
            self.R_CMB = R_CMB
            self.rho_CMB_top = rho_CMB_top
            self.rho_CMB_bot = rho_CMB_bot
            self.m_CMB = m_CMB
            self.g_CMB = g_CMB
            self.P_CMB = P_CMB
            self.T_CMB = T_CMB
            self.alpha_CMB_top = alpha_CMB_top
            self.alpha_CMB_bot = alpha_CMB_bot
            self.gamma_CMB_top = gamma_CMB_top
            self.gamma_CMB_bot = gamma_CMB_bot
            self.KT_CMB = KT_CMB
            self.KT_prime_CMB_top = KT_prime_CMB_top
            self.KT_prime_CMB_bot = KT_prime_CMB_bot
            self.KS_CMB_top = KS_CMB_top
            self.KS_CMB_bot = KS_CMB_bot
            self.Cp_CMB_top = Cp_CMB_top
            self.Cp_CMB_bot = Cp_CMB_bot
            self.DS_CMB_top = DS_CMB_top
            self.DS_CMB_bot = DS_CMB_bot

            # ICB interface
            self.R_ICB = R_ICB
            self.rho_ICB_top = rho_ICB_top
            self.rho_ICB_bot = rho_ICB_bot
            self.m_ICB = m_ICB
            self.g_ICB = g_ICB
            self.P_ICB = P_ICB
            self.T_ICB = T_ICB
            self.alpha_ICB_top = alpha_ICB_top
            self.alpha_ICB_bot = alpha_ICB_bot
            self.gamma_ICB_top = gamma_ICB_top
            self.gamma_ICB_bot = gamma_ICB_bot
            self.KT_ICB = KT_ICB
            self.KT_prime_ICB_top = KT_prime_ICB_top
            self.KT_prime_ICB_bot = KT_prime_ICB_bot
            self.KS_ICB_top = KS_ICB_top
            self.KS_ICB_bot = KS_ICB_bot
            self.Cp_ICB_top = Cp_ICB_top
            self.Cp_ICB_bot = Cp_ICB_bot
            self.DS_ICB_top = DS_ICB_top
            self.DS_ICB_bot = DS_ICB_bot

            # Center
            self.R_center = R_center
            self.rho_center = rho_center
            self.m_center = m_center
            self.g_center = g_center
            self.P_center = P_center
            self.T_center = T_center
            self.alpha_center = alpha_center
            self.gamma_center = gamma_center
            self.KT_center = KT_center
            self.KT_prime_center = KT_prime_center
            self.KS_center = KS_center
            self.Cp_center = Cp_center
            self.DS_center = DS_center

    return Output()


# %% Energy budget
# =============================================================================
# Mantle
# =============================================================================
# Mantle secular cooling
def P_S_mantle(S):
    Q = -integrate.trapezoid(
        4
        * np.pi
        * S.radius_mantle**2
        * S.rho_mantle
        * S.Cp_mantle
        * ((1 / 2) * np.nanmin([S.Ta_BMO_0, S.T_CMB]) / S.T_CMB),
        S.radius_mantle,
    )
    return Q


# Mantle surface flow
def Q_planet_fun(S):
    Q = S.Q_planet
    return Q


# Mantle radiogenic heating
def h_mantle(t, t0=t_earth):
    def lbd(t12):
        l = np.log(2) / t12
        return l

    def A0(c, a, e, lbd, M):
        q = c * a * e * N_A * lbd / M
        return q

    def A(t, t0, A0, lbd):
        q = A0 * np.exp(-(t - t0) * lbd)
        return q

    h = (
        A(
            t,
            t0,
            A0(c_U238_mantle, a_U238, e_U238, lbd(t12_U238), M_U238),
            lbd(t12_U238),
        )
        + A(
            t,
            t0,
            A0(c_U235_mantle, a_U235, e_U235, lbd(t12_U235), M_U235),
            lbd(t12_U235),
        )
        + A(
            t,
            t0,
            A0(c_Th232_mantle, a_Th232, e_Th232, lbd(t12_Th232), M_Th232),
            lbd(t12_Th232),
        )
        + A(
            t,
            t0,
            A0(c_K40_mantle, a_K40, e_K40, lbd(t12_K40), M_K40),
            lbd(t12_K40),
        )
    ) * f_R_mantle
    return h


def Q_R_mantle_fun(S, t):
    M_mantle = integrate.trapezoid(
        4 * np.pi * S.radius_mantle**2 * S.rho_mantle,
        S.radius_mantle,
    )
    Q = h_mantle(t, t_earth) * M_mantle
    return Q


# Basal magma ocean
# =============================================================================
# BMO partition coefficient
def D_BMO_fun(Xi_BMO, D_BMO_0):
    if isinstance(Xi_BMO, (int, float)):
        Xi_BMO = np.array([Xi_BMO])
    D_BMO = D_BMO_0 * np.ones(len(Xi_BMO))
    if len(Xi_BMO) == 1:
        D_BMO = D_BMO[0]

    return D_BMO


# CXI
def CXI_BMO(S, Xi_BMO):
    D_BMO = D_BMO_fun(Xi_BMO, D_BMO_0)
    DXi_BMO = Xi_BMO * (1 - D_BMO)
    CXI = (
        -4
        * np.pi
        * S.rho_BMO_0
        * S.R_BMO**2
        * DXi_BMO
        / integrate.trapezoid(
            4 * np.pi * S.radius_BMO_r_0**2 * S.rho_BMO_r_0,
            S.radius_BMO_r_0,
        )
    )
    return CXI


# CR
def CR_BMO(S, Xi_BMO):
    Ta = S.Ta_BMO_r_0[-2:]
    Tm = liquidus_mantle(S.P_BMO_r_0, Xi_BMO)[-2:]

    dTmdXi = np.diff(liquidus_mantle(S.P_BMO_0, np.array([0, 1])))[0]

    r = S.radius_BMO_r_0[-2:]
    CR = (S.Ta_BMO_0 / S.T_CMB) / (
        ((Tm[1] - Tm[0]) / (r[1] - r[0]) - (Ta[1] - Ta[0]) / (r[1] - r[0]))
        + CXI_BMO(S, Xi_BMO) * dTmdXi
    )
    return CR


# BMO radioactive heating (Q_R_BMO)
def Q_R_BMO_fun(S, t):
    M_BMO = integrate.trapezoid(
        4 * np.pi * S.radius_BMO_r_0_i**2 * S.rho_BMO_r_0_i,
        S.radius_BMO_r_0_i,
    )
    Q = h_mantle(t, t_earth) * M_BMO
    return Q


# BMO secular cooling (Q_S_BMO)
def P_S_BMO(S, Xi_BMO):
    Q = -integrate.trapezoid(
        4
        * np.pi
        * S.radius_BMO_r_0_i**2
        * S.rho_BMO_r_0_i
        * S.Cp_BMO_r_0_i
        * (S.Ta_BMO_r_0 / S.T_CMB),
        S.radius_BMO_r_0_i,
    )
    return Q


# BMO latent heat (Q_L_BMO)
def P_L_BMO(S, Xi_BMO):
    Q = (
        -4
        * np.pi
        * S.R_BMO**2
        * S.DS_BMO_0_i
        * S.Ta_BMO_0
        * S.rho_BMO_0_i
        * CR_BMO(S, Xi_BMO)
    )
    return Q


# BMO compositional/gravitational energy (Q_G_BMO)
def P_G_BMO(S, Xi_BMO):
    D_BMO = D_BMO_fun(Xi_BMO, D_BMO_0)
    DXi_BMO = Xi_BMO * (1 - D_BMO)
    alpha_c_BMO = Drho_BMO / (S.rho_mantle[0] * DXi_BMO)
    Q = integrate.trapezoid(
        -4
        * np.pi
        * alpha_c_BMO
        * S.radius_BMO_r_0_i**2
        * S.rho_BMO_r_0_i
        * S.psi_R_BMO_BMO_r_0_i
        * CR_BMO(S, Xi_BMO)
        * CXI_BMO(S, Xi_BMO),
        S.radius_BMO_r_0_i,
    )
    return Q


# BMO equivalent energy for the compositional sink (P_C_BMO)
def P_C_BMO(S, Xi_BMO):
    D_BMO = D_BMO_fun(Xi_BMO, D_BMO_0)
    DXi_BMO = Xi_BMO * (1 - D_BMO)
    alpha_c_BMO = Drho_BMO / (S.rho_mantle[0] * DXi_BMO)
    Q = integrate.trapezoid(
        4
        * np.pi
        * S.radius_BMO_r_0_i**2
        * S.rho_BMO_r_0_i
        * S.Cp_BMO_r_0_i
        * (alpha_c_BMO / S.alpha_BMO_r_0_i)
        * CR_BMO(S, Xi_BMO)
        * CXI_BMO(S, Xi_BMO),
        S.radius_BMO_r_0_i,
    )
    return Q


# BMO heat flow (Q_BMO)
def Q_BMO_fun(S, dT_CMBdt, t):
    Q = np.nansum(
        [Q_planet_fun(S), -Q_R_mantle_fun(S, t), -P_S_mantle(S) * dT_CMBdt], axis=0
    )
    return Q


# BMO adiabatic heat flow
def Q_A_BMO_fun(S, k_BMO):
    dTadr = (S.Ta_BMO_r_0[-1] - S.Ta_BMO_r_0[-2]) / (
        S.radius_BMO_r_0[-1] - S.radius_BMO_r_0[-2]
    )
    Q = -k_BMO * dTadr * 4 * np.pi * S.R_BMO**2
    return Q


# Core
# =============================================================================
# CR
def CR_core(S, Xi_core):
    Ta = S.T_OC[:2]
    Tm = liquidus_core(S.P_OC)[:2]

    r = S.radius_OC[:2]
    CR = (S.T_ICB / S.T_CMB) / (
        ((Tm[1] - Tm[0]) / (r[1] - r[0]) - (Ta[1] - Ta[0]) / (r[1] - r[0]))
    )
    return CR


# CXI
def CXI_core(S, Xi_core):
    DXi_core = Xi_core
    CXI = (
        4
        * np.pi
        * S.rho_ICB_top
        * S.R_ICB**2
        * DXi_core
        / integrate.trapezoid(4 * np.pi * S.radius_OC**2 * S.rho_OC, S.radius_OC)
    )
    return CXI


# Core radioactive heating (Q_R_core)
def h_core(t, t0=t_earth):
    def lbd(t12):
        l = np.log(2) / t12
        return l

    def A0(c, a, e, lbd, M):
        q = c * a * e * N_A * lbd / M
        return q

    def A(t, t0, A0, lbd):
        q = A0 * np.exp(-(t - t0) * lbd)
        return q

    h = A(
        t,
        t0,
        A0(c_K40_core, a_K40, e_K40, lbd(t12_K40), M_K40),
        lbd(t12_K40),
    )
    return h


def Q_R_core_fun(S, t):
    M_OC = integrate.trapezoid(4 * np.pi * S.radius_OC**2 * S.rho_OC, S.radius_OC)
    Q = h_core(t, t_earth) * M_OC
    return Q


# Core secular cooling (Q_S_core)
def P_S_core(S, Xi_core):
    Q = -integrate.trapezoid(
        4 * np.pi * S.radius_OC**2 * S.rho_OC * S.Cp_OC * (S.T_OC / S.T_CMB),
        S.radius_OC,
    )
    return Q


# Core latent heat (Q_L_core)
def P_L_core(S, Xi_core):
    Q = (
        4
        * np.pi
        * S.R_ICB**2
        * S.DS_ICB_top
        * S.T_ICB
        * S.rho_ICB_top
        * CR_core(S, Xi_core)
    )
    return Q


# Core compositional/gravitational energy (Q_G_core)
def P_G_core(S, Xi_core):
    DXi_core = Xi_core
    alpha_c_core = Drho_ICB / (S.rho_ICB_bot * DXi_core)
    Q = (
        alpha_c_core
        * CR_core(S, Xi_core)
        * CXI_core(S, Xi_core)
        * (
            integrate.trapezoid(
                4 * np.pi * S.radius_OC**2 * S.rho_OC * S.psi_R_CMB_OC,
                S.radius_OC,
            )
            - integrate.trapezoid(4 * np.pi * S.radius_OC**2 * S.rho_OC, S.radius_OC)
            * S.psi_R_CMB_ICB
        )
    )
    return Q


# Core equivalent energy for the compositional sink (P_C_core)
def P_C_core(S, Xi_core):
    DXi_core = Xi_core
    alpha_c_core = Drho_ICB / (S.rho_ICB_bot * DXi_core)
    Q = integrate.trapezoid(
        4
        * np.pi
        * S.radius_OC**2
        * S.rho_OC
        * S.Cp_OC
        * (alpha_c_core / S.alpha_OC)
        * CR_core(S, Xi_core)
        * CXI_core(S, Xi_core),
        S.radius_OC,
    )
    return Q


# Inner core radioactive heating (Q_R_IC)
def Q_R_IC_fun(S, t):
    M_IC = integrate.trapezoid(4 * np.pi * S.radius_IC**2 * S.rho_IC, S.radius_IC)
    Q = h_core(t, t_earth) * M_IC
    return Q


# Inner core secular cooling (Q_S_IC)
def P_S_IC(S, Xi_core):
    Q = -integrate.trapezoid(
        4 * np.pi * S.radius_IC**2 * S.rho_IC * S.Cp_IC * (S.T_ICB / S.T_CMB),
        S.radius_IC,
    )
    return Q


# CMB heat flow (Q_CMB)
def Q_CMB_fun(S, Xi_core, dT_CMBdt, t):
    Q = np.nansum(
        [
            Q_R_core_fun(S, t),
            P_S_core(S, Xi_core) * dT_CMBdt,
            P_L_core(S, Xi_core) * dT_CMBdt,
            P_G_core(S, Xi_core) * dT_CMBdt,
            Q_R_IC_fun(S, t),
            P_S_IC(S, Xi_core) * dT_CMBdt,
        ],
        axis=0,
    )
    return Q


# ICB heat flow (Q_ICB)
def Q_ICB_fun(S, Xi_core, dT_CMBdt, t):
    Q = np.nansum(
        [
            Q_R_IC_fun(S, t),
            P_S_IC(S, Xi_core) * dT_CMBdt,
        ],
        axis=0,
    )
    return Q


# Core adiabatic heat flow
def Q_A_core_fun(S, k_core):
    dTadr = (S.T_core[-1] - S.T_core[-2]) / (S.radius_core[-1] - S.radius_core[-2])
    Q = -k_core * dTadr * 4 * np.pi * S.R_CMB**2
    return Q


# %% Solver
# =============================================================================
def Q_budget(t, z, SE_0, Xi_BMO_0):
    # Variables
    T_CMB, Xi_BMO, Xi_core = z

    # Load structure
    T_CMB_file = structure_path + "/Output/T_CMB_%05d.pkl" % np.round(T_CMB)
    with open(T_CMB_file, "rb") as f:
        SE = pkl.load(f)

    # BMO processing
    S = BMO_processing(SE, T_CMB, Xi_BMO, SE_0, Xi_BMO_0)

    # Print
    if verbose:
        print(
            "t = %.3f Ga ; z =" % (s2y(t) / 1e9),
            z,
        )
        print(
            "Q_p = %.2f TW ; D_BMO = %.2f km ; R_ICB = %.2f km"
            % (Q_planet_fun(S) / 1e12, (S.R_BMO - SE_0.R_CMB) / 1e3, S.R_ICB / 1e3)
        )

    # Budget flags
    if ~np.isnan(S.R_ICB):
        IC_flag = True
    else:
        IC_flag = False

    if ~np.isnan(S.radius_OC[0]):
        OC_flag = True
    else:
        OC_flag = False

    if ~np.isnan(S.R_BMO):
        BMO_flag = True
    else:
        BMO_flag = False

    # Budgets
    # (- IC + OC + BMO)
    if not IC_flag and OC_flag and BMO_flag:
        dT_CMBdt = (
            Q_planet_fun(S)
            - Q_R_mantle_fun(S, t)
            - Q_R_BMO_fun(S, t)
            - Q_R_core_fun(S, t)
        ) / (
            P_S_BMO(S, Xi_BMO)
            + P_L_BMO(S, Xi_BMO)
            + P_G_BMO(S, Xi_BMO)
            + P_S_core(S, Xi_core)
            + P_S_mantle(S)
        )

    # (- IC + OC - BMO)
    if not IC_flag and OC_flag and not BMO_flag:
        dT_CMBdt = (Q_planet_fun(S) - Q_R_mantle_fun(S, t) - Q_R_core_fun(S, t)) / (
            P_S_core(S, Xi_core) + P_S_mantle(S)
        )

    # (+ IC + OC + BMO)
    if IC_flag and OC_flag and BMO_flag:
        dT_CMBdt = (
            Q_planet_fun(S)
            - Q_R_mantle_fun(S, t)
            - Q_R_BMO_fun(S, t)
            - Q_R_core_fun(S, t)
            - Q_R_IC_fun(S, t)
        ) / (
            P_S_BMO(S, Xi_BMO)
            + P_L_BMO(S, Xi_BMO)
            + P_G_BMO(S, Xi_BMO)
            + P_S_core(S, Xi_core)
            + P_L_core(S, Xi_core)
            + P_G_core(S, Xi_core)
            + P_S_IC(S, Xi_core)
            + P_S_mantle(S)
        )

    # (+ IC - OC + BMO)
    if IC_flag and not OC_flag and BMO_flag:
        dT_CMBdt = (
            Q_planet_fun(S)
            - Q_R_mantle_fun(S, t)
            - Q_R_BMO_fun(S, t)
            - Q_R_IC_fun(S, t)
        ) / (
            P_S_BMO(S, Xi_BMO)
            + P_L_BMO(S, Xi_BMO)
            + P_G_BMO(S, Xi_BMO)
            + P_S_IC(S, Xi_core)
            + P_S_mantle(S)
        )

    # (+ IC + OC - BMO)
    if IC_flag and OC_flag and not BMO_flag:
        dT_CMBdt = (
            Q_planet_fun(S)
            - Q_R_mantle_fun(S, t)
            - Q_R_core_fun(S, t)
            - Q_R_IC_fun(S, t)
        ) / (
            P_S_core(S, Xi_core)
            + P_L_core(S, Xi_core)
            + P_G_core(S, Xi_core)
            + P_S_IC(S, Xi_core)
            + P_S_mantle(S)
        )

    # (+ IC - OC - BMO)
    if IC_flag and not OC_flag and not BMO_flag:
        dT_CMBdt = (Q_planet_fun(S) - Q_R_mantle_fun(S, t) - Q_R_IC_fun(S, t)) / (
            P_S_IC(S, Xi_core) + P_S_mantle(S)
        )

    # Derivatives
    if BMO_flag:
        dR_BMOdt = CR_BMO(S, Xi_BMO) * dT_CMBdt
        if Xi_BMO > 1:
            dXi_BMOdt = 0
        else:
            dXi_BMOdt = CXI_BMO(S, Xi_BMO) * dR_BMOdt
    else:
        dXi_BMOdt = 0

    if IC_flag and OC_flag:
        dR_ICBdt = CR_core(S, Xi_core) * dT_CMBdt
        if Xi_core > 1:
            dXi_coredt = 0
        else:
            dXi_coredt = CXI_core(S, Xi_core) * dR_ICBdt
    else:
        dXi_coredt = 0

    return dT_CMBdt, dXi_BMOdt, dXi_coredt


# %% Post-processing
# =============================================================================
def structure_pp(t, T_CMB, Xi_BMO, SE0, Xi_BMO_initial):
    S = np.zeros(len(t), dtype=object)
    for i in range(len(t)):
        T_CMB_file = structure_path + "/Output/T_CMB_%05d.pkl" % np.round(T_CMB[i])
        with open(T_CMB_file, "rb") as f:
            SE = pkl.load(f)

        S[i] = BMO_processing(SE, T_CMB[i], Xi_BMO[i], SE0, Xi_BMO_initial)
    return S, SE


def temperature_pp(t, S, Q_BMO, Q_planet):
    T_mantle, k_mantle = (np.zeros((len(t), N)) for i in range(2))
    for i in range(len(t)):
        k_mantle[i] = k_solid(
            S[i].P_mantle,
            S[i].Ta_mantle,
            S[i].gamma_mantle,
            S[i].alpha_mantle,
            S[i].KT_mantle,
            S[i].KT_prime_mantle,
        )

        k_2 = k_mantle[i, 0]

        T2 = np.nanmin([S[i].Ta_BMO_0, S[i].T_CMB]) + Q_BMO[i] / (
            4 * np.pi * np.nanmax([S[i].R_BMO, S[i].R_CMB]) * k_2
        ) * (1 - S[i].radius_mantle / np.nanmax([S[i].R_BMO, S[i].R_CMB]))

        idx = np.argwhere(np.diff(np.sign(T2 - S[i].Ta_mantle))).flatten()[0]

        x = np.nanmin([S[i].Ta_BMO_0, S[i].T_CMB]) - T2[idx : idx + 2]
        y = T2[idx : idx + 2] - S[i].Ta_mantle[idx : idx + 2]
        a = (y[1] - y[0]) / (x[1] - x[0])
        b = y[0] - a * x[0]
        DT2 = -b / a

        delta_2 = (
            4
            * np.pi
            * (np.nanmax([S[i].R_BMO, S[i].R_CMB])) ** 2
            * k_2
            * DT2
            / Q_BMO[i]
        )

        idx = np.argwhere(S[i].radius_mantle > S[i].R_planet - S[i].delta_1)
        T_mantle[i, idx] = T_surface + Q_planet[i] / (
            4 * np.pi * S[i].R_planet * S[i].k_1
        ) * (1 - S[i].radius_mantle[idx] / S[i].R_planet)

        idx = np.argwhere(
            (S[i].radius_mantle <= S[i].R_planet - S[i].delta_1)
            & (S[i].radius_mantle > np.nanmax([S[i].R_BMO, S[i].R_CMB]) + delta_2)
        )
        T_mantle[i, idx] = S[i].Ta_mantle[idx]

        idx = np.argwhere(
            S[i].radius_mantle <= np.nanmax([S[i].R_BMO, S[i].R_CMB]) + delta_2
        )
        T_mantle[i, idx] = np.nanmin([S[i].Ta_BMO_0, S[i].T_CMB]) + Q_BMO[i] / (
            4 * np.pi * np.nanmax([S[i].R_BMO, S[i].R_CMB]) * k_2
        ) * (1 - S[i].radius_mantle[idx] / np.nanmax([S[i].R_BMO, S[i].R_CMB]))

    return T_mantle, k_mantle
