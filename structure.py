#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Victor Lherm

All rights reserved.

This code is provided for academic and research purposes only. Any use of this code in published work or presentations must include a citation to the original author. Unauthorized copying, modification, or distribution of this code is prohibited without prior written permission from the author.

Computes the internal structure of a planet with a given mass and core mass fraction, at multiple thermal states.

Returns a .pkl file for each CMB temperature (T_CMB_XXXXX.pkl) within the computed range. Each file contains a class with attributes describing the internal structure (e.g. density, pressure, temperature, and other thermodynamic properties) as a function of radius.

Created by Victor Lherm on 2025-05-07 10:00:00.
"""

# %% Modules
# =============================================================================
import os, time, warnings
import numpy as np
import dill as pkl

from scipy import integrate, optimize
from scipy.constants import G, mu_0, R, k, N_A, eV
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

# %% General parameters
# =============================================================================
# Folder parameters
output_dir = "Demo/Output"  # Folder name for the output

# Temperature (CMB) search range
T_CMB_search = np.arange(4000, 6000, 1)  # CMB temperature search range [K]

# Numerical parameters
N = 1024  # Number of grid points
zero = 1e-6  # Numerical zero
R_crit = 1e3  # Maximum radius of the innermost shell [m]

# Initial conditions
R_planet_initial_coef = 2.0  # Initial radius coefficient
radius_search_coef = 0.5  # Radius search coefficient

# Earth
M_earth = 5.972e24  # Earth mass [kg]
R_earth = 6371e3  # Earth radius [m]

# Planet
CMF = 0.32  # Core mass fraction

M_planet = M_earth  # Planet mass [kg]

R_planet_initial = R_earth  # Planet radius [m]
g_planet_initial = G * M_planet / R_planet_initial**2  # Planet surface gravity [m/s2]

T_surface = 300  # Surface temperature [K]
P_surface = 1.01325e5  # Surface pressure [Pa]

# Mantle
model_phase = "TT11"  # Model: "TT11" or "U17"


# %% Mass-radius relations (Seager et al., 2007)
# =============================================================================
def radius_mass(M, CMF):
    k1_m, k2_m, k3_m = -0.209490, 0.0804, 0.394
    m1_m, r1_m = 5.80, 2.52

    M_S = (M / M_earth) / m1_m
    log10_R_S = k1_m + (1 / 3) * np.log10(M_S) - k2_m * M_S**k3_m
    R_S = 10**log10_R_S
    R_m = R_S * r1_m

    k1_s, k2_s, k3_s = -0.209594, 0.0799, 0.413
    m1_s, r1_s = 10.55, 3.90

    M_S = (M / M_earth) / m1_s
    log10_R_S = k1_s + (1 / 3) * np.log10(M_S) - k2_s * M_S**k3_s
    R_S = 10**log10_R_S
    R_s = R_S * r1_s

    R = CMF * R_m + (1 - CMF) * R_s
    return R * R_earth


R_planet_initial = radius_mass(M_planet, CMF)
g_planet_initial = G * M_planet / R_planet_initial**2


# %% Planetary structure
# =============================================================================
# Phases
def phases(model, layer):
    if layer == "mantle":
        if model == "TT11":
            p = ["pd", "pv", "ppv", "pppv1"]
        elif model == "U17":
            p = ["pd", "pv", "ppv" "pppv1", "pppv2"]
    elif layer == "core":
        p = ["liquid_Fe", "solid_Fe"]
    return p


def transitions(model):
    if model == "TT11":
        t = ["pd-pv", "pv-ppv", "ppv-pppv"]
    elif model == "U17":
        t = ["pd-pv", "pv-ppv", "ppv-pppv1", "pppv1-pppv2"]
    return t


# Thermodynamic constants
def rho_0_solid(phase):
    if phase == "pd":
        rho0 = 3226
    elif phase == "pv":
        rho0 = 4109
    elif phase == "ppv":
        rho0 = 4260
    elif phase == "pppv1":
        rho0 = 4417
    elif phase == "pppv2":
        rho0 = 4579
    elif phase == "liquid_Fe":
        rho0 = 7700
    elif phase == "solid_Fe":
        rho0 = 8160
    return rho0


def K0_solid(phase):
    if phase == "pd":
        K0 = 128e9
    elif phase == "pv":
        K0 = 261e9
    elif phase == "ppv":
        K0 = 324e9
    elif phase == "pppv1":
        K0 = 402e9
    elif phase == "pppv2":
        K0 = 499e9
    elif phase == "liquid_Fe":
        K0 = 125e9
    elif phase == "solid_Fe":
        K0 = 165e9
    return K0


def K0_prime_solid(phase):
    if phase == "pd":
        K0_prime = 4.2
    elif phase == "pv":
        K0_prime = 4.0
    elif phase == "ppv":
        K0_prime = 3.3
    elif phase == "pppv1":
        K0_prime = 2.7
    elif phase == "pppv2":
        K0_prime = 2.2
    elif phase == "liquid_Fe":
        K0_prime = 5.5
    elif phase == "solid_Fe":
        K0_prime = 4.9
    return K0_prime


def gamma_0_solid(phase):
    if phase == "pd":
        gamma0 = 0.99
    elif phase == "pv":
        gamma0 = 1.0
    elif phase == "ppv":
        gamma0 = 1.48
    elif phase == "pppv1":
        gamma0 = 1.5
    elif phase == "pppv2":
        gamma0 = 1.5
    elif phase == "liquid_Fe":
        gamma0 = 1.6
    elif phase == "solid_Fe":
        gamma0 = 1.6
    return gamma0


def gamma_1_solid(phase):
    if phase == "pd":
        gamma1 = 2.1
    elif phase == "pv":
        gamma1 = 1.4
    elif phase == "ppv":
        gamma1 = 1.4
    elif phase == "pppv1":
        gamma1 = 1.4
    elif phase == "pppv2":
        gamma1 = 1.4
    elif phase == "liquid_Fe":
        gamma1 = 0.92
    elif phase == "solid_Fe":
        gamma1 = 0.92
    return gamma1


def alpha_0_solid(phase):
    if phase == "pd":
        alpha0 = 20e-6
    elif phase == "pv":
        alpha0 = 20e-6
    elif phase == "ppv":
        alpha0 = 20e-6
    elif phase == "pppv1":
        alpha0 = 20e-6
    elif phase == "pppv2":
        alpha0 = 20e-6
    elif phase == "liquid_Fe":
        alpha0 = 40e-6
    elif phase == "solid_Fe":
        alpha0 = 40e-6
    return alpha0


# Phase transitions
def DS_0_solid(layer):
    if layer == "mantle":
        DS0 = 300
    elif layer == "core":
        DS0 = 127
    return DS0


def P0_solid_trans(transition):
    if transition == "pd-pv":
        P0 = 28.3e9
    elif transition == "pv-ppv":
        P0 = 124e9
    elif transition == "ppv-pppv":
        P0 = 1060e9
    elif transition == "ppv-pppv1":
        P0 = 750e9
    elif transition == "pppv1-pppv2":
        P0 = 1300e9
    return P0


def T0_solid_trans(transition):
    if transition == "pd-pv":
        T0 = 0
    elif transition == "pv-ppv":
        T0 = 2500
    elif transition == "ppv-pppv":
        T0 = 0
    elif transition == "ppv-pppv1":
        T0 = 0
    elif transition == "pppv1-pppv2":
        T0 = 0
    return T0


def clapeyron(transition):
    if transition == "pd-pv":
        clapeyron = -2.8e6
    elif transition == "pv-ppv":
        clapeyron = 8e6
    elif transition == "ppv-pppv":
        clapeyron = -12e6
    elif transition == "ppv-pppv1":
        clapeyron = -10e6
    elif transition == "pppv1-pppv2":
        clapeyron = 6e6
    return clapeyron


def PT(T, transition):
    P = P0_solid_trans(transition) + clapeyron(transition) * (
        T - T0_solid_trans(transition)
    )
    return P


# Structure variables
def alpha_solid(rho, phase):
    x = rho_0_solid(phase) / rho
    alpha = alpha_0_solid(phase) * x**3
    return alpha


def grun_th_solid(rho, phase):
    x = rho_0_solid(phase) / rho
    gamma = gamma_0_solid(phase) * x ** gamma_1_solid(phase)
    return gamma


def KT_solid(rho, phase):
    theta = 3 / 2 * (K0_prime_solid(phase) - 1)
    x = rho_0_solid(phase) / rho
    KT = (
        K0_solid(phase)
        * x ** (-2 / 3)
        * (1 + (1 + theta * x ** (1 / 3)) * (1 - x ** (1 / 3)))
        * np.exp(theta * (1 - x ** (1 / 3)))
    )
    return KT


def KT_prime_solid(rho, phase):
    theta = 3 / 2 * (K0_prime_solid(phase) - 1)
    x = rho_0_solid(phase) / rho
    KT_prime = (1 / 3) * (
        2
        + theta * x ** (1 / 3)
        + (2 * x ** (1 / 3) + 2 * theta * x ** (2 / 3) - x ** (1 / 3) * (1 + theta))
        / (x ** (1 / 3) + (1 - x ** (1 / 3)) * (theta * x ** (1 / 3) + 2))
    )
    return KT_prime


def KS_solid(rho, T, phase):
    KS = KT_solid(rho, phase) * (
        1 + alpha_solid(rho, phase) * grun_th_solid(rho, phase) * T
    )
    return KS


def CP_solid(rho, T, phase):
    Cp = (alpha_solid(rho, phase) * KS_solid(rho, T, phase)) / (
        grun_th_solid(rho, phase) * rho
    )
    return Cp


def entropy_fusion(rho, layer):
    if layer == "core":
        DS = DS_0_solid(layer) * np.ones(len(rho))
    elif layer == "mantle":
        DS = DS_0_solid(layer) * np.ones(len(rho))
    return DS


# Variable derivatives
def mass(radius, rho):
    dmdr = 4 * np.pi * radius**2 * rho
    return dmdr


def gauss(radius, rho, m):
    dgdr = 4 * np.pi * G * rho - 2 * G * m / radius**3
    return dgdr


def hydrostatic(rho, g):
    dPdr = -rho * g
    return dPdr


def compressibility(rho, g, T, phase):
    drhodr = -(rho**2) * g / KS_solid(rho, T, phase)
    return drhodr


def adiabat(rho, g, T, phase):
    dTdr = -rho * g * grun_th_solid(rho, phase) * T / KS_solid(rho, T, phase)
    return dTdr


# Equation system
def structure_system(radius, z, phase):
    rho, m, g, P, T = z
    drhodr = compressibility(rho, g, T, phase)
    dmdr = mass(radius, rho)
    dgdr = gauss(radius, rho, m)
    dPdr = hydrostatic(rho, g)
    dTdr = adiabat(rho, g, T, phase)
    dYdr = [drhodr, dmdr, dgdr, dPdr, dTdr]
    return dYdr


# %% Liquidus
# =============================================================================
def liquidus_core(P):
    T = (6500 * (P / 340e9) ** 0.515) / (1 - np.log(0.87))
    return T


def solidus_mantle(P):
    T = 5400 * (P / 140e9) ** 0.48 / (1 - np.log(0.79))
    return T


# %% Structure integration
# =============================================================================
# Events
def phase_change(radius, z, transition):
    rho, m, g, P, T = z
    stop = P - (
        P0_solid_trans(transition)
        + clapeyron(transition) * (T - T0_solid_trans(transition))
    )
    return stop


def core_crystallization(radius, z):
    rho, m, g, P, T = z
    stop = T - liquidus_core(P)
    return stop


def CMB(radius, z):
    rho, m, g, P, T = z
    stop = CMF - m / M_planet
    return stop


def center(radius, z):
    rho, m, g, P, T = z
    stop = m
    return stop


# Continuity
def continuity(rho, sol_i, phase, l):
    # KT (bottom)
    theta_bot = 3 / 2 * (K0_prime_solid(phase[l]) - 1)
    x_bot = rho_0_solid(phase[l]) / rho
    KT_bot = (
        K0_solid(phase[l])
        * x_bot ** (-2 / 3)
        * (1 + (1 + theta_bot * x_bot ** (1 / 3)) * (1 - x_bot ** (1 / 3)))
        * np.exp(theta_bot * (1 - x_bot ** (1 / 3)))
    )
    # KT (top)
    KT_top = KT_solid(sol_i.y[0, -1], phase[l - 1])
    return KT_bot - KT_top


# Solver
def structure_solver(x, T_CMB_solidus=True, T_CMB=0, R0=1e-3):
    # Warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Independent variables
    R_planet = x

    # Integration
    sol, phase, layer = [], [], []
    l, l_m, l_c = 0, 0, 0
    while True:
        # Phases
        if l == 0:
            layer.append("mantle")
            phase.append(phases(model_phase, layer[l])[l])
            R_CMB = np.nan
            R_ICB = np.nan
        else:
            if event_phase == 1:
                layer.append("mantle")
                phase.append(phases(model_phase, layer[l])[l_m])
            if event_CMB == 1:
                layer.append("core")
                P_CMB = sol_i.y[3, -1]
                if T_CMB_solidus:
                    T_core_status = solidus_mantle(P_CMB)
                else:
                    T_core_status = T_CMB
                if T_core_status > liquidus_core(P_CMB):
                    phase.append(phases(model_phase, layer[l])[0])
                else:
                    phase.append(phases(model_phase, layer[l])[1])
                    R_ICB = sol_i.t_events[1][0]
                R_CMB = sol_i.t_events[1][0]
            if event_core == 1:
                layer.append("core")
                phase.append(phases(model_phase, layer[l])[1])
                R_ICB = sol_i.t_events[2][0]

        # Transition
        if l_m < len(transitions(model_phase)):
            transition = transitions(model_phase)[l_m]

        # Initial conditions
        if l == 0:
            R_start = R_planet
            Y0 = [
                rho_0_solid(phase[l]),
                M_planet,
                G * M_planet / R_planet**2,
                P_surface,
                T_surface,
            ]
        else:
            # Continuity
            rho_start = optimize.root(
                lambda rho: continuity(rho, sol_i, phase, l), sol_i.y[0, -1]
            ).x[0]

            # Definition
            R_start = sol_i.t[-1]
            if event_CMB == 1:
                P_CMB = sol_i.y[3, -1]
                if T_CMB_solidus:
                    T_start = solidus_mantle(P_CMB)
                else:
                    T_start = T_CMB
            else:
                T_start = sol_i.y[4, -1]
            Y0 = [
                rho_start,
                sol_i.y[1, -1],
                sol_i.y[2, -1],
                sol_i.y[3, -1],
                T_start,
            ]

        # Radius
        r_span = [R_start, R0]

        # Events definition
        phase_lbd = lambda r, z: phase_change(r, z, transition)
        CMB_lbd = lambda r, z: CMB(r, z)
        core_lbd = lambda r, z: core_crystallization(r, z)
        center_lbd = lambda r, z: center(r, z)

        if layer[l] == "mantle":
            if l_m < len(transitions(model_phase)):
                phase_lbd.terminal = True
                CMB_lbd.terminal = True
                core_lbd.terminal = False
                center_lbd.terminal = True
            else:
                phase_lbd.terminal = False
                CMB_lbd.terminal = True
                core_lbd.terminal = False
                center_lbd.terminal = True
        elif layer[l] == "core":
            if phase[l] == "liquid_Fe":
                phase_lbd.terminal = False
                CMB_lbd.terminal = False
                core_lbd.terminal = True
                center_lbd.terminal = True
            elif phase[l] == "solid_Fe":
                phase_lbd.terminal = False
                CMB_lbd.terminal = False
                core_lbd.terminal = False
                center_lbd.terminal = True

        # Solver
        r_tol = 1e-3
        while True:
            try:
                sol_i = integrate.solve_ivp(
                    lambda r, z: structure_system(r, z, phase[l]),
                    r_span,
                    Y0,
                    dense_output=True,
                    method="RK45",
                    events=[phase_lbd, CMB_lbd, core_lbd, center_lbd],
                    rtol=r_tol,
                    atol=1e-6,
                    max_step=1.0e3,
                )
            except ValueError:
                r_tol *= 1e-1
                continue
            break

        # Concatenation
        sol.append(sol_i)

        # Events resolution
        if center_lbd.terminal:
            event_center = np.size(sol_i.t_events[3])

        if sol_i.status == 0 or event_center == 1 or sol_i.t[-1] < R0:
            break

        if phase_lbd.terminal:
            event_phase = np.size(sol_i.t_events[0])
        else:
            event_phase = 0
        if CMB_lbd.terminal:
            event_CMB = np.size(sol_i.t_events[1])
        else:
            event_CMB = 0
        if core_lbd.terminal:
            event_core = np.size(sol_i.t_events[2])
        else:
            event_core = 0

        if layer[l] == "mantle":
            l_m += 1
        if layer[l] == "core":
            l_c += 1

        # Iteration
        l += 1
    return sol, layer, phase, R_CMB, R_ICB


# Planetary radius search
def radius_search(R_planet_initial, R_c, T_CMB_solidus=True, T_CMB=0, verbose=False):
    c = 0
    R_min, R_max = 0, R_planet_initial
    R_search, r_search = np.array([]), np.array([])
    while True:
        if R_max == R_planet_initial:
            sol, layer, phase, R_CMB, R_ICB = structure_solver(
                R_max, T_CMB_solidus, T_CMB
            )
            c += 1
            if verbose:
                print("Iteration %d: R = %.2f km" % (c, R_max / 1e3))
            R_search = np.append(R_search, R_max)
            r_search = np.append(r_search, sol[-1].t[-1])
            Ri = sol[-1].t[-1]
            gi = sol[-1].y[2]
            g_max = 2 * g_planet_initial

        if Ri < R_c and np.max(gi) < g_max:
            R_planet = R_max
            break

        if np.max(gi) < g_max:
            R = [R_min, R_max]
            while True:
                R[0] = R[1]
                R[1] = 0.5 * (R_min + R[1])
                sol, layer, phase, R_CMB, R_ICB = structure_solver(
                    R[1], T_CMB_solidus, T_CMB
                )
                c += 1
                if verbose:
                    print("Iteration %d: R = %.2f km" % (c, R[1] / 1e3))
                R_search = np.append(R_search, R[1])
                r_search = np.append(r_search, sol[-1].t[-1])
                Ri = sol[-1].t[-1]
                gi = sol[-1].y[2]
                if Ri < R_c or np.max(gi) > g_max:
                    R_max = R[0]
                    R_min = R[1]
                    break

        if Ri < R_c and np.max(gi) < g_max:
            R_planet = R_min
            break

        if np.max(gi) > g_max:
            R = [R_min, R_max]
            while True:
                R[1] = R[0]
                R[0] = R[0] + radius_search_coef * (R_max - R[0])
                sol, layer, phase, R_CMB, R_ICB = structure_solver(
                    R[0], T_CMB_solidus, T_CMB
                )
                c += 1
                if verbose:
                    print("Iteration %d: R = %.2f km" % (c, R[0] / 1e3))
                R_search = np.append(R_search, R[0])
                r_search = np.append(r_search, sol[-1].t[-1])
                Ri = sol[-1].t[-1]
                gi = sol[-1].y[2]
                if np.max(gi) < g_max:
                    R_max = R[0]
                    R_min = R[1]
                    break
    return R_planet, R_search, r_search


# %% Structure preprocessing
# =============================================================================
def structure_processing(sol, layer, phase, T_CMB, R_CMB, R_ICB):
    radius, radius_lim, rho, m, g, P, T, alpha, gamma, KT, KT_prime, KS, Cp, DS = (
        [] for i in range(14)
    )
    for i in range(len(sol)):
        radius_i = sol[i].t
        radius_lim_i = [radius_i[0], radius_i[-1]]

        rho_i = sol[i].y[0]
        m_i = sol[i].y[1]
        g_i = sol[i].y[2]
        P_i = sol[i].y[3]
        T_i = sol[i].y[4]

        alpha_i = alpha_solid(rho_i, phase[i])
        gamma_i = grun_th_solid(rho_i, phase[i])
        KT_i = KT_solid(rho_i, phase[i])
        KT_prime_i = KT_prime_solid(rho_i, phase[i])
        KS_i = KS_solid(rho_i, T_i, phase[i])
        DS_i = entropy_fusion(rho_i, layer[i])
        Cp_i = CP_solid(rho_i, T_i, phase[i])

        radius = np.append(radius, radius_i)
        radius_lim = np.append(radius_lim, radius_lim_i)
        rho = np.append(rho, rho_i)
        m = np.append(m, m_i)
        g = np.append(g, g_i)
        P = np.append(P, P_i)
        T = np.append(T, T_i)
        alpha = np.append(alpha, alpha_i)
        gamma = np.append(gamma, gamma_i)
        KT = np.append(KT, KT_i)
        KT_prime = np.append(KT_prime, KT_prime_i)
        KS = np.append(KS, KS_i)
        Cp = np.append(Cp, Cp_i)
        DS = np.append(DS, DS_i)

    radius = np.flip(radius)
    radius_lim = np.flip(radius_lim)
    rho = np.flip(rho)
    m = np.flip(m)
    g = np.flip(g)
    P = np.flip(P)
    T = np.flip(T)
    alpha = np.flip(alpha)
    gamma = np.flip(gamma)
    KT = np.flip(KT)
    KT_prime = np.flip(KT_prime)
    KS = np.flip(KS)
    Cp = np.flip(Cp)
    DS = np.flip(DS)

    class Output:
        def __init__(self):
            self.layer = layer
            self.phase = phase
            self.radius_lim = radius_lim

            self.T_CMB = T_CMB
            self.R_CMB = R_CMB
            self.R_ICB = R_ICB

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

    return Output()


# %% MPI Execution
# =============================================================================
# Main function
def structure_SE(T_CMB):
    print("T_CMB = %d K" % T_CMB)
    R_planet_0, R_search, r_search = radius_search(
        R_planet_initial_coef * R_planet_initial,
        R_crit,
        T_CMB_solidus=False,
        T_CMB=T_CMB,
        verbose=False,
    )
    sol, layer, phase, R_CMB, R_ICB = structure_solver(
        R_planet_0, T_CMB_solidus=False, T_CMB=T_CMB
    )
    S = structure_processing(sol, layer, phase, T_CMB, R_CMB, R_ICB)
    return S.__dict__


# Saving directory
os.makedirs(output_dir, exist_ok=True)

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
    print("T_CMB range: %d - %d K" % (T_CMB_search[0], T_CMB_search[-1]))

    with MPIPoolExecutor() as pool:
        S = np.zeros(len(T_CMB_search), dtype=object)
        for i, result in enumerate(pool.map(structure_SE, T_CMB_search)):
            with open(output_dir + "/T_CMB_%05d.pkl" % T_CMB_search[i], "wb") as f:
                pkl.dump(Struct(**result), f)

    print("Done.")

    toc = time.time() - tic
    print("Elapsed time: %.1f s" % toc)
