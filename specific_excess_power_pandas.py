#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:38:50 2023

@author: data-scientist
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =============================================================================
# ADRpy module
# pip install ADRPy module
# =============================================================================

from ADRpy import atmospheres as at

# =============================================================================
# Define Constants SI units
# =============================================================================

g = 9.80665  # m/sec Sq
a0 = 340.294  # m/Sec

p0 = 101325  # N/sqm or Pascals
gamma = 1.4  # ratio of specific heat

# =============================================================================
# Define Constants Imperial units
# =============================================================================

gc = 32.17  # lbm/slug
R = 96.93  # ft-lb/lb-K

# =============================================================================
# Import standard atmosphere
# =============================================================================

isa = at.Atmosphere()

# =============================================================================
# Create all definitions
# Reference Para 5.5.1 of USAFTPS FTM 108 dated 30 Sep 22
# =============================================================================


def knot2mps(airspeed):
    """Convert airspeed from knots to meters per second."""
    return airspeed * 1852 / 3600


def ft2meters(distance):
    """Convert distance from feet to meters."""
    return distance * 0.3048


def Qc(calibrated_airspeed):
    """
    Convert Calibrated Air Speed in mps to Dynamic Pressure in Pascals.

    (Eq 5.25)
    """
    return p0 * (((1 + 0.2 * (calibrated_airspeed / a0) ** 2) ** 3.5) - 1)


def Pa(altitude):
    """
    Convert altitude in feet to Static Pressure in Pascals.

    (Eq 5.26)
    """
    return p0 * ((1 - (6.875585599999999e-06 * altitude)) ** 5.255863)


def Mach(impact_pressure, ambient_pressure):
    """
    Convert Impact Pressure and Static Pressure (same units) to Mach Number.

    (Eq 5.27)
    """
    return math.sqrt(
        (2 / (gamma - 1))
        * (
            ((impact_pressure / ambient_pressure) + 1) ** ((gamma - 1) / gamma)
            - 1
        )
    )


def TAS(mach, ambient_temperature):
    """
    Convert Mach and Ambient Temp to TAS(mps).

    (Eq 5.30)
    """
    return mach * math.sqrt(gamma * gc * R * ambient_temperature) * 0.3048


def Eh(altitude, true_airspeed):
    """
    Convert altitude in meters and TAS in mps to Energy Height.

    (Eq 5.7)
    """
    return altitude + (true_airspeed**2 / (2 * 9.81))


def Ps(energy_height, time):
    """
    Find the derivative of energy height in meters with respect to time in seconds.

    (pg 5.30)
    """
    return np.gradient(energy_height, time)


# =============================================================================
# Define constants and assign variables for Standard Day Correction
# =============================================================================

EmpWt = 9000
Ta_std_ssl = 288
delThrust = 0  # Equation 5.36 USNTPS FTM 108
delDrag = 0  # Equation 5.36 USNTPS FTM 108
Wstd = 11000

# =============================================================================
# Define functions for Standard Day Correction
# =============================================================================


def Vtstd(
    true_airspeed_test, ambient_temperature_test, ambient_temperature_std
):
    """
    Calculate Standard TAS from Test Values.

    (Eq 5.33)
    """
    return true_airspeed_test * (
        math.sqrt(ambient_temperature_std / ambient_temperature_test)
    )


def climb_angle(rate_of_climb, true_airspeed):
    """
    Calculate CLimb Angle from ROC and TAS both in meters per second.

    (Eq 5.13)
    """
    return math.asin(rate_of_climb / true_airspeed)


def VbyH(airspeed, height):
    """
    Calculate dV/dH from airspeed in meters per second and height in meters.

    In support of climb correction factor
    """
    dH_raw = np.gradient(height)
    dH = np.where(dH_raw != 0.0, dH_raw, np.nan)  # avoid divide by zero
    return np.gradient(airspeed) / dH


def CCF(true_airspeed_std, height):
    """
    Calculate Climb Correction Factor.

    (Eq 5.50)
    """
    return 1 + ((true_airspeed_std / g) * VbyH(true_airspeed_std, height))


def Psstd(
    specific_excess_power,
    weight_test,
    weight_std,
    true_airspeed_test,
    true_airspeed_std,
    delta_net_thrust_parallel_flight_path,
    delta_drag,
):
    """Calculate SEP Standard from Test Values (Eq 5.51)."""
    return (
        specific_excess_power
        * (weight_test / weight_std)
        * (true_airspeed_std / true_airspeed_test)
    ) + (
        (true_airspeed_std / weight_std)
        * (delta_net_thrust_parallel_flight_path - delta_drag)
    )


def climb_angle_std(
    specific_excess_power_std, climb_correction_factor, true_airspeed_std
):
    """
    Calculate standardized climb angle from Ps_std, CCF, Vt_std.

    (Eq 5.52) & (Eq 5.53)
    """
    return math.asin(
        (specific_excess_power_std / climb_correction_factor)
        / true_airspeed_std
    )


# =============================================================================
# Create the dataframe and sub-dataframes
# =============================================================================

# =============================================================================
# To modify this code for your own data:
#   Change the file name (Excel) to match your file
#   Change the column names in the Code
#   Ensure the units are as given in the function definitions
#   Determine row index values to match your test points
#   Rename your variables to match your use (use find/replace in your editor)
# =============================================================================

# select the rows of 1st Level Accel and store in a new data frame
df_500 = pd.read_excel("level_accel_500_clean.xlsx")
df_10k = pd.read_excel("level_accel_10k_clean.xlsx")
df_20k = pd.read_excel("level_accel_20k_clean.xlsx")
df_30k = pd.read_excel("level_accel_30k_clean.xlsx")

# =============================================================================
# Compute the SEP from CAS and Altitude
# This assumes that Mach is not available
# =============================================================================

# Use .apply() on a column; provide it to a function w/ a single input argument

df_500["ASL_m"] = df_500["ASL"].apply(ft2meters)
df_10k["ASL_m"] = df_10k["ASL"].apply(ft2meters)
df_20k["ASL_m"] = df_20k["ASL"].apply(ft2meters)
df_30k["ASL_m"] = df_30k["ASL"].apply(ft2meters)

df_500["IAS_mps"] = df_500["IAS"].apply(knot2mps)
df_10k["IAS_mps"] = df_10k["IAS"].apply(knot2mps)
df_20k["IAS_mps"] = df_20k["IAS"].apply(knot2mps)
df_30k["IAS_mps"] = df_30k["IAS"].apply(knot2mps)

df_500["Qc"] = df_500["IAS_mps"].apply(Qc)
df_10k["Qc"] = df_10k["IAS_mps"].apply(Qc)
df_20k["Qc"] = df_20k["IAS_mps"].apply(Qc)
df_30k["Qc"] = df_30k["IAS_mps"].apply(Qc)

df_500["Pa"] = df_500["ASL"].apply(Pa)
df_10k["Pa"] = df_10k["ASL"].apply(Pa)
df_20k["Pa"] = df_20k["ASL"].apply(Pa)
df_30k["Pa"] = df_30k["ASL"].apply(Pa)

# Many functions can accept a dataframe column (Series) and return a column

df_500["Ta"] = isa.airtemp_k(df_500["ASL_m"])
df_10k["Ta"] = isa.airtemp_k(df_10k["ASL_m"])
df_20k["Ta"] = isa.airtemp_k(df_20k["ASL_m"])
df_30k["Ta"] = isa.airtemp_k(df_30k["ASL_m"])

# Use a lambda function on the dataframe for more than one column

df_500["MACH"] = df_500.apply(lambda x: Mach(x["Qc"], x["Pa"]), axis=1)
df_10k["MACH"] = df_10k.apply(lambda x: Mach(x["Qc"], x["Pa"]), axis=1)
df_20k["MACH"] = df_20k.apply(lambda x: Mach(x["Qc"], x["Pa"]), axis=1)
df_30k["MACH"] = df_30k.apply(lambda x: Mach(x["Qc"], x["Pa"]), axis=1)

df_500["TAS_mps"] = df_500.apply(lambda x: TAS(x["MACH"], x["Ta"]), axis=1)
df_10k["TAS_mps"] = df_10k.apply(lambda x: TAS(x["MACH"], x["Ta"]), axis=1)
df_20k["TAS_mps"] = df_20k.apply(lambda x: TAS(x["MACH"], x["Ta"]), axis=1)
df_30k["TAS_mps"] = df_30k.apply(lambda x: TAS(x["MACH"], x["Ta"]), axis=1)

df_500["Eh"] = df_500.apply(lambda x: Eh(x["ASL_m"], x["TAS_mps"]), axis=1)
df_10k["Eh"] = df_10k.apply(lambda x: Eh(x["ASL_m"], x["TAS_mps"]), axis=1)
df_20k["Eh"] = df_20k.apply(lambda x: Eh(x["ASL_m"], x["TAS_mps"]), axis=1)
df_30k["Eh"] = df_30k.apply(lambda x: Eh(x["ASL_m"], x["TAS_mps"]), axis=1)

df_500["Ps"] = savgol_filter(
    np.gradient(df_500["Eh"], df_500["Time"]), len(df_500) // 5, 2
)
df_10k["Ps"] = savgol_filter(
    np.gradient(df_10k["Eh"], df_10k["Time"]), len(df_10k) // 5, 2
)
df_20k["Ps"] = savgol_filter(
    np.gradient(df_20k["Eh"], df_20k["Time"]), len(df_20k) // 5, 2
)
df_30k["Ps"] = savgol_filter(
    np.gradient(df_30k["Eh"], df_30k["Time"]), len(df_30k) // 5, 2
)


# =============================================================================
# Standard Day Correction
# =============================================================================

df_500["Vtstd"] = df_500.apply(
    lambda x: Vtstd(x["TAS_mps"], x["Ta"], Ta_std_ssl), axis=1
)
df_10k["Vtstd"] = df_10k.apply(
    lambda x: Vtstd(x["TAS_mps"], x["Ta"], Ta_std_ssl), axis=1
)
df_20k["Vtstd"] = df_20k.apply(
    lambda x: Vtstd(x["TAS_mps"], x["Ta"], Ta_std_ssl), axis=1
)
df_30k["Vtstd"] = df_30k.apply(
    lambda x: Vtstd(x["TAS_mps"], x["Ta"], Ta_std_ssl), axis=1
)

df_500["ROC"] = np.gradient(df_500["ASL_m"], df_500["Time"])
df_10k["ROC"] = np.gradient(df_10k["ASL_m"], df_10k["Time"])
df_20k["ROC"] = np.gradient(df_20k["ASL_m"], df_20k["Time"])
df_30k["ROC"] = np.gradient(df_30k["ASL_m"], df_30k["Time"])

df_500["cl_ang_test"] = df_500.apply(
    lambda x: climb_angle(x["ROC"], x["TAS"]), axis=1
)
df_10k["cl_ang_test"] = df_10k.apply(
    lambda x: climb_angle(x["ROC"], x["TAS"]), axis=1
)
df_20k["cl_ang_test"] = df_20k.apply(
    lambda x: climb_angle(x["ROC"], x["TAS"]), axis=1
)
df_30k["cl_ang_test"] = df_30k.apply(
    lambda x: climb_angle(x["ROC"], x["TAS"]), axis=1
)

df_500["CCF"] = CCF(df_500["Vtstd"], df_500["ASL_m"])
df_10k["CCF"] = CCF(df_10k["Vtstd"], df_10k["ASL_m"])
df_20k["CCF"] = CCF(df_20k["Vtstd"], df_20k["ASL_m"])
df_30k["CCF"] = CCF(df_30k["Vtstd"], df_30k["ASL_m"])

df_500["Wtest"] = EmpWt + df_500["fuel"]
df_10k["Wtest"] = EmpWt + df_10k["fuel"]
df_20k["Wtest"] = EmpWt + df_20k["fuel"]
df_30k["Wtest"] = EmpWt + df_30k["fuel"]

df_500["Ps_std"] = df_500.apply(
    lambda x: Psstd(
        x["Ps"],
        x["Wtest"],
        Wstd,
        x["TAS_mps"],
        x["Vtstd"],
        delThrust,
        delDrag,
    ),
    axis=1,
)
df_10k["Ps_std"] = df_10k.apply(
    lambda x: Psstd(
        x["Ps"],
        x["Wtest"],
        Wstd,
        x["TAS_mps"],
        x["Vtstd"],
        delThrust,
        delDrag,
    ),
    axis=1,
)
df_20k["Ps_std"] = df_20k.apply(
    lambda x: Psstd(
        x["Ps"],
        x["Wtest"],
        Wstd,
        x["TAS_mps"],
        x["Vtstd"],
        delThrust,
        delDrag,
    ),
    axis=1,
)
df_30k["Ps_std"] = df_30k.apply(
    lambda x: Psstd(
        x["Ps"],
        x["Wtest"],
        Wstd,
        x["TAS_mps"],
        x["Vtstd"],
        delThrust,
        delDrag,
    ),
    axis=1,
)

df_500["cl_ang_std"] = df_500.apply(
    lambda x: climb_angle_std(
        x["Ps_std"],
        x["CCF"],
        x["Vtstd"],
    ),
    axis=1,
)
df_10k["cl_ang_std"] = df_10k.apply(
    lambda x: climb_angle_std(
        x["Ps_std"],
        x["CCF"],
        x["Vtstd"],
    ),
    axis=1,
)
df_20k["cl_ang_std"] = df_20k.apply(
    lambda x: climb_angle_std(
        x["Ps_std"],
        x["CCF"],
        x["Vtstd"],
    ),
    axis=1,
)
df_30k["cl_ang_std"] = df_30k.apply(
    lambda x: climb_angle_std(
        x["Ps_std"],
        x["CCF"],
        x["Vtstd"],
    ),
    axis=1,
)

# =============================================================================
# End of calculations
# begining of visualisation
# =============================================================================

f0 = plt.figure()
plt.plot(df_500["MACH"], df_500["Ps"])
plt.grid()
plt.xlabel("Mach")
plt.ylabel("Sp Excess Power")
plt.legend(["500ft"])
plt.title("Specific Excess Power Variation with Mach (DCS:F16")
plt.savefig("Ps_500ft.png", dpi=300)

# =============================================================================
#
# =============================================================================

f1 = plt.figure()
plt.plot(df_10k["MACH"], df_10k["Ps"])
plt.grid()
plt.xlabel("Mach")
plt.ylabel("Sp Excess Power")
plt.legend(["10kft"])
plt.title("Specific Excess Power Variation with Mach (DCS:F16")
plt.savefig("Ps_10kft.png", dpi=300)

# =============================================================================
#
# =============================================================================

f2 = plt.figure()
plt.plot(df_20k["MACH"], df_20k["Ps"])
plt.grid()
plt.xlabel("Mach")
plt.ylabel("Sp Excess Power")
plt.legend(["20kft"])
plt.title("Specific Excess Power Variation with Mach (DCS:F16")
plt.savefig("Ps_20kft.png", dpi=300)

# =============================================================================
#
# =============================================================================

f3 = plt.figure()
plt.plot(df_30k["MACH"], df_30k["Ps"])
plt.grid()
plt.xlabel("Mach")
plt.ylabel("Sp Excess Power")
plt.legend(["30kft"])
plt.title("Specific Excess Power Variation with Mach (DCS:F16")
plt.savefig("Ps_30kft.png", dpi=300)

# =============================================================================
#
# =============================================================================

f4 = plt.figure()
plt.plot(df_500["MACH"], df_500["Ps"])
plt.plot(df_10k["MACH"], df_10k["Ps"])
plt.plot(df_20k["MACH"], df_20k["Ps"])
plt.plot(df_30k["MACH"], df_30k["Ps"])
plt.grid()
plt.xlabel("Mach")
plt.ylabel("Sp Excess Power")
plt.legend(["500ft", "10kft", "20kft", "30kft"])
plt.title("Variation of Specific Excess Power with Altitude (DCS:F16")
plt.savefig("Ps_Altitude.png", dpi=300)

# =============================================================================
#
# =============================================================================

f5 = plt.figure()
plt.plot(df_500["MACH"], df_500["Ps_std"], "--")
plt.plot(df_10k["MACH"], df_10k["Ps_std"], "--")
plt.plot(df_20k["MACH"], df_20k["Ps_std"], "--")
plt.plot(df_30k["MACH"], df_30k["Ps_std"], "--")
plt.grid()
plt.xlabel("Mach")
plt.ylabel("Sp Excess Power")
plt.legend(["500ft", "10kft", "20kft", "30kft"])
plt.title("Variation of Standard SEP  with Altitude (DCS:F16")
plt.savefig("Standard_Ps_Altitude.png", dpi=300)

# =============================================================================
#
# =============================================================================

f6 = plt.figure()
plt.plot(df_10k["MACH"], df_10k["Ps"])
plt.plot(df_10k["MACH"], df_10k["Ps_std"], "--")
plt.grid()
plt.xlabel("Mach")
plt.ylabel("Sp Excess Power")
plt.legend(["Ps_test", "Ps_std"])
plt.title("Comparison of Ps Std and Ps Test,Alt:10kft(DCS:F16)")
plt.savefig("Comparison_Ps_Std_Ps_Test_10kft.png", dpi=300)
