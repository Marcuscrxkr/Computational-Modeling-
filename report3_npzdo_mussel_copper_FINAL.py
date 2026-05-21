"""
Report 3 — Impact of Copper Toxicity on Blue Mussel Biofiltration Capacity
DTU Course 25314 — Computational Marine Ecological Modelling
Marc, May 2025

Coupled 1-D NPZDO water-column model with Dynamic Energy Budget (DEB)
sub-model for Mytilus spp. and an external Cu2+ toxicity forcing.

Three scenarios: control (2 µg/L background), low Cu (42 µg/L pulse),
high Cu (82 µg/L pulse). Pulse starts day 60, lasts 30 days, then decays.
Saves 8 figures as PNG files.

References: Buer et al. (2020), Maar et al. (2015, 2018),
            van der Veer et al. (2006), Kooijman (2010),
            Fasham et al. (1990), Viarengo et al. (1999)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# --- Grid -------------------------------------------------------------------
z_Bottom = 20.0          # water column depth [m]
dz       = 1.0           # cell thickness [m]
nz       = int(z_Bottom / dz)
z_cell   = np.linspace(dz / 2, z_Bottom - dz / 2, nz)   # cell centres [m]
z_face   = np.linspace(0, z_Bottom, nz + 1)              # cell faces [m]

# --- Light ------------------------------------------------------------------
kw          = 0.45       # seawater attenuation [m⁻¹] (turbid Kattegat, CDOM)
kp          = 0.05       # phytoplankton self-shading [m² mmol N⁻¹]
L0_summer   = 400.0      # surface PAR, summer [µmol photons m⁻² s⁻¹]
L0_winter   = 10.0       # surface PAR, winter [µmol photons m⁻² s⁻¹]
kL          = 30.0       # light half-saturation [µmol photons m⁻² s⁻¹]

# --- Mixing -----------------------------------------------------------------
kappa_top    = 50.0      # peak winter diffusivity [m² day⁻¹]
kappa_bottom = 1.0       # summer thermocline diffusivity [m² day⁻¹]
z_Mix        = 10.0      # summer mixed-layer depth [m]
zeta_mix     = 2.0       # pycnocline width [m]
z_MixWinter  = 18.0      # winter mixed-layer depth [m]

# --- Nutrients --------------------------------------------------------------
N_Bottom  = 10.0         # bottom boundary nutrient concentration [mmol N m⁻³]
kappa_N   = 10.0         # nutrient exchange velocity at bottom [m day⁻¹]

# --- Phytoplankton ----------------------------------------------------------
g_Pmax = 1.5             # max growth rate [day⁻¹]
kN     = 0.5             # nutrient half-saturation [mmol N m⁻³]
m_P    = 0.05            # linear mortality [day⁻¹] (Fasham et al. 1990)
w_P    = 0.0             # sinking velocity [m day⁻¹]

# --- Zooplankton ------------------------------------------------------------
g_Zmax = 0.5             # max grazing rate [day⁻¹]
k_Z    = 0.5             # grazing half-saturation [mmol N m⁻³] (~40 mg C m⁻³)
eps_N  = 0.3             # fraction of ingested food excreted as DIN
eps_D  = 0.3             # fraction egested as detritus
m_Z    = 0.2             # quadratic mortality [(mmol N m⁻³)⁻¹ day⁻¹]

# --- Detritus ---------------------------------------------------------------
tau = 0.15               # remineralisation rate [day⁻¹]
w_D = 2.0                # sinking velocity [m day⁻¹]

# --- Oxygen -----------------------------------------------------------------
gamma_O  = 8.625         # O₂:N ratio from Redfield stoichiometry [mmol O₂ mmol N⁻¹]
O2_sat   = 300.0         # atmospheric saturation [mmol O₂ m⁻³]
k_piston = 1.0           # air-sea gas exchange velocity [m day⁻¹]
k_anox   = 0.5           # anoxia half-saturation for remineralisation [mmol O₂ m⁻³]

# --- Initial conditions (winter, pre-bloom) ---------------------------------
N_init_val = 10.0        # [mmol N m⁻³]
P_init_val = 0.05        # [mmol N m⁻³]
Z_init_val = 0.02        # [mmol N m⁻³]
D_init_val = 0.5         # [mmol N m⁻³]
O_init_val = O2_sat      # [mmol O₂ m⁻³]

# --- DEB parameters (Buer et al. 2020; Maar et al. 2015) -------------------
Lm        = 15.0         # maximum structural length [cm]
kappa_deb = 0.7          # fraction of mobilised energy to soma (κ-rule)
v_dot     = 0.05         # energy conductance [cm day⁻¹]
p_M       = 0.002        # somatic maintenance cost [day⁻¹]

# Arrhenius temperature correction (van der Veer et al. 2006; Maar et al. 2018)
T_A   = 5800.0           # Arrhenius temperature [K]
T_ref = 293.0            # reference temperature, 20°C [K]
T_AL  = 45000.0          # low-temperature Arrhenius parameter [K]
T_AH  = 30000.0          # high-temperature Arrhenius parameter [K]
T_L   = 278.0            # lower boundary, 5°C [K]
T_H   = 298.0            # upper boundary, 25°C [K]

# Ingestion (Holling Type II on chlorophyll-a as food proxy)
F_max     = 0.10         # max filtration rate at Lm [m³ ind⁻¹ day⁻¹]
k_food    = 0.5          # ingestion half-saturation [mg chl-a m⁻³]
N_to_chla = 1.5          # mmol N m⁻³ → mg chl-a m⁻³ (N:Chl ≈ 8 gN gChl⁻¹)

# Population
n_mussel     = 500.0     # mussel density [ind m⁻²] (suspended longline)
z_mussel_idx = nz // 2  # cell index for mussel layer (~10 m)

# Initial mussel state
L_init = 2.0             # shell length [cm]
e_init = 0.5             # scaled reserve density [-]

# --- Copper parameters (Viarengo et al. 1999; Wang & Rainbow 2005) ----------
EC50_Cu       = 50.0     # half-inhibition concentration [µg Cu L⁻¹]
n_hill        = 2.0      # Hill coefficient (sigmoidal steepness)
Cu_background = 2.0      # ambient background [µg Cu L⁻¹] (Förstner & Wittmann 1981)
Cu_pulse_day  = 60.0     # pulse start [day]
Cu_pulse_dur  = 30.0     # pulse duration [days]
Cu_decay_rate = 0.05     # exponential decay constant [day⁻¹]

# --- Time -------------------------------------------------------------------
t_end  = 365.0
t_eval = np.arange(0.0, t_end + 1.0, 1.0)


# --- Forcing functions ------------------------------------------------------

def seasonal_surface_light(t):
    """Surface PAR, sinusoidal seasonal cycle [µmol photons m⁻² s⁻¹]."""
    doy = t % 365.0
    L0 = (L0_winter
          + 0.5 * (L0_summer - L0_winter)
          * (1.0 + np.sin(2.0 * np.pi * (doy - 80.0) / 365.0)))
    return max(L0, 5.0)


def compute_light_profile(P, t):
    """Beer-Lambert attenuation with phytoplankton self-shading [µmol photons m⁻² s⁻¹]."""
    L0     = seasonal_surface_light(t)
    L_z    = np.zeros(nz)
    cum_kp = 0.0
    for i in range(nz):
        L_z[i] = L0 * np.exp(-(kw * z_face[i] + kp * cum_kp))
        cum_kp += P[i] * dz
    return L_z


def phytoplankton_growth(P, N, t):
    """Liebig's Law: g_P = g_Pmax * min(f_light, f_nutrient) [day⁻¹]."""
    L_z = compute_light_profile(P, t)
    f_L = L_z / (L_z + kL)
    f_N = N   / (N   + kN)
    return g_Pmax * np.minimum(f_L, f_N)


def seasonal_diffusivity(t):
    """
    κ(z, t) [m² day⁻¹] at each cell face.
    Mixed-layer depth and amplitude both follow a cosine seasonal cycle:
    deep and turbulent in winter (Sverdrup bloom suppression),
    shallow and calm in summer (stratification).
    """
    doy = t % 365.0
    z_mean      = 0.5 * (z_Mix + z_MixWinter)
    z_amp       = 0.5 * (z_MixWinter - z_Mix)
    z_MixSeason = z_mean + z_amp * np.cos(2.0 * np.pi * doy / 365.0)
    kappa_season = (0.5 * (kappa_top + kappa_bottom)
                    + 0.5 * (kappa_top - kappa_bottom)
                    * np.cos(2.0 * np.pi * doy / 365.0))
    return (0.5 * (1.0 - np.tanh((z_face - z_MixSeason) / zeta_mix))
            * (kappa_season - kappa_bottom) + kappa_bottom)


def seasonal_temperature_C(t):
    """Water temperature [°C], sinusoidal: ~2°C Feb, ~18°C Aug."""
    doy = t % 365.0
    return 10.0 + 8.0 * np.sin(2.0 * np.pi * (doy - 30.0) / 365.0)


def arrhenius_correction(T_K):
    """
    Dome-shaped Arrhenius correction for mussel physiology [-].
    Maximum at T_ref = 20°C, suppressed at low and high temperatures.
    From van der Veer et al. (2006) and Maar et al. (2018).
    """
    f_T    = np.exp(T_A  * (1.0 / T_ref - 1.0 / T_K))
    f_low  = 1.0 / (1.0 + np.exp(T_AL * (1.0 / T_K - 1.0 / T_L)))
    f_high = 1.0 / (1.0 + np.exp(T_AH * (1.0 / T_H - 1.0 / T_K)))
    return f_T * f_low * f_high


def copper_concentration(t, Cu_pulse_magnitude):
    """
    External Cu2+ forcing [µg Cu L⁻¹].
    Background + rectangular pulse followed by exponential decay.
    Spatially uniform (no advection), per course instructions.
    """
    Cu = Cu_background
    if t > Cu_pulse_day:
        elapsed = t - Cu_pulse_day
        if elapsed <= Cu_pulse_dur:
            Cu += Cu_pulse_magnitude
        else:
            Cu += Cu_pulse_magnitude * np.exp(-Cu_decay_rate * (elapsed - Cu_pulse_dur))
    return Cu


def copper_inhibition(Cu):
    """
    Hill dose-response: f_Cu = 1 / (1 + (Cu/EC50)^n) [-].
    n=2 gives sigmoidal shape consistent with cooperative gill cilia inhibition.
    f_Cu = 1: no inhibition. f_Cu → 0: total inhibition.
    """
    return 1.0 / (1.0 + (Cu / EC50_Cu) ** n_hill)


# --- Numerical flux functions -----------------------------------------------

def diffusive_flux(C, kappa_z):
    """Central-difference diffusive flux at each face [mmol m⁻² day⁻¹]."""
    J = np.zeros(nz + 1)
    for i in range(1, nz):
        J[i] = -kappa_z[i] * (C[i] - C[i-1]) / dz
    return J


def sinking_flux(C, w):
    """Upwind advective flux for sinking material [mmol m⁻² day⁻¹]."""
    J = np.zeros(nz + 1)
    if w > 0:
        for i in range(1, nz):
            J[i] = w * C[i - 1]
    return J


# --- ODE system -------------------------------------------------------------
# State vector layout:
#   y[0   : nz  ]  N  [mmol N m⁻³]
#   y[nz  : 2nz ]  P  [mmol N m⁻³]
#   y[2nz : 3nz ]  Z  [mmol N m⁻³]
#   y[3nz : 4nz ]  D  [mmol N m⁻³]
#   y[4nz : 5nz ]  O  [mmol O₂ m⁻³]
#   y[5nz      ]   L  mussel shell length [cm]
#   y[5nz + 1  ]   e  mussel reserve density [-]

def model_rhs(t, y, Cu_pulse_magnitude):
    """Right-hand side of the coupled NPZDO + DEB ODE system."""

    # Unpack state vector
    N   = np.maximum(y[0*nz : 1*nz], 0.0)
    P   = np.maximum(y[1*nz : 2*nz], 1e-10)
    Z   = np.maximum(y[2*nz : 3*nz], 1e-10)
    D   = np.maximum(y[3*nz : 4*nz], 0.0)
    O   = np.maximum(y[4*nz : 5*nz], 0.0)
    L_m = max(float(y[5*nz]),     0.01)
    e_m = np.clip(float(y[5*nz + 1]), 0.0, 1.0)

    # Forcing
    Cu      = copper_concentration(t, Cu_pulse_magnitude)
    f_Cu    = copper_inhibition(Cu)
    T_K     = seasonal_temperature_C(t) + 273.15
    f_T     = arrhenius_correction(T_K)
    kappa_z = seasonal_diffusivity(t)

    # NPZDO rate terms
    g_P  = phytoplankton_growth(P, N, t)       # [day⁻¹] per cell
    g_Z  = g_Zmax * P / (P + k_Z)             # zooplankton grazing rate [day⁻¹]
    g_O2 = O / (O + k_anox)                   # O₂ limitation for remineralisation

    # DEB mussel sub-model
    chl_a  = P[z_mussel_idx] / N_to_chla      # food proxy [mg chl-a m⁻³]
    F_rate = F_max * (L_m / Lm)**2 * f_T * f_Cu   # filtration rate [m³ ind⁻¹ day⁻¹]
    f_food = chl_a / (chl_a + k_food)         # Holling II food saturation [-]

    # Phytoplankton cleared from the mussel cell [mmol N m⁻³ day⁻¹]
    P_cleared  = F_rate * f_food * P[z_mussel_idx] * (n_mussel / dz)
    N_excreted = 0.20 * P_cleared             # ~20% of ingested N returned as DIN

    # DEB reserve and growth dynamics (Kooijman 2010; Maar et al. 2015)
    f_food_eff = min(f_Cu * f_food, 1.0)      # Cu reduces effective food intake
    de_dt = (f_food_eff * f_T - e_m) * v_dot / max(L_m, 0.01)
    p_mob = e_m * v_dot / max(L_m, 0.01)
    dL_dt = max((kappa_deb * p_mob - p_M) * L_m / 3.0, -0.005)

    # Physical transport
    J_N_d = diffusive_flux(N, kappa_z)
    J_P_d = diffusive_flux(P, kappa_z)
    # Zooplankton uses reduced diffusivity — motile, not a passive tracer
    J_Z_d = diffusive_flux(Z, np.full(nz + 1, 0.5))
    J_D_d = diffusive_flux(D, kappa_z)
    J_O_d = diffusive_flux(O, kappa_z)
    J_P_s = sinking_flux(P, w_P)
    J_D_s = sinking_flux(D, w_D)

    # Derivatives
    dN = np.zeros(nz)
    dP = np.zeros(nz)
    dZ = np.zeros(nz)
    dD = np.zeros(nz)
    dO = np.zeros(nz)

    for i in range(nz):
        phyto_growth = g_P[i] * P[i]
        zoo_grazing  = g_Z[i] * Z[i]
        phyto_mort   = m_P * P[i]
        zoo_mort     = m_Z * Z[i]**2
        remineralise = tau * g_O2[i] * D[i]

        div_N = -(J_N_d[i+1] - J_N_d[i]) / dz
        div_P = -(J_P_d[i+1] - J_P_d[i] + J_P_s[i+1] - J_P_s[i]) / dz
        div_Z = -(J_Z_d[i+1] - J_Z_d[i]) / dz
        div_D = -(J_D_d[i+1] - J_D_d[i] + J_D_s[i+1] - J_D_s[i]) / dz
        div_O = -(J_O_d[i+1] - J_O_d[i]) / dz

        dN[i] = -phyto_growth + eps_N * zoo_grazing + remineralise + div_N
        dP[i] =  phyto_growth - zoo_grazing - phyto_mort + div_P
        dZ[i] = (1.0 - eps_N - eps_D) * zoo_grazing - zoo_mort + div_Z
        dD[i] =  phyto_mort + zoo_mort + eps_D * zoo_grazing - remineralise + div_D
        dO[i] =  gamma_O * phyto_growth - gamma_O * (eps_N * zoo_grazing + remineralise) + div_O

        if i == z_mussel_idx:
            dP[i] -= P_cleared
            dN[i] += N_excreted
            dO[i] -= 0.10 * P_cleared

    # Boundary conditions
    dO[0]     += k_piston * (O2_sat - O[0]) / dz          # air-sea O₂ exchange
    dN[nz-1]  += kappa_N * (N_Bottom - N[nz-1]) / dz**2  # deep nutrient flux

    return np.concatenate([dN, dP, dZ, dD, dO, [dL_dt, de_dt]])


# --- Initial conditions vector ----------------------------------------------
y0 = np.concatenate([
    np.full(nz, N_init_val),
    np.full(nz, P_init_val),
    np.full(nz, Z_init_val),
    np.full(nz, D_init_val),
    np.full(nz, O_init_val),
    [L_init, e_init],
])


# --- Run simulations --------------------------------------------------------

def run_scenario(label, Cu_pulse_magnitude):
    print(f"  Running '{label}'...", end=" ")
    sol = solve_ivp(
        fun      = lambda t, y: model_rhs(t, y, Cu_pulse_magnitude),
        t_span   = (0.0, t_end),
        y0       = y0.copy(),
        method   = 'RK45',
        t_eval   = t_eval,
        rtol     = 1e-3,
        atol     = 1e-6,
        max_step = 2.0,
    )
    status = "OK" if sol.success else f"failed: {sol.message}"
    print(f"{len(sol.t)} steps — {status}")
    return sol

print("Running simulations...")
sol_ctrl = run_scenario("Control  (2 µg/L background)", Cu_pulse_magnitude=0.0)
sol_low  = run_scenario("Low Cu   (42 µg/L peak)",      Cu_pulse_magnitude=40.0)
sol_high = run_scenario("High Cu  (82 µg/L peak)",      Cu_pulse_magnitude=80.0)


# --- Extract results --------------------------------------------------------

def extract(sol):
    """Unpack solution array into named variables."""
    t = sol.t
    N = sol.y[0*nz : 1*nz, :]
    P = sol.y[1*nz : 2*nz, :]
    Z = sol.y[2*nz : 3*nz, :]
    D = sol.y[3*nz : 4*nz, :]
    O = sol.y[4*nz : 5*nz, :]
    L = sol.y[5*nz,         :]
    e = sol.y[5*nz + 1,     :]
    return t, N, P, Z, D, O, L, e

t, Nc, Pc, Zc, Dc, Oc, Lc, ec = extract(sol_ctrl)
_, Nl, Pl, Zl, Dl, Ol, Ll, el = extract(sol_low)
_, Nh, Ph, Zh, Dh, Oh, Lh, eh = extract(sol_high)

# Cu time series
def cu_series(t_arr, Cu_pk):
    return np.array([copper_concentration(tt, Cu_pk) for tt in t_arr])

Cu_c = cu_series(t, 0.0)
Cu_l = cu_series(t, 40.0)
Cu_h = cu_series(t, 80.0)

fCu_c = copper_inhibition(Cu_c)
fCu_l = copper_inhibition(Cu_l)
fCu_h = copper_inhibition(Cu_h)

T_K_arr = np.array([seasonal_temperature_C(tt) + 273.15 for tt in t])
fT_arr  = np.array([arrhenius_correction(tk) for tk in T_K_arr])

# Filtration rates [m³ ind⁻¹ day⁻¹]
FR_c = F_max * (Lc / Lm)**2 * fT_arr * fCu_c
FR_l = F_max * (Ll / Lm)**2 * fT_arr * fCu_l
FR_h = F_max * (Lh / Lm)**2 * fT_arr * fCu_h

# Population Biofiltration Capacity [m³ m⁻² day⁻¹]
BFC_c = FR_c * n_mussel
BFC_l = FR_l * n_mussel
BFC_h = FR_h * n_mussel

def ti(t_target):
    return int(np.argmin(np.abs(t - t_target)))

C  = {'ctrl': '#2166ac', 'low': '#f4a582', 'high': '#d6604d'}

print("Generating figures...")


# --- Figure 1: Cu concentration and filtration inhibition -------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for Cu, fc, lbl, c, ls in [
    (Cu_c, fCu_c, 'Control (0 µg/L pulse)',  C['ctrl'], '-'),
    (Cu_l, fCu_l, 'Low Cu (40 µg/L pulse)',  C['low'],  '--'),
    (Cu_h, fCu_h, 'High Cu (80 µg/L pulse)', C['high'], ':'),
]:
    ax1.plot(t, Cu, color=c, lw=2, ls=ls, label=lbl)
    ax2.plot(t, fc * 100, color=c, lw=2, ls=ls)

ax1.axhline(EC50_Cu, ls='-.', color='k', lw=1, label=f'EC₅₀ = {EC50_Cu} µg/L')
ax1.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.12, color='tomato',
            label='Cu contamination window')
ax1.set_ylabel('[Cu²⁺] (µg L⁻¹)')
ax1.set_title('Figure 1 — Copper Concentration Scenarios & Mussel Filtration Inhibition')
ax1.legend(loc='upper right')
ax1.set_ylim(bottom=0)

ax2.axhline(50, ls='-.', color='k', lw=1, label='50% inhibition')
ax2.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.12, color='tomato')
ax2.set_ylabel('Filtration capacity (%)')
ax2.set_xlabel('Time (days)')
ax2.set_ylim(0, 105)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('Fig1_Cu_inhibition.png', bbox_inches='tight')
plt.close()
print("  Fig1_Cu_inhibition.png")


# --- Figure 2: Depth-averaged NPZDO time series (3 scenarios) ---------------
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
plot_vars = [
    (np.mean(Nc,0), np.mean(Nl,0), np.mean(Nh,0), 'Nutrients (N)',     'mmol N m⁻³',  axes[0,0]),
    (np.mean(Pc,0), np.mean(Pl,0), np.mean(Ph,0), 'Phytoplankton (P)', 'mmol N m⁻³',  axes[0,1]),
    (np.mean(Zc,0), np.mean(Zl,0), np.mean(Zh,0), 'Zooplankton (Z)',   'mmol N m⁻³',  axes[1,0]),
    (np.mean(Dc,0), np.mean(Dl,0), np.mean(Dh,0), 'Detritus (D)',      'mmol N m⁻³',  axes[1,1]),
    (np.mean(Oc,0), np.mean(Ol,0), np.mean(Oh,0), 'Oxygen (O)',        'mmol O₂ m⁻³', axes[2,0]),
]
for vc, vl, vh, lbl, yu, ax in plot_vars:
    ax.plot(t, vc, color=C['ctrl'], lw=1.8, label='Control')
    ax.plot(t, vl, color=C['low'],  lw=1.8, ls='--', label='Low Cu')
    ax.plot(t, vh, color=C['high'], lw=1.8, ls=':',  label='High Cu')
    ax.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.13, color='tomato')
    ax.set_title(lbl)
    ax.set_ylabel(yu)
    ax.legend(loc='best')

axes[2,0].set_xlabel('Time (days)')
axes[2,1].axis('off')
fig.suptitle('Figure 2 — Depth-Averaged NPZDO State Variables (Three Copper Scenarios)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('Fig2_NPZDO_timeseries.png', bbox_inches='tight')
plt.close()
print("  Fig2_NPZDO_timeseries.png")


# --- Figure 3: Vertical profiles at 3 key moments (control) ----------------
fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)

moments = [
    (ti(57),  'Spring bloom (day 57)'),
    (ti(75),  'Cu pulse peak (day 75)'),
    (ti(150), 'Summer (day 150)'),
]
for ax, (i_t, lbl) in zip(axes, moments):
    ax.plot(Nc[:, i_t], -z_cell, color='steelblue',  lw=2.0, label='N')
    ax.plot(Pc[:, i_t], -z_cell, color='seagreen',   lw=2.0, label='P')
    ax.plot(Zc[:, i_t], -z_cell, color='darkorange', lw=2.0, label='Z')
    ax.plot(Dc[:, i_t], -z_cell, color='sienna',     lw=2.0, label='D')
    ax.axhline(-10.0, ls=':', color='gray', lw=1.2, label='Pycnocline (~10m)')
    ax.set_xlabel('Concentration (mmol N m⁻³)')
    ax.set_title(lbl)
    ax.legend(loc='lower right', fontsize=8)

axes[0].set_ylabel('Depth (m)')
fig.suptitle('Figure 3 — Vertical Profiles of NPZDO Variables (Control Scenario)',
             fontsize=13)
plt.tight_layout()
plt.savefig('Fig3_vertical_profiles.png', bbox_inches='tight')
plt.close()
print("  Fig3_vertical_profiles.png")


# --- Figure 4: Light vs. nutrient limitation by depth (day 300) -------------
i_ss = ti(300)
P_ss = Pc[:, i_ss]
N_ss = Nc[:, i_ss]
L_z  = compute_light_profile(P_ss, 300.0)
f_L  = L_z / (L_z + kL)
f_N  = N_ss / (N_ss + kN)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

ax1.plot(f_L * 100, -z_cell, color='goldenrod', lw=2, label='Light  L/(L+kL)')
ax1.plot(f_N * 100, -z_cell, color='steelblue', lw=2, label='Nutrients  N/(N+kN)')
ax1.axhline(-z_Mix, ls=':', color='gray', lw=1.2)
ax1.set_xlabel('Limitation factor (%)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('Light vs. Nutrient Limitation')
ax1.legend(loc='lower right')
ax1.set_xlim(0, 105)

colors_bar = np.where(f_L < f_N, 'goldenrod', 'steelblue')
for i in range(nz):
    ax2.barh(-z_cell[i], 1, height=dz * 0.9, color=colors_bar[i], alpha=0.8)
ax2.set_xlim(0, 1)
ax2.set_title('Limiting Factor by Depth')
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color='goldenrod', label='Light-limited'),
                    Patch(color='steelblue',  label='Nutrient-limited')],
           loc='lower center')
ax2.set_xticks([])

fig.suptitle('Figure 4 — Phytoplankton Growth Limitation by Depth (Day 300, Control)',
             fontsize=13)
plt.tight_layout()
plt.savefig('Fig4_limiting_factor.png', bbox_inches='tight')
plt.close()
print("  Fig4_limiting_factor.png")


# --- Figure 5: DEB mussel growth (shell length and reserve density) ---------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

for L, e, lbl, c, ls in [
    (Lc, ec, 'Control', C['ctrl'], '-'),
    (Ll, el, 'Low Cu',  C['low'],  '--'),
    (Lh, eh, 'High Cu', C['high'], ':'),
]:
    ax1.plot(t, L,     color=c, lw=2.2, ls=ls, label=lbl)
    ax2.plot(t, e*100, color=c, lw=2.2, ls=ls, label=lbl)

for ax in (ax1, ax2):
    ax.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.15,
               color='tomato', label='Cu pulse' if ax is ax1 else None)
    ax.legend(loc='upper left')

ax1.set_ylabel('Shell length L (cm)')
ax1.set_title('Figure 5 — Blue Mussel DEB Growth Under Copper Stress')
ax2.set_ylabel('Reserve density e (%)')
ax2.set_xlabel('Time (days)')
plt.tight_layout()
plt.savefig('Fig5_mussel_growth.png', bbox_inches='tight')
plt.close()
print("  Fig5_mussel_growth.png")


# --- Figure 6: Population Biofiltration Capacity ----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

for BFC, lbl, c, ls in [
    (BFC_c, 'Control', C['ctrl'], '-'),
    (BFC_l, 'Low Cu',  C['low'],  '--'),
    (BFC_h, 'High Cu', C['high'], ':'),
]:
    ax1.plot(t, BFC, color=c, lw=2.5, ls=ls, label=lbl)

ax1.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.15, color='tomato',
            label='Cu contamination')
ax1.set_ylabel('BFC (m³ m⁻² day⁻¹)')
ax1.set_title('Figure 6 — Population Biofiltration Capacity (KEY RESULT)')
ax1.legend(loc='upper right')

red_l = np.maximum((BFC_c - BFC_l) / (BFC_c + 1e-10) * 100, 0)
red_h = np.maximum((BFC_c - BFC_h) / (BFC_c + 1e-10) * 100, 0)
ax2.fill_between(t, red_l, alpha=0.4, color=C['low'],  label='Low Cu reduction')
ax2.fill_between(t, red_h, alpha=0.4, color=C['high'], label='High Cu reduction')
ax2.plot(t, red_l, color=C['low'],  lw=1.5, ls='--')
ax2.plot(t, red_h, color=C['high'], lw=1.5, ls=':')
ax2.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.12, color='tomato')
ax2.set_ylabel('BFC reduction vs. Control (%)')
ax2.set_xlabel('Time (days)')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Fig6_biofiltration_capacity.png', bbox_inches='tight')
plt.close()
print("  Fig6_biofiltration_capacity.png")


# --- Figure 7: Phytoplankton at the mussel layer ----------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
for P, lbl, c, ls in [
    (Pc, 'Control', C['ctrl'], '-'),
    (Pl, 'Low Cu',  C['low'],  '--'),
    (Ph, 'High Cu', C['high'], ':'),
]:
    ax.plot(t, P[z_mussel_idx, :], color=c, lw=2, ls=ls, label=lbl)

ax.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.15, color='tomato',
           label='Cu pulse')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Phytoplankton P at mussel layer\n(mmol N m⁻³)')
ax.set_title('Figure 7 — Phytoplankton at the Mussel Layer\n'
             '(Higher Cu → less grazing → phytoplankton accumulates)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Fig7_phyto_at_mussel_layer.png', bbox_inches='tight')
plt.close()
print("  Fig7_phyto_at_mussel_layer.png")


# --- Figure 8: Summary dashboard --------------------------------------------
fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38)

ax_a = fig.add_subplot(gs[0, 0])
Cu_range   = np.linspace(0, 200, 500)
f_Cu_range = copper_inhibition(Cu_range)
ax_a.plot(Cu_range, f_Cu_range * 100, color='darkred', lw=2.5)
ax_a.axhline(50, ls='--', color='gray', lw=1.2)
ax_a.axvline(EC50_Cu, ls='--', color='gray', lw=1.2)
for pk, cc, lbl in [(Cu_background, C['ctrl'], 'Background'),
                     (40 + Cu_background, C['low'],  'Low Cu peak'),
                     (80 + Cu_background, C['high'], 'High Cu peak')]:
    fval = copper_inhibition(pk) * 100
    ax_a.scatter([pk], [fval], s=70, color=cc, zorder=5)
    ax_a.annotate(f'{fval:.0f}%', (pk, fval), textcoords='offset points',
                  xytext=(5, 4), fontsize=8, color=cc)
ax_a.set_xlim(0, 200); ax_a.set_ylim(0, 105)
ax_a.set_xlabel('[Cu²⁺] (µg L⁻¹)'); ax_a.set_ylabel('Filtration capacity (%)')
ax_a.set_title('A — Cu Dose-Response (Hill eq.)')

ax_b = fig.add_subplot(gs[0, 1])
i_pk = ti(Cu_pulse_day + 15)
for Nv, cc, lbl in [(Nc, C['ctrl'], 'Control'), (Nh, C['high'], 'High Cu')]:
    ax_b.plot(Nv[:, i_pk], -z_cell, color=cc, lw=2, label=lbl)
ax_b.set_xlabel('N (mmol N m⁻³)'); ax_b.set_ylabel('Depth (m)')
ax_b.set_title(f'B — Nutrient Profile at Day {t[i_pk]:.0f}')
ax_b.legend()

ax_c = fig.add_subplot(gs[1, 0])
for L, cc, lbl, ls in [(Lc, C['ctrl'], 'Control', '-'),
                        (Ll, C['low'],  'Low Cu',  '--'),
                        (Lh, C['high'], 'High Cu', ':')]:
    ax_c.plot(t, L, color=cc, lw=2, ls=ls, label=lbl)
ax_c.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.15, color='tomato')
ax_c.set_xlabel('Time (days)'); ax_c.set_ylabel('Shell length (cm)')
ax_c.set_title('C — Mussel Growth'); ax_c.legend()

ax_d = fig.add_subplot(gs[1, 1])
for BFC, cc, lbl, ls in [(BFC_c, C['ctrl'], 'Control', '-'),
                          (BFC_l, C['low'],  'Low Cu',  '--'),
                          (BFC_h, C['high'], 'High Cu', ':')]:
    ax_d.plot(t, BFC, color=cc, lw=2, ls=ls, label=lbl)
ax_d.axvspan(Cu_pulse_day, Cu_pulse_day + Cu_pulse_dur, alpha=0.15, color='tomato')
ax_d.set_xlabel('Time (days)'); ax_d.set_ylabel('BFC (m³ m⁻² day⁻¹)')
ax_d.set_title('D — Biofiltration Capacity'); ax_d.legend()

fig.suptitle('Figure 8 — Summary Dashboard: Cu Toxicity Impact on NPZDO + Mussel System',
             fontsize=13, fontweight='bold')
plt.savefig('Fig8_dashboard.png', bbox_inches='tight')
plt.close()
print("  Fig8_dashboard.png")


# --- Summary table ----------------------------------------------------------
i_pk = ti(Cu_pulse_day + 15)
W    = 38

print("\n" + "=" * 72)
print(f"  Summary — state at Cu pulse peak (day {t[i_pk]:.0f})")
print("=" * 72)
print(f"{'Metric':<{W}} {'Control':>10} {'Low Cu':>10} {'High Cu':>10}")
print("-" * 72)

rows = [
    ("Peak [Cu²⁺] (µg L⁻¹)",
     Cu_c[i_pk], Cu_l[i_pk], Cu_h[i_pk], ".1f"),
    ("Filtration inhibition (%)",
     (1-fCu_c[i_pk])*100, (1-fCu_l[i_pk])*100, (1-fCu_h[i_pk])*100, ".1f"),
    ("Biofiltration Capacity (m³ m⁻² day⁻¹)",
     BFC_c[i_pk], BFC_l[i_pk], BFC_h[i_pk], ".3f"),
    ("Mussel shell length L (cm)",
     Lc[i_pk], Ll[i_pk], Lh[i_pk], ".2f"),
    ("Reserve density e (%)",
     ec[i_pk]*100, el[i_pk]*100, eh[i_pk]*100, ".1f"),
    ("P at mussel layer (mmol N m⁻³)",
     Pc[z_mussel_idx, i_pk], Pl[z_mussel_idx, i_pk], Ph[z_mussel_idx, i_pk], ".2f"),
    ("Depth-mean O₂ (mmol m⁻³)",
     np.mean(Oc[:, i_pk]), np.mean(Ol[:, i_pk]), np.mean(Oh[:, i_pk]), ".1f"),
]
for name, vc, vl, vh, fmt in rows:
    print(f"{name:<{W}} {vc:>10{fmt}} {vl:>10{fmt}} {vh:>10{fmt}}")
print("=" * 72)