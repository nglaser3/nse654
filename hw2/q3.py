import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla


# Load multigroup data
with np.load("mgxs.npz") as mgxs:
    # Check available data
    print("Available data:", mgxs.files)

    # Load the data
    G = mgxs['G']
    E = mgxs['E']
    sigma_t = mgxs["sigma_t"]
    sigma_s = mgxs["sigma_s"] # [E_out, E_in]
    sigma_f = mgxs["sigma_f"]
    nu = mgxs["nu"] # Fission multiplication
    chi = mgxs["chi"] # Fission spectrum
    v = mgxs["v"] # Speed [cm/s]

# Energy grid properties
E_mid = 0.5 * (E[1:] + E[:-1])
dE = (E[1:] - E[:-1])

# Multigroup source
Qmax = 1.0
Q = np.zeros(G)
Q[-1] = Qmax

# Fission production XS
nu_sigma_f = nu * sigma_f

# Total scattering
sigma_s_total = np.zeros(G)
for g_in in range(G):
    for g_out in range(G):
        sigma_s_total[g_in] += sigma_s[g_out, g_in]

N = 1000
times = np.logspace(-9, 0, N)
times = np.insert(times, 0, 0.0)
dts = times[1:] - times[:-1]

# =========================================================================== #
# Solve
# =========================================================================== #
def yscale(flux):
    return E_mid * flux / dE

def steady(source):
    M = np.diag(sigma_t) - sigma_s - np.diag(chi*nu_sigma_f)
    return sla.solve(M, source)

steady_flux = steady(Q)

fig, ax = plt.subplots()
ax.loglog(E_mid, yscale(steady_flux))
ax.set_xlabel("Energy  [eV]")
fig.savefig("figs/q3b")

# =========================================================================== #

def backward_euler():
    eta_ = np.diag(v)@(np.diag(sigma_t) - sigma_s - np.diag(chi*nu_sigma_f))
    fluxes = []
    fluxes.append(np.zeros_like(Q))
    for i, dt in enumerate(dts):
        eta = dt * eta_
        inner = fluxes[i] + v * dt * Q
        flux = sla.solve(np.eye(v.shape[0]) + eta, inner)
        fluxes.append(flux)
    return fluxes

fluxes = backward_euler()

fig, ax = plt.subplots()
for it in (1, 250, 500, 600):
    ax.loglog(E_mid, yscale(fluxes[it]), label=f"Step: {it}")
ax.legend()
ax.set_xlabel("Energy  [eV]")
fig.savefig("figs/q3ci")

fig, ax = plt.subplots()
rf = [np.dot(sigma_f, flux) for flux in fluxes]
ax.semilogx(times, rf)
ax.set_xlabel("Time  [s]")
fig.savefig("figs/q3cii")

# =========================================================================== #

flux_collapse = steady(Q)

flux_int = np.dot(flux_collapse, dE)

collapse = lambda param: np.dot(param*flux_collapse, dE) / flux_int
v1 = collapse(v)
sigma_t1 = collapse(sigma_t)
sigma_s1 = collapse(sigma_s_total)
sigma_f1 = collapse(sigma_f)
nu_sigma_f1 = collapse(nu_sigma_f * chi)

names = ["v", "sigt", "sigs", "sigf", "nusigf"]
vals = [v1, sigma_t1, sigma_s1, sigma_f1, nu_sigma_f1]
for name, val in zip(names, vals):
    print(f"{name}: {val:.5e}")

# =========================================================================== #

Q1 = np.sum(Q) 

eta_ = v1 * (sigma_t1 - nu_sigma_f1 - sigma_s1)
fluxes = [0.0]
for i, dt in enumerate(dts):
    eta = eta_ * dt
    inner = fluxes[i] + v1 * dt * Q1
    flux = (1 / (1 + eta)) * inner
    fluxes.append(flux)

rf = [sigma_f1 * flux for flux in fluxes]
fig, ax = plt.subplots()
ax.semilogx(times, rf)
fig.savefig("figs/q3e")
plt.show()