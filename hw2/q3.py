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
    M = np.diag(sigma_t) - sigma_s - np.outer(chi, nu_sigma_f)
    return sla.solve(M, source)

steady_flux = steady(Q)

fig, ax = plt.subplots()
ax.loglog(E_mid, yscale(steady_flux))
ax.set_xlabel("Energy  [eV]")
fig.savefig("figs/q3b")

# =========================================================================== #

def backward_euler(v, sigma_t, sigma_s, chi_nu_sigma_f, Q):
    eta_ = np.diag(v)@(np.diag(sigma_t) - sigma_s - chi_nu_sigma_f)
    fluxes = []
    fluxes.append(np.zeros_like(Q))
    for i, dt in enumerate(dts):
        eta = dt * eta_
        inner = fluxes[i] + v * dt * Q
        flux = sla.solve(np.eye(v.shape[0]) + eta, inner)
        fluxes.append(flux)
    return fluxes

fluxes = backward_euler(v, sigma_t, sigma_s, np.outer(chi, nu_sigma_f), Q)

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

flux_int = np.sum(flux_collapse)

collapse = lambda param: np.dot(param,flux_collapse) / flux_int
v1 = 1 / collapse(1/v)
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

# =========================================================================== #

def collapse(bounds, dE, flux, v, sigma_t, sigma_f, Q):
    _v = []
    _sigmat = []
    _sigma_f =[]
    _Q = []
    num_groups = len(bounds) - 1
    for i in range(num_groups):
        low, hi = bounds[i:i+2]
        _collapse = lambda param: np.sum(param[low : hi] * flux[low : hi]) / np.sum(flux[low : hi])
        _v.append(_collapse(1/v))
        _sigmat.append(_collapse(sigma_t))
        _sigma_f.append(_collapse(sigma_f))
        _Q.append(np.sum(Q[low: hi]))
    return 1/np.asarray(_v), np.asarray(_sigmat), np.asarray(_sigma_f),  np.asarray(_Q)

def collapse_matrix(bounds, dE, matrix, flux):
    num_groups = len(bounds) - 1
    _matrix = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        out_low, out_hi = bounds[i: i+2]
        for j in range(num_groups):
            in_low, in_hi = bounds[j: j+2]
            chunk = matrix[out_low: out_hi, in_low: in_hi]
            _flux = flux[in_low:in_hi]
            top = np.sum(chunk @ _flux)
            bottom = np.sum(_flux)
            _matrix[i, j] = top/bottom
    return _matrix

flux_collapse = steady(Q)
bounds2 = [0, 180, 361]
_collapse = lambda _bounds: collapse(_bounds, dE, flux_collapse, v, sigma_t, sigma_f, Q) 
v2, sigma_t2, sigma_f2, Q2 = _collapse(bounds2)
sigma_s2 = collapse_matrix(bounds2, dE, sigma_s, flux_collapse)
chi_nu_sigmaf2 = collapse_matrix(bounds2, dE, np.outer(chi, nu_sigma_f), flux_collapse)


fluxes = backward_euler(v2, sigma_t2, sigma_s2, chi_nu_sigmaf2, Q2)
rf = [np.dot(sigma_f2, flux) for flux in fluxes]
fig, ax = plt.subplots()
ax.semilogx(times, rf)
fig.savefig("figs/q4a")
plt.show()

flux_collapse = steady(Q)
casmo8 = np.array([1.0e-11, 5.8e-2,1.4e-1, 2.8e-1, 6.25e-1, 4.0, 5.53e3, 8.21e5, 1.0e7])
casmo8_bounds = np.searchsorted(E, casmo8)
casmo8_bounds[-1] = len(dE)
v8, sigma_t8, sigma_f8, Q8 = _collapse(casmo8_bounds)
sigma_s8 = collapse_matrix(casmo8_bounds, dE, sigma_s, flux_collapse)
chi_nu_sigmaf8 = collapse_matrix(casmo8_bounds, dE, np.outer(chi, nu_sigma_f), flux_collapse)


fluxes = backward_euler(v8, sigma_t8, sigma_s8, chi_nu_sigmaf8, Q8)
rf = [np.dot(sigma_f8, flux) for flux in fluxes]
fig, ax = plt.subplots()
ax.semilogx(times, rf)
fig.savefig("figs/q4b")
plt.show()