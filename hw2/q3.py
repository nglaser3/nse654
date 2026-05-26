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
ax.set_title("Question 3 Part B")
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
ax.loglog(E_mid, yscale(steady(Q)), label="Steady")
ax.legend()
ax.set_xlabel("Energy  [eV]")
ax.set_title("Question 3 part C: i")
fig.savefig("figs/q3ci")

fig, ax = plt.subplots()
rf = [np.dot(sigma_f, flux) for flux in fluxes]
ax.semilogx(times, rf, label="361 Group")
ax.set_xlabel("Time  [s]")
ax.set_title("Question 3 Part C: ii")
#fig.savefig("figs/q3cii")

# =========================================================================== #

flux_collapse = steady(Q)

flux_int = np.sum(flux_collapse)

collapse = lambda param: np.dot(param,flux_collapse) / flux_int
v1 = 1 / collapse(1/v)
sigma_t1 = collapse(sigma_t)
sigma_s1 = collapse(sigma_s_total)
sigma_f1 = collapse(sigma_f)
nu_sigma_f1 = collapse(nu_sigma_f)

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
#fig, ax = plt.subplots()
ax.semilogx(times, rf, label = "1 Group")
ax.set_title("Question 3 Part E")
#fig.savefig("figs/q3e")

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
#fig, ax = plt.subplots()
ax.semilogx(times, rf, label="2 Group")
ax.set_title("Question 4 Part A")
#fig.savefig("figs/q4a")

flux_collapse = steady(Q)
casmo40 = np.array(
    [0., 1.5e-2, 3.e-2, 4.2e-2, 5.8e-2, 8.e-2, 1.e-1, 1.4e-1,
    1.8e-1, 2.2e-1, 2.8e-1, 3.5e-1, 6.25e-1, 8.5e-1, 9.5e-1,
    9.72e-1, 1.02, 1.097, 1.15, 1.3, 1.5, 1.855, 2.1, 2.6, 3.3, 4.,
    9.877, 1.5968e1, 2.77e1, 4.8052e1, 1.4873e2, 5.53e3, 9.118e3,
    1.11e5, 5.e5, 8.21e5, 1.353e6, 2.231e6, 3.679e6, 6.0655e6, 2.e7]
)
casmo40_bounds = np.searchsorted(E, casmo40)
casmo40_bounds[-1] = len(dE)
v40, sigma_t40, sigma_f40, Q40 = _collapse(casmo40_bounds)
sigma_s40 = collapse_matrix(casmo40_bounds, dE, sigma_s, flux_collapse)
chi_nu_sigmaf40 = collapse_matrix(casmo40_bounds, dE, np.outer(chi, nu_sigma_f), flux_collapse)


fluxes = backward_euler(v40, sigma_t40, sigma_s40, chi_nu_sigmaf40, Q40)
rf = [np.dot(sigma_f40, flux) for flux in fluxes]
#fig, ax = plt.subplots()
ax.semilogx(times, rf, label="CASMO-40")
ax.legend()
ax.set_title("All Fission Rates")
fig.savefig("figs/fiss_all")
plt.show()