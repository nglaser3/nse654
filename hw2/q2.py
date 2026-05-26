import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.integrate import quad
from scipy.linalg import norm


# ======================================================================================
# Problem description
# ======================================================================================

# Parameters
v = 5.2 # cm/ns
sigma = 0.8 # /cm
Qmax = 10.0 # /cm^3-ns
ts = 1.0 # ns
s = 0.025 # ns^2
t_final = 5.0 # ns

# Source
def Q(t):
    return Qmax * np.exp(-(t - ts)**2 / (2*s))


# ======================================================================================
# Analytical solution
# ======================================================================================

def phi_anaytical(t):
    def integrand(t):
        return np.exp(-(t - ts)**2 / (2*s)) * np.exp(v*sigma*t)
    integral = quad(integrand, 0, t)[0]
    return v * Qmax * np.exp(-v*sigma*t) * integral


# Generate reference solution for plotting
N_ref = 10000
t_ref = np.linspace(0.0, t_final, N_ref)
phi_ref = np.zeros(N_ref)
for n in range(N_ref):
    phi_ref[n] = phi_anaytical(t_ref[n])


# ======================================================================================
# Forward Euler
# ======================================================================================

def FE_step(t_start, dt, phi_old):
    eta = v * sigma * dt
    t_end = t_start + dt
    Q_bar = quad(Q, t_start, t_end)[0] / dt
    phi_new = eta * Q_bar / sigma + (1.0 - eta) * phi_old 
    return phi_new

# ======================================================================================
# Backward Euler
# ======================================================================================

def BE_step(t_start, dt, phi_old):
    eta = v * sigma * dt
    t_end = t_start + dt
    Q_bar = quad(Q, t_start, t_end)[0] / dt
    phi_new = 1.0 / (1.0 + eta) * (eta * Q_bar / sigma + phi_old)
    return phi_new

# ======================================================================================
# Crank-Nicholson
# ======================================================================================

def CN_step(t_start, dt, phi_old):
    eta = v * sigma * dt
    t_end = t_start + dt
    Q_bar = quad(Q, t_start, t_end)[0] / dt
    phi_new = (1 - eta/2) / (1 + eta/2) * phi_old + eta / (sigma * (1 + eta/2)) * Q_bar
    return phi_new

# ======================================================================================
# TR-BDF2
# ======================================================================================

def TRBDF2_step(t_start, dt, phi_old):
    eta = v * sigma * dt
    tmid = t_start + dt / 2
    t_end = t_start + dt
    Q_barm = quad(Q, t_start, tmid)[0] / (dt / 2)
    Q_bar = quad(Q, t_start, t_end)[0] / dt
    Q_ =  Q_bar - eta / (6 * (1 + eta/4)) * Q_barm
    phi_new = (12 - 5*eta) / (12 + 7*eta + eta**2) * phi_old + (3 * eta ) / (sigma * (3 + eta)) * Q_
    return phi_new

# ======================================================================================
# Multiple Balance
# ======================================================================================

def MB_step(t_start, dt, phi_old):
    eta = v * sigma * dt
    tmid = t_start + dt / 2
    t_end = t_start + dt
    Q_barm = quad(Q, tmid, t_end)[0] / (dt / 2)
    Q_bar = quad(Q, t_start, t_end)[0] / dt 
    Q_ = (eta / sigma) * (eta * Q_barm / 2 + Q_bar)
    phi_new = 1 / (1 + eta + eta**2 / 2) * (phi_old + Q_)
    return phi_new


# ======================================================================================
# Generate results
# ======================================================================================

# Numerical method parameters
N = 100
dt = t_final / N
t = np.linspace(0.0, t_final, N + 1)

# Numerical method driver
def numerical_driver(step):
    phi = np.zeros(N + 1)
    for n in range(N):
        phi[n + 1] = step(t[n], dt, phi[n])
    return phi

# Forward Euler
phi_FE = numerical_driver(FE_step)

# Backward Euler
phi_BE = numerical_driver(BE_step)

# Crank-Nicholson
phi_CN = numerical_driver(CN_step)

# TR-BDF2
phi_TRBDF2 = numerical_driver(TRBDF2_step)

# MB
phi_MB = numerical_driver(MB_step)

# ======================================================================================
# Plot
# ======================================================================================

plt.figure(figsize=(4,3))
plt.plot(t_ref, phi_ref, '-k', label='Analytical')
plt.plot(t, phi_FE, '--y', label='FE')
plt.plot(t, phi_BE, '--ob', fillstyle='none', label='BE')
plt.plot(t, phi_CN, '--sr', fillstyle='none', label='CN')
plt.plot(t, phi_TRBDF2, '--*m', fillstyle='none', label='TR-BDF2')
plt.plot(t, phi_MB, '--Dg', fillstyle='none', label='MB')

plt.grid()
plt.legend()
plt.ylabel(r'$\phi(t)$')
plt.ylim(-1.0, 10.0)
plt.xlabel(r'$t$ [ns]')
eta = v * sigma * dt
plt.title(f"$N={N}$,  $\\eta={eta}$")
plt.savefig(f'figs/q2_{N}')
plt.show()

ns = [1, 4, 10, 100, 1000, 10000]
dts = []
errors = np.zeros((len(ns), 5))
analytical_flux = phi_anaytical(t[-1])
for i, N in enumerate(ns):
    dt = t_final / N
    dts.append(dt)
    t = np.linspace(0, t_final, N+1)
    analytical_flux = np.array([phi_anaytical(_t) for _t in t])
    inv_analytical_norm = 1 / norm(analytical_flux, 2)
    for j, step in enumerate([FE_step, BE_step, CN_step, TRBDF2_step, MB_step]):
        error = (numerical_driver(step)[-1] - analytical_flux[-1]) / analytical_flux[-1] # norm(numerical_driver(step) - analytical_flux, 2) * inv_analytical_norm
        errors[i, j] = abs(error)

fig, ax = plt.subplots()
names = ["FE", "BE", "CN", "TR-BDF2", "MB"]
styles = ["--y", "--ob", "--sr", "--*m", "--Dg"]
for i, (name, style) in enumerate(zip(names, styles)):
    _errors = errors[:, i]
    ax.loglog(dts, _errors, style, fillstyle=None, label = name)
ax.legend()
plt.show()

errors = pd.DataFrame(errors, index=[f"N={n}" for n in ns], columns=names)
print(errors)