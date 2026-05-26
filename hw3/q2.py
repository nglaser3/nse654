import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss
import matplotlib.pyplot as plt
from scipy import (
    interpolate as interp,
    linalg as sla
)

with np.load("fe56-1.npz") as data:
    fe1_p = data['p']
    fe1_mu = data['mu']
    fe1_sigma_s = data['sigma_s']
fe1_pmu = interp.interp1d(fe1_mu, fe1_p*fe1_sigma_s)

with np.load("fe56-30.npz") as data:
    fe30_p = data['p']
    fe30_mu = data['mu']
    fe30_sigma_s = data['sigma_s']
fe30_pmu = interp.interp1d(fe30_mu, fe30_p*fe30_sigma_s)

N = [1, 3, 5, 10]

def coefficients(func, order):
    mu, weights = leggauss(order)

    func_vals = func(mu)
    legendre_vals = [Legendre.basis(n)(mu) for n in range(order + 1)]
    coeffs = np.array([np.sum(func_vals * leg_vals * weights) for leg_vals in legendre_vals])
    on_constants = (2*np.arange(0, order + 1) + 1) / 2

    expansion = Legendre(on_constants * coeffs)
    return expansion 

def error(mu, expansion, true):
    rms = sla.norm((expansion(mu) - true(mu)) / true(mu))
    return rms / len(mu)

fig, ax = plt.subplots()
mu = np.linspace(-1, 1, 10000)
ax.plot(fe1_mu, fe1_p*fe1_sigma_s, "--D", ms=2, color="k", label=r"$\sigma_s$")
for n in N:
    expansion = coefficients(fe1_pmu, n)
    ax.plot(mu, expansion(mu), label=f"Order {n}")
    print(f"Order {n}:\t {error(fe1_mu, expansion, fe1_pmu)}")
ax.legend()
ax.set_xlabel(r"$\mu_c$")
ax.set_ylabel(r"$\sigma_s(\mu_c)$  [cm]")
ax.set_xlim(-1, 1)
fig.savefig("q2b")
plt.show()

fig, ax = plt.subplots()
ax.plot(fe30_mu, fe30_p*fe30_sigma_s, "--D", ms=2, color="k", label=r"$\sigma_s$")
N += [100]
for n in N:
    expansion = coefficients(fe30_pmu, n)
    ax.plot(mu, expansion(mu), label=f"Order {n}")
    print(f"Order {n}:\t {error(fe30_mu, expansion, fe30_pmu)}")
ax.legend()
ax.set_xlabel(r"$\mu_c$")
ax.set_ylabel(r"$\sigma_s(\mu_c)$  [cm]")
ax.set_xlim(-1, 1)
fig.savefig("q2c")
plt.show()