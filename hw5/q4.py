import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from analytical import omega as ana_omega
from numerical import omega as num_omega

from warnings import filterwarnings
filterwarnings("ignore")

sigt_hs = 3 / np.linspace(1, 15, 150)
labels = ["DD-DSA", "SC-DSA", "SC-DSA relaxed"]

fig, ax = plt.subplots()
fig_phi, ax_phi = plt.subplots()
for label, args in zip(labels, ([False, False], [True, False], [True, True])):
  ana_omegas = [ana_omega(sigt_h, 16, *args) for sigt_h in sigt_hs]
  ns = [int(n * (len(sigt_hs) - 1) / 5) for n in range(6)]
  fig_lam, ax_lam = plt.subplots()
  for n in ns:
    omegas, lambdas = ana_omegas[n]
    ax_lam.plot(lambdas, omegas, label = fr"$\sigma_t h={sigt_hs[n]}$")
  ax_lam.legend()
  ax_lam.set_xlabel(r"$\lambda$")
  ax_lam.set_ylabel(r"$\omega$")
  ax_lam.set_title(label)
  fig_lam.savefig(f"figs/omega_lam_{label}")
  max_ana_omegas = [np.max(omegas) for omegas, _ in ana_omegas]
  num_omegas, phi, xarr = num_omega(sigt_hs, 16, *args)
  [ana_line] = ax.plot(sigt_hs, max_ana_omegas, label=label)
  ax.plot(sigt_hs, num_omegas, label=f"{label} num", linestyle=None, marker="x", ms=2.0, color=ana_line.get_color())
  ax_phi.plot(xarr, phi(xarr), label=label)

ax.axhline(1.0, color="k", label="SI")
ax.legend()
ax_phi.legend()
ax.set_xlabel(r"$\sigma_t h$")
ax_phi.set_xlabel(r"$\sigma_t x$")
ax.set_ylabel(r"$\rho$")
ax_phi.set_ylabel(r"$\phi$")
ax.set_xlim(0.0, 3.0)
ax.set_ylim(0, 2.0)
fig.savefig("figs/ana_and_num")
fig_phi.savefig("figs/q4task2_validation")
plt.show()

sigt_hs = 3 / np.arange(1, 16)
table = np.zeros((len(sigt_hs), 7))
table[:, 0] = sigt_hs
for i, (label, args) in enumerate(zip(labels, ([False, False], [True, False], [True, True]))):
  ana_omegas = [ana_omega(sigt_h, 16, *args) for sigt_h in sigt_hs]
  max_ana_omegas = [np.max(omegas) for omegas, _ in ana_omegas]
  table[:, 2 * i + 1] = max_ana_omegas
  num_omegas, _, _ = num_omega(sigt_hs, 16, *args)
  table[:, 2 * i + 2] = num_omegas

table_labels = ["Sigma_t h"]
for label in labels:
  table_labels.append(label + " Analytical")
  table_labels.append(label + " Numerical")
table = pd.DataFrame(table, columns=table_labels)
