from solver import CrossSection, CrossSections, Settings, Simulation
import numpy as np
import matplotlib.pyplot as plt

X = 20.0
h = 0.1
N = int(X / h)
bounds = [0.0, X]

xarr = np.linspace(*bounds, N+1)
dx = np.diff(xarr)[0]

# materials
total_xs = CrossSection(
  values=[1.0],
  xbounds=bounds,
  xarr=xarr
)
scattering_xs = CrossSection(
  values=[[0.999]],
  xbounds=bounds,
  xarr=xarr
)
xs = CrossSections(total=total_xs, scattering=scattering_xs)

fig, ax = plt.subplots()
for dsa in [False, True]:
  for method in ["diamond", "step"]:
    # settings
    settings = Settings(
      xarr=xarr,
      order=16,
      right_bc="reflective",
      method=method,
      dsa=dsa,
      tolerance=1e-8,
      max_its=10000
    )

    # source
    source = np.ones((settings.order, N))

    sim = Simulation(xs, settings, source)
    phi, spectral_radius, iteration = sim.solve(False)

    label = ""
    label += "DSA-" if dsa else "SI-"
    label += "DD, " if method == "diamond" else "SC, "
    label += rf"$\rho={spectral_radius:.5e}$, $i={iteration}$"
    ax.plot(xarr, phi(xarr), label=label)
plt.legend()
plt.show()