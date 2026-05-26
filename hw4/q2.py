from solver import CrossSection, CrossSections, Settings, Simulation
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
N = 50 
bounds = [0.0, 10.0]
for sigma_s in [0.6, 0.9, 0.95, 0.99]:
  xarr = np.linspace(*bounds, N+1)

  # materials
  total_xs = CrossSection(
    values=[1.0],
    xbounds=bounds,
    xarr=xarr
  )
  scattering_xs = CrossSection(
    values=[[sigma_s]],
    xbounds=bounds,
    xarr=xarr
  )
  xs = CrossSections(total=total_xs, scattering=scattering_xs)

  # settings
  settings = Settings(
    xarr=xarr,
    order=8,
  )

  # source
  source = np.ones((settings.order, N))

  sim = Simulation(xs, settings, source)
  phi = sim.solve(False)

  ax.plot(xarr, phi(xarr), label=rf"$\sigma_s$ = {sigma_s}")

ax.legend()
ax.grid(False)
ax.set_xlim(*bounds)
ax.set_xlabel("Position [cm]")
ax.set_ylabel(r"Scalar Flux  [n$\cdot$cm$^{-2}$]")
plt.show()