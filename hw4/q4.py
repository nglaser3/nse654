from solver import CrossSection, CrossSections, Settings, Simulation
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
N = 100
for X0 in [9.0, 8.0, 7.0, 6.0, 5.0]:
  bounds = [0.0, X0, 10.0]
  xarr = np.linspace(bounds[0], bounds[-1], N+1)
  dx=np.diff(xarr)[0]

  # materials
  total_xs = CrossSection(
    values=[1.0, 1.0],
    xbounds=bounds,
    xarr=xarr
  )
  scattering_xs = CrossSection(
    values=[[0.8],[1.0]],
    xbounds=bounds,
    xarr=xarr
  )
  xs = CrossSections(total=total_xs, scattering=scattering_xs)

  # settings
  settings = Settings(
    xarr=xarr,
    order=16,
    right_bc="reflective",
  )

  # source
  source = np.ones((settings.order, N))

  sim = Simulation(xs, settings, source)
  phi, spectral_radius = sim.solve(False)

  ax.plot(xarr, phi(xarr), label=rf"$X_0$ = {X0}")

ax.legend()
ax.grid(False)
ax.set_xlim(bounds[0], bounds[-1])
ax.set_xlabel("Position [cm]")
ax.set_ylabel(r"Scalar Flux  [n$\cdot$cm$^{-2}$]")
plt.show()