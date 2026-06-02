from solver import CrossSection, CrossSections, Settings, Simulation
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
N = 160
bounds = [0.0, 2.0, 4.0, 5.0, 6.0, 8.0]

xarr = np.linspace(bounds[0], bounds[-1], N+1)
dx=np.diff(xarr)[0]

# materials
total_xs = CrossSection(
  values=[1.0, 1.0, 0.0, 5.0, 50.0],
  xbounds=bounds,
  xarr=xarr
)
scattering_xs = CrossSection(
  values=[[0.9], [0.9], [0.0], [0.0], [0.0]],
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
source = [0.0, 1.0, 0.0, 0.0, 50.0]
_source = [np.full((1, settings.order), s) for s in source]
_xarr = xarr[:-1] + dx/2
indices = np.searchsorted(bounds, _xarr, "right") - 1
source = np.squeeze((np.asarray(_source)[indices]).T)
print(source.shape)

# run
sim = Simulation(xs, settings, source)
phi, spectral_radius = sim.solve(False)

ax.plot(xarr, phi(xarr))

ax.grid(False)
ax.set_xlim(bounds[0], bounds[-1])
ax.set_xlabel("Position [cm]")
ax.set_ylabel(r"Scalar Flux  [n$\cdot$cm$^{-2}$]")
plt.show()