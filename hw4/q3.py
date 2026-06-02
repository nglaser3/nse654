from solver import CrossSection, CrossSections, Settings, Simulation
import matplotlib.pyplot as plt
import numpy as np

hs = [0.1, 0.25]
orders = [16, 4]
sigma_t = 1.0
for h in hs:
  for order in orders:
    output_matrix = np.empty((4, 5))
    for isig, sigma_s in enumerate([0.8, 0.9, 0.96, 0.999]):
      for ix, X in enumerate([5, 10, 15, 20]):
        bounds = [0, X]
        N = int(X / h)

        xarr = np.linspace(bounds[0], bounds[-1], N+1)
        dx=np.diff(xarr)[0]

        # materials
        total_xs = CrossSection(
          values=[sigma_t],
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
          order=order,
          right_bc=1,
          left_bc=1,
          max_its=10000,
        )

        # source
        source = np.ones((settings.order, N))

        sim = Simulation(xs, settings, source)
        phi, spectral_radius = sim.solve(False)
        output_matrix[isig, ix] = spectral_radius
      output_matrix[isig, -1] = sigma_s/sigma_t
    print(f"H: {h} Order: {order}")
    print(output_matrix)

