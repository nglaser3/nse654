from solver import CrossSection, CrossSections, Settings, Simulation
import numpy as np
import matplotlib.pyplot as plt

def omega(sigth, order, sc=False, relaxed=False):
  sigma_t = 1.0
  X = 30.0 * sigma_t
  h_arr = sigth / sigma_t
  bounds = [0.0, X]

  rhos = []
  for h in h_arr:
    N = int(X / h)
    xarr = np.linspace(*bounds, N+1)
    dx = np.diff(xarr)[0]

    # materials
    total_xs = CrossSection(
      values=[sigma_t],
      xbounds=bounds,
      xarr=xarr
    )
    scattering_xs = CrossSection(
      values=[[sigma_t]],
      xbounds=bounds,
      xarr=xarr
    )
    xs = CrossSections(total=total_xs, scattering=scattering_xs)

    # settings
    settings = Settings(
      xarr=xarr,
      order=order,
      right_bc="reflective",
      left_bc=0.5,
      method="step" if sc else "diamond",
      dsa=True,
      tolerance=1e-8,
      relaxed=relaxed
    )

    # source
    source = np.zeros((settings.order, N))
    xc = 0.5 * (xarr[:-1] + xarr[1:])
    source[:, (xc/sigma_t > 18) & (xc/sigma_t <= 24)] = 0.05

    sim = Simulation(xs, settings, source)
    phi, spectral_radius, iteration = sim.solve(False, False)
    rhos.append(spectral_radius)
  rhos = np.asarray(rhos)
  return rhos, phi, xarr
