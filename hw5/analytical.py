import numpy as np

def omega(sigt_h, order, sc=False, relaxed=False):
  lambdas = np.linspace(0, np.pi / sigt_h, 100)
  taus = lambdas[1:-1] * sigt_h / 2
  omegas = np.zeros_like(lambdas)
  c = 1.0
  for i, tau in enumerate(taus):
    Gamma = np.tan(tau)
    mus, weights = np.polynomial.legendre.leggauss(order)
    pos_mus = mus > 0 
    if sc:
      alpha = 1 / np.tanh(sigt_h / (2*mus[pos_mus])) - 2 * mus[pos_mus] / (sigt_h)
    else:
      alpha = 0
    _help = (alpha + 2*mus[pos_mus]/ sigt_h)

    A = c  * np.sum(
      weights[pos_mus] * (1 + alpha * ( _help) * Gamma**2)/ 
      (1 + _help**2 * Gamma**2)
    )
    gamma = c * (
      (A - 1) / 
      (1 - c + 1/3 * (2/sigt_h * np.sin(tau))**2)
    )

    if relaxed and sc:
      A1 = c
      gamma1 = -c
      A2 = c*np.sum(
        alpha * weights[pos_mus]/ 
        _help
      )
      gamma2 = (A2 - 1) / (1 - c + 4 / (3 * sigt_h**2))
      Beta = 1 + (A2 + A1) / (gamma1 + gamma2)
    else:
      A1 = 1
      gamma1 = 0
      A2 = 0
      gamma2 = -c * (
        -1 /
        (1 - c + 1/3 * (2 / sigt_h)**2)
      )
      Beta = 0

    omega = A + (1 - Beta) * gamma
    omegas[i] = np.abs(omega)
  return omegas[1:-1], lambdas[1:-1]