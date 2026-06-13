import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, init
np.set_printoptions(linewidth=1000)
init(autoreset=True)

from numpy.typing import NDArray
from dataclasses import dataclass

from scipy.interpolate import interp1d

class CrossSection():
  def __init__(self, values, xbounds, xarr):
    values = np.asarray(values)

    edge_indices = np.searchsorted(xbounds, xarr, side="right") - 1
    edge_indices = np.clip(edge_indices, 0, len(values) - 1)
    self._edge_values = values[edge_indices]

    xarr = [xarr[i] + dx/2 for i, dx in enumerate(np.diff(xarr))]
    indices = np.searchsorted(xbounds, xarr, "right") - 1
    self._values = np.asarray(values)[indices]

  def __getitem__(self, i, j=None):
    if j is None:
      return self._values[i]
    else:
      return self._values[i, j]

  def array(self):
    return self._values

class CrossSections():
  total: CrossSection
  scattering: CrossSection
  absorption: CrossSection
  transport: CrossSection

  def __init__(self, total, scattering):
    self.total = total
    self.scattering = scattering
    absorption = CrossSection.__new__(CrossSection)
    absorption._values = total.array() - scattering.array()[:, 0]
    self.absorption = absorption
    sigma_tr = CrossSection.__new__(CrossSection)
    try:
      scattering_term = scattering._values[:, 1]
    except:
      scattering_term = 0.0
    sigma_tr._values = total._values - scattering_term
    self.transport = sigma_tr

@dataclass
class Settings():
  xarr: NDArray[np.float64]
  order: int = 4,
  expansion: int = 0
  left_bc: str | float = "vacuum"
  right_bc: str | float = "vacuum"
  tolerance: float = 1e-6
  max_its: int = 1000
  method: str = "diamond"
  dsa: bool = False
  relaxed: bool = False 

class Simulation():
  def __init__(self, xs:CrossSections, settings:Settings, source):
    self._mus, self._mu_weights = np.polynomial.legendre.leggauss(settings.order)
    pns = [np.polynomial.legendre.Legendre.basis(n) for n in range(settings.expansion + 1)]
    self._legendre = np.array([pn(self._mus) for pn in pns])
    self._total_xs = xs.total
    self._scattering_xs = xs.scattering
    self._absorption_xs = xs.absorption
    self._transport_xs = xs.transport
    self._xarr = settings.xarr
    self._h = np.diff(settings.xarr)
    self._psi = np.zeros((len(self._mus), len(settings.xarr)))
    self._source = source / 2
    self._indep_source = source
    self._settings = settings

  def solve(self, printout=True, report=True):
    if report:
      method = "Diamond Difference" if self._settings.method == "diamond" else "Step Characteristics"
      dsa = " - DSA" if self._settings.dsa else ""
      relaxed = " (relaxed)" if self._settings.relaxed and self._settings.dsa else ""
      print(Fore.YELLOW + f"Method: \t{method + dsa + relaxed}")
    flux, errors, iteration = self.iterate(printout, report)
    x_arr = (self._xarr[:-1] + self._xarr[1:])/2
    spectral_radii = np.abs(errors[1:] / errors[:-1])
    ave_spectral = np.average(spectral_radii)
    if report:
      print(Fore.BLUE + "Spectral Radius:\n"
            f"\tFinal: \t {spectral_radii[-1]:.5e}\n"
            f"\tAverage: {ave_spectral:.5e}\n"
            f"\tMax: \t {np.max(spectral_radii):.5e}\n")
    return interp1d(x_arr, flux, fill_value="extrapolate"), spectral_radii[-1], iteration

  def iterate(self, printout=True, report=True):
    errors = []
    phi_old = np.zeros(len(self._xarr) - 1)
    outstring = lambda iteration, error: f"{iteration:<10}|  {error:.5e}"
    if printout:
      print("Iteration |  Relative Error")
      print("-"*10+"|" + "-"*16)
    for iteration in range(self._settings.max_its + 1):
      self.sweep()
      phi_new = self._phi_moment(0)
      if self._settings.dsa:
        phi_corr = self._dsa_correction(phi_old, phi_new)
        beta = self._get_beta()
        phi_new += (1 - beta) * phi_corr
      error = np.linalg.norm((phi_new - phi_old) / phi_new, np.inf)
      if printout:
        print(outstring(iteration, error))
      errors.append(np.linalg.norm(phi_new - phi_old, np.inf))
      if error < self._settings.tolerance:
        break
      self.update_source(phi_new)
      phi_old = phi_new
    if printout:
      print("-"*10+"|" + "-"*16, "\n\n")

    if report:
      if iteration < self._settings.max_its:
        print(Fore.GREEN + f"Converged in {iteration} iterations")
      else:
        print(Fore.RED + "Failed to converge")

    return phi_new, np.asarray(errors), iteration

  def sweep(self):
    left_bc = self._settings.left_bc
    right_bc = self._settings.right_bc
    if left_bc == "reflective" and right_bc == "reflective":
      RuntimeError("Cannot have two reflective BCs")
    elif right_bc == "reflective":
      self._right_sweep(left_bc)
      self._left_sweep(right_bc)
    else:
      self._left_sweep(right_bc)
      self._right_sweep(left_bc)

  def _right_sweep(self, boundary_type):
    positive_mus = (self._mus > 0)
    if boundary_type == "vacuum":
      self._psi[positive_mus, 0] = 0
    elif boundary_type == "reflective":
      negative_mus = (self._mus < 0)
      self._psi[positive_mus, 0] = self._psi[negative_mus, 0][::-1]
    elif isinstance(boundary_type, float):
      self._psi[positive_mus, 0] = boundary_type
    else:
      RuntimeError("bad bc type")

    if self._settings.method == "diamond":
      self._right_diamond(positive_mus)
    elif self._settings.method == "step":
      self._right_step_characteristic(positive_mus)
    else:
      raise ValueError

  def _right_diamond(self, positive_mus):
    for n in np.where(positive_mus)[0]:
      psi_n = self._psi[n]
      tau = self._total_xs.array() * self._h / self._mus[n]
      for j, psi_j in enumerate(psi_n[:-1]):
        if self._total_xs[j] == 0.0:
          psi_n[j+1] = psi_next
          continue
        psi_next = (1 - tau[j] / 2) / (1 + tau[j] / 2) * psi_j
        psi_next += (tau[j] / self._total_xs[j]) * self._source[n, j] / (1 + tau[j] / 2)
        psi_n[j+1] = psi_next

  def _right_step_characteristic(self, positive_mus):
    for n in np.where(positive_mus)[0]:
      psi_n = self._psi[n]
      tau = self._total_xs.array() * self._h / self._mus[n]
      for j, psi_j in enumerate(psi_n[:-1]):
        if self._total_xs[j] == 0.0:
          psi_n[j+1] = psi_j 
          continue
        psi_next = psi_j * np.exp(-tau[j])
        psi_next +=  (self._source[n, j] / self._total_xs[j]) * (1 - np.exp(-tau[j]))
        psi_n[j+1] = psi_next

  def _left_sweep(self, boundary_type):
    negative_mus = (self._mus < 0)
    if boundary_type == "vacuum":
      self._psi[negative_mus, -1] = 0
    elif boundary_type == "reflective":
      positive_mus = (self._mus > 0)
      self._psi[negative_mus, -1] = self._psi[positive_mus, -1][::-1]
    elif isinstance(boundary_type, float):
      self._psi[negative_mus, -1] = boundary_type
    else:
      RuntimeError("bad bc type")
    
    if self._settings.method == "diamond":
      self._left_diamond(negative_mus)
    elif self._settings.method == "step":
      self._left_step_characteristic(negative_mus)
    else:
      raise ValueError

  def _left_diamond(self, negative_mus):
    for n in np.where(negative_mus)[0]:
      psi_n = self._psi[n]
      mu_n = (self._mus[n])
      tau = self._total_xs.array() * self._h / mu_n
      for j in reversed(range(len(psi_n[1:]))):
        psi_j = psi_n[j+1]
        if self._total_xs[j] == 0.0:
          psi_n[j] = psi_j
          continue
        psi_prev = (1 + tau[j] / 2) / (1 - tau[j] / 2) * psi_j
        psi_prev -= (tau[j] / self._total_xs[j]) * self._source[n, j] / (1 - tau[j] / 2)
        psi_n[j] = psi_prev

  def _left_step_characteristic(self, negative_mus):
    for n in np.where(negative_mus)[0]:
      psi_n = self._psi[n]
      mu_n = (self._mus[n])
      tau = self._total_xs.array() * self._h / abs(mu_n)
      for j in reversed(range(len(psi_n[1:]))):
        psi_j = psi_n[j+1]
        if self._total_xs[j] == 0.0:
          psi_n[j+1] = psi_j
          continue
        psi_prev = psi_j * np.exp(-tau[j])
        psi_prev +=  (self._source[n, j] / self._total_xs[j]) * (1 - np.exp(-tau[j]))
        psi_n[j] = psi_prev

  def update_source(self, phi_new):
    self._source = self._indep_source.copy()/2
    sigma_s = self._scattering_xs.array()
    moments = np.asarray([phi_new / 2] + [ (2*n + 1)/2 * self._phi_moment(n) for n in range(1, self._settings.expansion + 1)])
    self._source += np.einsum("lm,lx->mx", self._legendre, sigma_s.T * moments)

  def _phi_moment(self, order):
    if self._settings.method == "diamond":
      psi_mids = (self._psi[:, :-1] + self._psi[:, 1:]) / 2 
    elif self._settings.method == "step":
      psi_mids = np.zeros((len(self._mus), len(self._total_xs.array())))
      for n, mu_n in enumerate(self._mus):
        tau = self._total_xs.array() * self._h / (mu_n)
        left = self._psi[n, :-1] * (1/tau - np.exp(-tau) / (1-np.exp(-tau)))
        right = self._psi[n, 1:] * (1 / (1-np.exp(-tau)) - 1 / tau)
        psi_mids[n, :] = left + right
    moment = psi_mids.T @ (self._mu_weights*self._legendre[order])
    return moment
  
  def _dsa_correction(self, phi_old, phi_new):
    h = np.diff(self._xarr)
    sigma_s0 = self._scattering_xs[:, 0]
    dsa_source = sigma_s0 * (phi_new - phi_old) * h

    N = len(phi_old)
    A = np.zeros((N, N))

    h_edge = np.zeros(N + 1)
    h_edge[0] = h[0] / 2
    h_edge[1:-1] = (h[:-1] + h[1:])/2
    h_edge[-1] = h[-1] / 2
    tr = self._transport_xs.array()

    # face-centered transport xs: length N+1
    sigma_tr = np.empty(N + 1)
    sigma_tr[0] = tr[0]
    sigma_tr[-1] = tr[-1]
    sigma_tr[1:-1] = (tr[:-1] * h[:-1] + tr[1:] * h[1:]) / (h[:-1] + h[1:])

    A += np.diag(-1 / (3 * sigma_tr[1:-1] * h_edge[1:-1]), -1)
    A += np.diag(-1 / (3 * sigma_tr[1:-1] * h_edge[1:-1]),  1)

    middle = np.zeros(N)
    middle[0] = (
        1 / (3 * sigma_tr[0] * h_edge[0])
        + 1 / (3 * sigma_tr[1] * h_edge[1])
        + self._absorption_xs[0] * h[0]
        + 4 / (4 + 3 * tr[0] * h[0])*(-1/(3*sigma_tr[0]*h_edge[0]))
    )
    middle[1:-1] = (
        1 / (3 * sigma_tr[1:-2] * h_edge[1:-2])
        + 1 / (3 * sigma_tr[2:-1] * h_edge[2:-1])
        + self._absorption_xs[1:-1] * h[1:-1]
    )
    middle[-1] = (
        1 / (3 * sigma_tr[-2] * h_edge[-2])
        + self._absorption_xs[-1] * h[-1]
    )
    A += np.diag(middle) 
    
    F = np.linalg.solve(A, dsa_source)

    return F 

  def _get_beta(self):
    if self._settings.relaxed and self._settings.method == "step":
      c_arr = (self._scattering_xs.array()[:, 0]) / (self._total_xs.array())
      sigt_hs = self._h * self._total_xs.array()
      Beta = np.zeros_like(sigt_hs)
      pos_mus = self._mus[self._mus > 0]
      pos_weights = self._mu_weights[self._mus > 0]
      for i, (sigt_h, c) in enumerate(zip(sigt_hs, c_arr)):
        alpha = 1 / np.tanh(sigt_h / (2*pos_mus)) - 2 * pos_mus / (sigt_h)
        A1 = c
        gamma1 = -c
        A2 = c*np.sum(
          alpha * pos_weights/ 
          (alpha + 2*pos_mus/ sigt_h)
        )
        gamma2 = c * (A2 - 1) / (1 - c + 4 / (3 * sigt_h**2))
        beta = 1 + (A2 + A1) / (gamma1 + gamma2)
        Beta[i] = beta
    else:
      Beta = 0.0
    return Beta