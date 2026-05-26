import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, init
init(autoreset=True)

from numpy.typing import NDArray
from dataclasses import dataclass

from scipy.interpolate import interp1d

class CrossSection():
  def __init__(self, values, xbounds, xarr):
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

@dataclass
class CrossSections():
  total: CrossSection
  scattering: CrossSection

@dataclass
class Settings():
  xarr: NDArray[np.float64]
  order: int = 4,
  expansion: int = 0
  left_bc: str | float = "vacuum"
  right_bc: str | float = "vacuum"
  tolerance: float = 1e-6
  max_its: int = 1000

class Simulation():
  def __init__(self, xs:CrossSections, settings:Settings, source):
    self._mus, self._mu_weights = np.polynomial.legendre.leggauss(settings.order)
    pns = [np.polynomial.legendre.Legendre.basis(n) for n in range(settings.expansion + 1)]
    self._legendre = np.array([pn(self._mus) for pn in pns])
    self._total_xs = xs.total
    self._scattering_xs = xs.scattering
    self._xarr = settings.xarr
    self._h = np.diff(settings.xarr)
    self._psi = np.zeros((len(self._mus), len(settings.xarr)))
    self._source = source 
    self._indep_source = source
    self._settings = settings

  def solve(self, printout=True):
    flux, errors = self.iterate(printout)
    x_arr = (self._xarr[:-1] + self._xarr[1:])/2
    spectral_radii = errors[1:] / errors[:-1]
    print(Fore.BLUE + "Spectral Radius:\n"
          f"\tFinal: \t {spectral_radii[-1]:.5e}\n"
          f"\tAverage: {np.average(spectral_radii):.5e}")
    return interp1d(x_arr, flux, fill_value="extrapolate") 

  def iterate(self, printout=True):
    errors = []
    phi_old = np.ones(len(self._xarr) - 1)
    outstring = lambda iteration, error: f"{iteration:<10}|  {error:.5e}"
    if printout:
      print("Iteration |  Relative Error")
      print("-"*10+"|" + "-"*16)
    for iteration in range(self._settings.max_its + 1):
      self.sweep()
      phi_new = self._phi_moment(0)
      error = np.linalg.norm(phi_new - phi_old, 2) / np.linalg.norm(phi_old, 2)
      if printout:
        print(outstring(iteration, error))
      errors.append(error)
      if error < self._settings.tolerance:
        break
      phi_old = phi_new
      self.update_source()
    if printout:
      print("-"*10+"|" + "-"*16, "\n\n")
    if iteration < self._settings.max_its:
      print(Fore.GREEN + f"Converged in {iteration} iterations")
    else:
      print(Fore.RED + "Failed to converge")

    return phi_new, np.asarray(errors)

  def sweep(self):
    left_bc = self._settings.left_bc
    right_bc = self._settings.right_bc
    if left_bc == "reflective" and right_bc == "reflective":
      RuntimeError("Cannot have two reflective BCs")
    elif left_bc == "reflective":
      self._left_sweep(right_bc)
      self._right_sweep(left_bc)
    else:
      self._right_sweep(left_bc)
      self._left_sweep(right_bc)

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

    for n in np.where(positive_mus)[0]:
      psi_n = self._psi[n]
      tau = self._total_xs.array() * self._h / self._mus[n]
      for j, psi_j in enumerate(psi_n[:-1]):
        psi_next = (1 - tau[j] / 2) / (1 + tau[j] / 2) * psi_j
        psi_next += tau[j] / self._total_xs[j] * self._source[n, j] / (1 + tau[j] / 2)
        psi_n[j+1] = psi_next

  def _left_sweep(self, boundary_type):
    negative_mus = (self._mus < 0)
    if boundary_type == "vacuum":
      self._psi[negative_mus, -1] = 0
    elif boundary_type == "reflective":
      positive_mus = (self._mus > 0)
      self._psi[negative_mus, -1] = self._psi[positive_mus, -1][::-1]
    elif isinstance(boundary_type, float):
      self._psi[negative_mus, 0] = boundary_type
    else:
      RuntimeError("bad bc type")
    
    for n in np.where(negative_mus)[0]:
      psi_n = self._psi[n]
      mu_n = (self._mus[n])
      tau = self._total_xs.array() * self._h / mu_n
      for j in reversed(range(len(psi_n[1:]))):
        psi_j = psi_n[j]
        psi_prev = (1 + tau[j] / 2) / (1 - tau[j] / 2) * psi_j
        psi_prev -= tau[j] / self._total_xs[j] * self._source[n, j-1] / (1 - tau[j] / 2)
        psi_n[j-1] = psi_prev
    
  def update_source(self):
    self._source = self._indep_source.copy()
    sigma_s = self._scattering_xs.array()
    moments = sigma_s.T * np.asarray([ (2*n + 1)/2 * self._phi_moment(n) for n in range(self._settings.expansion + 1)])
    self._source += np.einsum("lm,lx->mx", self._legendre, moments)

  def _phi_moment(self, order):
    psi_mids = (self._psi[:, :-1] + self._psi[:, 1:]) / 2 
    moment = psi_mids.T @ (self._mu_weights*self._legendre[order])
    return moment