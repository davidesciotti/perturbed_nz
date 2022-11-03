import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import quadpy
from matplotlib import cm
from mpire import WorkerPool
from numba import njit
from scipy.integrate import quad, quad_vec, simpson, dblquad, simps
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf

project_path = Path.cwd().parent

sys.path.append(str(project_path))
sys.path.append(str(project_path.parent / 'common_data/common_config'))

# project modules
# import proj_lib.cosmo_lib as csmlb
# import config.config as cfg
# general configuration modules
import ISTF_fid_params as ISTF
import mpl_cfg as mpl_cfg

# update plot paramseters
rcParams = mpl_cfg.mpl_rcParams_dict
plt.rcParams.update(rcParams)
matplotlib.use('Qt5Agg')

###############################################################################
###############################################################################
###############################################################################


script_start = time.perf_counter()
c_lightspeed = ISTF.constants['c']

H0 = ISTF.primary['h_0'] * 100
Om0 = ISTF.primary['Om_m0']
Ob0 = ISTF.primary['Om_b0']

gamma = ISTF.extensions['gamma']

z_edges = ISTF.photoz_bins['zbin_edges']
z_minus = z_edges[:-1]
z_plus = z_edges[1:]
z_m = ISTF.photoz_bins['z_median']
zbins = ISTF.photoz_bins['zbins']

z_0 = z_m / np.sqrt(2)
z_mean = (z_plus + z_minus) / 2
z_min = z_edges[0]
z_max = z_edges[-1]

omega_out = ISTF.photoz_pdf['f_out']  # ! can be varied
sigma_in = ISTF.photoz_pdf['sigma_b']  # ! can be varied
sigma_out = ISTF.photoz_pdf['sigma_o']
c_in = ISTF.photoz_pdf['c_b']
c_out = ISTF.photoz_pdf['c_o']
z_in = ISTF.photoz_pdf['z_b']
z_out = ISTF.photoz_pdf['z_o']  # ! can be varied

n_gal = ISTF.other_survey_specs['n_gal']

A_IA = ISTF.IA_free['A_IA']
eta_IA = ISTF.IA_free['eta_IA']
beta_IA = ISTF.IA_free['beta_IA']
C_IA = ISTF.IA_fixed['C_IA']

sqrt2 = np.sqrt(2)
sqrt2pi = np.sqrt(2 * np.pi)

# various parameters for the perturbed photo-z PDF
N_pert = 20
rng = np.random.default_rng()
nu_pert = rng.uniform(-1, 1, N_pert)
nu_pert /= np.sum(nu_pert)  # normalize the weights to 1
zminus_pert = rng.uniform(-0.15, 0.15, N_pert)
zplus_pert = rng.uniform(-0.15, 0.15, N_pert)
omega_fid_pert = rng.uniform(0.69, 0.99)
c_pert = np.ones((N_pert,))


# TODO z_edges[-1] = 2.5?? should it be 4 instead?


# a couple simple functions to go from (z_minus, z_plus) to (zparam, sigma)
def z_n(z_minus, z_plus):
    return -2 * z_minus / (1 + z_minus / z_plus)


def sigma_n(z_minus, z_plus, sigma_in):
    ratio = z_minus / z_plus
    return sigma_in * (1 - ratio) / (1 + ratio)


zparam_pert = z_n(zminus_pert, zplus_pert)
sigma_pert = sigma_n(zminus_pert, zplus_pert, sigma_in)


# n_bar = np.genfromtxt("%s/output/n_bar.txt" % project_path)
# lumin_ratio = np.genfromtxt("%s/input/scaledmeanlum-E2Sa_EXTRAPOLATED.txt" % project_path)
# TODO do not redefine functions here! Take as many as you can from wf.py

####################################### function definition


@njit
def base_gaussian(z_p, z, nu, c, z_param, sigma):
    """one of the terms used int the sum of gaussians
    the name zparam is not the best, but I cannot smply sue z as it is a variable in the function.
    note: the weights (nu) are included in the function! this could change in the future.
    """
    result = (nu * c) / (sqrt2pi * sigma * (1 + z)) * np.exp(-0.5 * ((z - c * z_p - z_param) / (sigma * (1 + z))) ** 2)
    return result


@njit
def pph_fid(z_p, z):
    """nu is just a weight for the sum of gaussians, in this case it's just
     (1 - omega_out) * pph_in + omega_out * pph_out"""
    result = (1 - omega_out) * base_gaussian(z_p, z, nu=1, c=c_in, z_param=z_in, sigma=sigma_in) + \
             omega_out * base_gaussian(z_p, z, nu=1, c=c_out, z_param=z_out, sigma=sigma_out)
    return result


@njit
def pph_pert(z_p, z):
    result = base_gaussian(z_p, z, nu_pert, c_pert, zparam_pert, sigma_pert)
    return np.sum(result)


@njit
def pph_tot(z_p, z, omega_fid=omega_fid_pert):
    return omega_fid * pph_fid(z_p, z) + (1 - omega_fid) * pph_pert(z_p, z)


@njit
def P_out(z_p, z, z_in, z_out):
    return (z_in - z_out) * (2 * z_p - 2 * z + z_in + z_out)  # sigma_in == sigma_out


@njit
def R_out(z_p, z, z_in, z_out, sigma_in, sigma_out):
    print('sum is mising in this function')
    return sigma_in / sigma_out * np.exp(- P_out(z_p, z, z_in, z_out) / (2 * (sigma_in * (1 + z)) ** 2))


@njit
def n(z):
    result = (z / z_0) ** 2 * np.exp(-(z / z_0) ** 1.5)
    return result


# define a grid passing through all the z_edges points, to have exact integration limits


# intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
zp_points = 2_000
zp_num_per_bin = int(zp_points / zbins)
zp_grid = np.empty(0)
zp_bin_grid = np.zeros((zbins, zp_num_per_bin))
for i in range(zbins):
    zp_bin_grid[i, :] = np.linspace(z_edges[i], z_edges[i + 1], zp_num_per_bin)

# alternative way: define a grid, then include the bin edges in the grid to have exact integration limits
zp_bin_grid = np.linspace(z_edges[0], z_edges[-1], zp_points)
zp_bin_grid = np.append(zp_bin_grid, z_edges)  # add bin edges
zp_bin_grid = np.sort(zp_bin_grid)
zp_bin_grid = np.unique(zp_bin_grid)  # remove duplicates (first and last edges were already included)
zp_bin_grid = np.tile(zp_bin_grid, (zbins, 1))  # repeat the grid for each bin (in each row)
for i in range(zbins):  # remove all the points below the bin edge
    zp_bin_grid[i, :] = np.where(zp_bin_grid[i, :] > z_edges[i], zp_bin_grid[i, :], 0)


def niz_unnormalized_simps(z, i, pph):
    """numerator of Eq. (112) of ISTF, with simpson integration"""
    assert type(i) == int, 'zbin_idx must be an integer'
    niz_unnorm_integrand = pph(zp_bin_grid[i, :], z)
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[i, :])
    niz_unnorm_integral *= n(z)
    return niz_unnorm_integral


def niz_unnormalized(z, i, pph):
    """
    :param z: float, does not accept an array. Same as above, but with quad_vec
    """
    assert type(i) == int, 'zbin_idx must be an integer'
    niz_unnorm = quad_vec(pph, z_edges[i], z_edges[i + 1], args=z)[0]
    niz_unnorm *= n(z)
    return niz_unnorm


def niz_normalization(i, niz_unnormalized_func, pph):
    assert type(i) == int, 'zbin_idx must be an integer'
    return quad(niz_unnormalized_func, z_edges[0], z_edges[-1], args=(i, pph))[0]


def niz_normalized(z, zbin_idx, pph):
    """this is a wrapper function which normalizes the result. The if-else is needed not to compute the normalization
    for each z, but only once for each zbin_idx"""

    if type(z) == float or type(z) == int:
        return niz_unnormalized(z, zbin_idx, pph) / niz_normalization(zbin_idx, niz_unnormalized, pph)

    elif type(z) == np.ndarray:
        niz_unnormalized_arr = np.asarray([niz_unnormalized(z_value, zbin_idx, pph) for z_value in z])
        return niz_unnormalized_arr / niz_normalization(zbin_idx, niz_unnormalized, pph)

    else:
        raise TypeError('z must be a float, an int or a numpy array')


def niz_unnorm_stef(z, i):
    """the one used by Stefano in the PyCCL notebook"""
    addendum_1 = erf((z - z_out - c_out * z_edges[i]) / sqrt2 / (1 + z) / sigma_out)
    addendum_2 = erf((z - z_out - c_out * z_edges[i + 1]) / sqrt2 / (1 + z) / sigma_out)
    addendum_3 = erf((z - z_in - c_in * z_edges[i]) / sqrt2 / (1 + z) / sigma_in)
    addendum_4 = erf((z - z_in - c_in * z_edges[i + 1]) / sqrt2 / (1 + z) / sigma_in)

    result = n(z) * 1 / 2 / c_out / c_in * \
             (c_in * omega_out * (addendum_1 - addendum_2) + c_out * (1 - omega_out) * (addendum_3 - addendum_4))
    return result


def mean_z(zbin_idx, pph):
    """mean redshift of the galaxies in the zbin_idx-th bin"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad_vec(lambda z: z * niz_normalized(z, zbin_idx, pph), z_edges[zbin_idx], z_edges[zbin_idx + 1])[0]


z_num = 200
z_grid = np.linspace(z_min, z_max, z_num)

# linspace from rainbow colormap
colors = np.linspace(0, 1, zbins)
cmap = cm.get_cmap('rainbow')
colors = cmap(colors)



plt.figure()
for zbin_idx in range(zbins):
    niz_fid = niz_normalized(z_grid, zbin_idx, pph_fid)
    niz_tot = niz_normalized(z_grid, zbin_idx, pph_tot)
    z_mean = mean_z(zbin_idx, pph_fid)

    plt.plot(z_grid, niz_fid, label='niz_fid', color=colors[zbin_idx])
    plt.plot(z_grid, niz_tot, label='niz_tot', color=colors[zbin_idx], ls='--')
    plt.axvline(z_mean, color=colors[zbin_idx], linestyle='dotted', label='z_mean')

plt.legend()

assert 1 > 2

z_p = 0.2
pph_pert_list = [pph_pert(z_p, z) for z in z_grid]
pph_fid_list = [pph_fid(z_p, z) for z in z_grid]
pph_tot_list = [pph_tot(z_p, z, omega_fid_pert) for z in z_grid]
pph_pert_nosum_list = [base_gaussian(z_p, z, nu_pert, c_pert, zparam_pert, sigma_pert) for z in z_grid]

plt.figure()
plt.plot(z_grid, pph_pert_list, label='pph_pert_list')
plt.plot(z_grid, pph_fid_list, label='pph_fid_list')
plt.plot(z_grid, pph_tot_list, label='pph_tot_list')
plt.plot(z_grid, pph_pert_nosum_list, label='pph_pert_nosum_list')
plt.legend()

print('done')
