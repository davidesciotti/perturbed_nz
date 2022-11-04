import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import ray
import time
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

ray.shutdown()
ray.init()

start = time.perf_counter()

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


N_pert = 20
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
rng = np.random.default_rng()
nu_pert = rng.uniform(-1, 1, N_pert)
nu_pert /= np.sum(nu_pert)  # normalize the weights to 1
z_minus_pert = rng.uniform(-0.15, 0.15, N_pert)
z_plus_pert = rng.uniform(-0.15, 0.15, N_pert)
omega_fid = rng.uniform(0.69, 0.99)  # ! should this be called omega_fid_pert?
c_pert = np.ones((N_pert,))
nu_out = np.ones((N_pert,))  # weights for Eq. (7)


# TODO z_edges[-1] = 2.5?? should it be 4 instead?
# TODO do not redefine functions here! Take as many as you can from wf.py

# a couple simple functions to go from (z_minus_case, z_plus_case) to (z_case, sigma_case);
# _case should be _out, _n or _eff
@njit
def z_case_func(z_minus_case, z_plus_case):
    return -2 * z_minus_case / (1 + z_minus_case / z_plus_case)


@njit
def sigma_case_func(z_minus_case, z_plus_case, sigma_in):
    ratio = z_minus_case / z_plus_case
    return sigma_in * (1 - ratio) / (1 + ratio)


# the opposite: functions to go from (z_case, sigma_case) to (z_minus_case, z_plus_case)
@njit
def z_minus_case_func(sigma_in, z_in, sigma_case, z_case):
    return -(sigma_in * z_case + sigma_case * z_in) / (sigma_case + sigma_in)


@njit
def z_plus_case_func(sigma_in, z_in, sigma_case, z_case):
    return (sigma_in * z_case - sigma_case * z_in) / (sigma_case - sigma_in)


z_pert = z_case_func(z_minus_pert, z_plus_pert)
sigma_pert = sigma_case_func(z_minus_pert, z_plus_pert, sigma_in)


# and maybe
# z_eff = z_case(z_minus_eff, z_plus_eff)
# sigma_eff = sigma_case(z_minus_eff, z_plus_eff, sigma_in)


####################################### function definition


@njit
def n(z):
    result = (z / z_0) ** 2 * np.exp(-(z / z_0) ** 1.5)
    return result


@njit
def base_gaussian(z_p, z, nu_case, c_case, z_case, sigma_case):
    """one of the terms used int the sum of gaussians
    in this function, _case can be _out, _n or _eff or also _in
    note: the weights (nu_case) are included in the function! this could change in the future.
    """
    result = (nu_case * c_case) / (sqrt2pi * sigma_case * (1 + z)) * np.exp(
        -0.5 * ((z - c_case * z_p - z_case) / (sigma_case * (1 + z))) ** 2)
    return result


@njit
def pph_fid(z_p, z):
    """nu_case is just a weight for the sum of gaussians, in this case it's just
     (1 - omega_out) * pph_in + omega_out * pph_out"""
    result = (1 - omega_out) * base_gaussian(z_p, z, nu_case=1, c_case=c_in, z_case=z_in, sigma_case=sigma_in) + \
             omega_out * base_gaussian(z_p, z, nu_case=1, c_case=c_out, z_case=z_out, sigma_case=sigma_out)
    return result


@njit
def pph_pert(z_p, z):
    result = base_gaussian(z_p, z, nu_pert, c_pert, z_pert, sigma_pert)
    return np.sum(result)


@njit
def pph_true(z_p, z, omega_fid=omega_fid):
    return omega_fid * pph_fid(z_p, z) + (1 - omega_fid) * pph_pert(z_p, z)


##################################################### P functions ######################################################

# @njit
def P(z_p, z, zbin_idx, z_case, sigma_case, z_in, sigma_in):
    """ parameters named with ..._case shpuld be _out, _n or _eff"""
    assert type(zbin_idx) == int, "zbin_idx must be an integer"

    if sigma_case == sigma_in:

        # I don't need these parameters in this case
        return (z_in - z_case[zbin_idx]) * (2 * z_p - 2 * z + z_in + z_case[zbin_idx])

    else:
        print('sigma_case, sigma_in = ', sigma_case, sigma_in)

        z_minus_case = z_minus_case_func(sigma_in, z_in, sigma_case, z_case)
        z_plus_case = z_plus_case_func(sigma_in, z_in, sigma_case, z_case)

        # I don't need these parameters in this case
        assert z_case is None, 'the function does not need the z_case parameter if sigma_in is not equal to sigma_out'
        assert z_in is None, 'the function does not need the z_in parameter if sigma_in is not equal to sigma_out'
        assert np.allclose(sigma_in, sigma_case, atol=0, rtol=1e-4) is False, 'sigma_in must not be equal to sigma_case'

        result = (z_p - z - z_minus_case[zbin_idx]) * (z_p - z - z_plus_case[zbin_idx]) * \
                 ((sigma_case[zbin_idx] / sigma_in[zbin_idx]) ** (-2) - 1)
        return result


# these are just convenience wrapper functions
# @njit
def P_eff(z_p, z, zbin_idx):
    # ! these 3 paramenters have to be found by solving Eqs. 16 to 19
    return P(z_p, z, zbin_idx, z_eff, sigma_eff, z_in, sigma_in)


# @njit
def P_out(z_p, z, zbin_idx):
    return P(z_p, z, zbin_idx, z_out, sigma_out, z_in, sigma_in)


# @njit
def P_pert(z_p, z, zbin_idx):
    return P(z_p, z, zbin_idx, z_pert, sigma_pert, z_in, sigma_in)


##################################################### R functions ######################################################
# @njit
def R(z_p, z, zbin_idx, nu_case, z_case, sigma_case, z_in, sigma_in):
    P_shortened = P(z_p, z, zbin_idx, z_case, sigma_case, z_in, sigma_in)

    result = nu_case[zbin_idx] * sigma_in[zbin_idx] / sigma_case[zbin_idx] * np.exp(
        - P_shortened / (2 * (sigma_in[zbin_idx] * (1 + z)) ** 2))
    return result


# @njit
def R_pert(z_p, z, zbin_idx):
    to_sum = [R(z_p, z, zbin_idx, nu_n, z_pert, sigma_pert, z_in, sigma_in) for nu_n in nu_pert]
    return np.sum(to_sum)


# @njit
def R_out(z_p, z, zbin_idx):
    return R(z_p, z, zbin_idx, nu_out, z_out, sigma_out, z_in, sigma_in)


# define a grid passing through all the z_edges points, to have exact integration limits


# intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
zp_points = 500
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


def niz_unnormalized_simps(z, zbin_idx, pph):
    """numerator of Eq. (112) of ISTF, with simpson integration"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    niz_unnorm_integrand = pph(zp_bin_grid[zbin_idx, :], z)
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[zbin_idx, :])
    niz_unnorm_integral *= n(z)
    return niz_unnorm_integral


def niz_unnormalized(z, zbin_idx, pph):
    """
    :param z: float, does not accept an array. Same as above, but with quad_vec
    """
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    niz_unnorm = quad_vec(pph, z_edges[zbin_idx], z_edges[zbin_idx + 1], args=z)[0]
    niz_unnorm *= n(z)
    return niz_unnorm


def niz_normalization(zbin_idx, niz_unnormalized_func, pph):
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad(niz_unnormalized_func, z_edges[0], z_edges[-1], args=(zbin_idx, pph))[0]


@ray.remote
def niz_normalized_ray(z, zbin_idx, pph):  # ! the only difference with the one below is the decorator!
    """this is a wrapper function which normalizes the result. The if-else is needed not to compute the normalization
    for each z, but only once for each zbin_idx"""

    if type(z) == float or type(z) == int:
        return niz_unnormalized(z, zbin_idx, pph) / niz_normalization(zbin_idx, niz_unnormalized, pph)

    elif type(z) == np.ndarray:
        niz_unnormalized_arr = np.asarray([niz_unnormalized(z_value, zbin_idx, pph) for z_value in z])
        return niz_unnormalized_arr / niz_normalization(zbin_idx, niz_unnormalized, pph)

    else:
        raise TypeError('z must be a float, an int or a numpy array')


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
    return quad_vec(lambda z: z * niz_normalized(z, zbin_idx, pph), z_edges[0], z_edges[-1])[0]


@ray.remote
def mean_z_simps(zbin_idx, pph):
    """mean redshift of the galaxies in the zbin_idx-th bin; faster version with simpson integration"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    z_grid = np.linspace(z_edges[0], z_edges[-1], 500)
    integrand = z_grid * niz_normalized(z_grid, zbin_idx, pph)
    return simps(y=integrand, x=z_grid)


# check: construct niz_true using R and P, that is, implement Eq. (5)
# TODO make this function moore generic and/or use simps?
def integrand(z_p, z, zbin_idx):
    integrand = 1 + omega_out / (1 - omega_out) * R_out(z_p, z, zbin_idx) + (1 - omega_fid) / \
                ((1 - omega_out) * omega_fid) * R_pert(z_p, z, zbin_idx) * \
                base_gaussian(z_p, z, nu_case=1, c_case=c_in, z_case=z_in, sigma_case=sigma_in)
    return integrand


@ray.remote
def niz_true_RP_ray(z, zbin_idx):
    return omega_fid * (1 - omega_out) * n(z) * quad_vec(integrand, z_min, z_max, args=(z, zbin_idx))[0]


z_num = 200
z_grid = np.linspace(z_min, z_max, z_num)

# linspace from rainbow colormap
colors = np.linspace(0, 1, zbins)
colors = cm.get_cmap('rainbow')(colors)

niz_fid = np.zeros((zbins, z_num))
niz_true = np.zeros((zbins, z_num))
niz_true_RP = np.zeros((zbins, z_num))
niz_shifted = np.zeros((zbins, z_num))
zmean_fid = np.zeros(zbins)
zmean_tot = np.zeros(zbins)

for zbin_idx in range(zbins):
    niz_fid[zbin_idx, :] = ray.get(niz_normalized_ray.remote(z_grid, zbin_idx, pph_fid))
    niz_true[zbin_idx, :] = ray.get(niz_normalized_ray.remote(z_grid, zbin_idx, pph_true))
    niz_true_RP[zbin_idx, :] = ray.get(niz_true_RP_ray.remote(z_grid, zbin_idx))
    zmean_fid[zbin_idx] = ray.get(mean_z_simps.remote(zbin_idx, pph_fid))
    zmean_tot[zbin_idx] = ray.get(mean_z_simps.remote(zbin_idx, pph_true))

delta_z = zmean_tot - zmean_fid  # ! free to vary, zbins additional parameters
for zbin_idx in range(zbins):
    niz_shifted[zbin_idx, :] = niz_normalized(z_grid - delta_z[zbin_idx], zbin_idx, pph_fid)

plt.figure()
linestyles = ['-', '--', ':']

label_switch = 1  # this is to display the labels only for the first iteration
for zbin_idx in range(zbins):
    if zbin_idx != 0:
        label_switch = 0
    plt.plot(z_grid, niz_fid[zbin_idx, :], label='niz_fid' * label_switch, color=colors[zbin_idx], ls=linestyles[0])
    plt.plot(z_grid, niz_true[zbin_idx, :], label='niz_true' * label_switch, color=colors[zbin_idx], ls=linestyles[1])
    plt.plot(z_grid, niz_shifted[zbin_idx, :], label='niz_shifted' * label_switch, color=colors[zbin_idx],
             ls=linestyles[1])
    plt.axvline(zmean_fid[zbin_idx], label='zmean_fid' * label_switch, color=colors[zbin_idx], ls=linestyles[2])
    plt.axvline(zmean_tot[zbin_idx], label='zmean_tot' * label_switch, color=colors[zbin_idx], ls=linestyles[2])

plt.legend()

# nicer legend
# dummy_lines = []
# linestyle_labels = ['fiducial', 'shifted/true', 'zmean']
# for i in range(len(linestyles)):
#     dummy_lines.append(plt.plot([], [], c_case="black", ls=linestyles[i])[0])
#
# linestyles_legend = plt.legend(dummy_lines, linestyle_labels)
# plt.gca().add_artist(linestyles_legend)

print(f'******* done in {(time.perf_counter() - start):.2f} s *******')

assert 1 > 2

z_p = 0.2
pph_pert_list = [pph_pert(z_p, z) for z in z_grid]
pph_fid_list = [pph_fid(z_p, z) for z in z_grid]
pph_tot_list = [pph_tot(z_p, z, omega_fid) for z in z_grid]
pph_pert_nosum_list = [base_gaussian(z_p, z, nu_pert, c_pert, z_pert, sigma_pert) for z in z_grid]

plt.figure()
plt.plot(z_grid, pph_pert_list, label='pph_pert_list')
plt.plot(z_grid, pph_fid_list, label='pph_fid_list')
plt.plot(z_grid, pph_tot_list, label='pph_tot_list')
plt.plot(z_grid, pph_pert_nosum_list, label='pph_pert_nosum_list')
plt.legend()

print('done')
