import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import ray
import time
import numpy as np
import logging
from matplotlib import cm
from mpire import WorkerPool
from numba import njit
from scipy.integrate import quad, quad_vec, simpson, dblquad, simps
from scipy.special import erf

project_path = Path.cwd().parent

sys.path.append(str(project_path))

# project modules
# import proj_lib.cosmo_lib as csmlb
# import config.config as cfg
# general configuration modules
sys.path.append(str(project_path.parent / 'common_data/common_config'))
import ISTF_fid_params as ISTF
import mpl_cfg as mpl_cfg

sys.path.append(str(project_path.parent / 'common_data/common_lib'))
import my_module as mm

# update plot paramseters
rcParams = mpl_cfg.mpl_rcParams_dict
plt.rcParams.update(rcParams)
matplotlib.use('Qt5Agg')

ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, dashboard_host='0.0.0.0')
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

z_edges = ISTF.photoz_bins['all_zbin_edges']
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
rng = np.random.default_rng(seed=42)
nu_pert = rng.uniform(-1, 1, N_pert)
nu_pert /= np.sum(nu_pert)  # normalize the weights to 1
z_minus_pert = rng.uniform(-0.15, 0.15, N_pert)
z_plus_pert = rng.uniform(-0.15, 0.15, N_pert)
omega_fid = rng.uniform(0.69, 0.99)  # ! should this be called omega_fid_pert?
c_pert = np.ones(N_pert)
nu_out = 1.  # weights for Eq. (7)
nu_in = 1.

zero_cut = 1e-4  # ! dangerous?
manual_zmax = 4.


# TODO z_edges[-1] = 2.5?? should it be 4 instead?
# TODO do not redefine functions here! Take as many as you can from wf.py

# a couple simple functions to go from (z_minus_case, z_plus_case) to (z_case, sigma_case);
# _case should be _out, _n or _eff. These have been obtained by solving the system of equations in the paper (Eq. 9)
@njit
def z_case_func(z_minus_case, z_plus_case, z_in):
    return -(z_in * z_minus_case + z_in * z_plus_case + 2 * z_minus_case * z_plus_case) / (
            2 * z_in + z_minus_case + z_plus_case)


@njit
def sigma_case_func(z_minus_case, z_plus_case, z_in, sigma_in):
    return -sigma_in * (z_minus_case - z_plus_case) / (2 * z_in + z_minus_case + z_plus_case)


# the opposite: functions to go from (z_case, sigma_case) to (z_minus_case, z_plus_case)
@njit
def z_minus_case_func(sigma_in, z_in, sigma_case, z_case):
    return -(sigma_in * z_case + sigma_case * z_in) / (sigma_case + sigma_in)


@njit
def z_plus_case_func(sigma_in, z_in, sigma_case, z_case):
    return (sigma_in * z_case - sigma_case * z_in) / (sigma_case - sigma_in)


z_pert = z_case_func(z_minus_pert, z_plus_pert, z_in)
sigma_pert = sigma_case_func(z_minus_pert, z_plus_pert, z_in, sigma_in)


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

    # if np.abs(result) < zero_cut:
    #     return 0.

    # ! XXX I'm here, trying to understand how to implement the cut in an efficient way

    return result


@njit
def pph_in(z_p, z):
    return (1 - omega_out) * base_gaussian(z_p, z, nu_in, c_in, z_in, sigma_in)


@njit
def pph_out(z_p, z):
    return omega_out * base_gaussian(z_p, z, nu_out, c_out, z_out, sigma_out)


@njit
def pph_fid(z_p, z):
    return pph_in(z_p, z) + pph_out(z_p, z)


# @njit
def pph_pert(z_p, z):
    """this function is vectorized in 'z_p', not in the '_pert' input arrays"""
    tosum = np.array([base_gaussian(z_p, z, nu_pert[i], c_pert[i], z_pert[i], sigma_pert[i]) for i in range(N_pert)])
    return np.sum(tosum, axis=0)


# @njit
def pph_true(z_p, z):
    return omega_fid * pph_fid(z_p, z) + (1 - omega_fid) * pph_pert(z_p, z)


# not yet used!
pph_true_ray = ray.remote(pph_true)
pph_pert_ray = ray.remote(pph_pert)


##################################################### P functions ######################################################

# @njit
def P(z_p, z, zbin_idx, z_case, sigma_case, z_in, sigma_in):
    """
    parameters named as "<name>_case" shpuld be "_out", "_n" or "_eff"
    """
    assert type(zbin_idx) == int, "zbin_idx must be an integer"

    if np.allclose(sigma_case, sigma_in, atol=0, rtol=1e-5):
        return (z_in - z_case[zbin_idx]) * (2 * z_p - 2 * z + z_in + z_case[zbin_idx])

    assert np.allclose(sigma_in, sigma_case, atol=0,
                       rtol=1e-5) is False, 'sigma_in must not be equal to sigma_<case>'

    z_minus_case = z_minus_case_func(sigma_in, z_in, sigma_case, z_case)
    z_plus_case = z_plus_case_func(sigma_in, z_in, sigma_case, z_case)
    print('z_minus_case', z_minus_case)

    result = (z_p - z - z_minus_case[zbin_idx]) * (z_p - z - z_plus_case[zbin_idx]) * \
             ((sigma_case[zbin_idx] / sigma_in) ** (-2) - 1)
    return result


# these are just convenience wrapper functions
# @njit
def P_eff(z_p, z, zbin_idx):
    # ! these 3 paramenters have to be found by solving Eqs. 16 to 19
    return P(z_p, z, zbin_idx, z_eff, sigma_eff, z_in, sigma_in)


# @njit
def P_out(z_p, z, zbin_idx):
    return P(z_p, z, zbin_idx, z_out * np.ones(N_pert), sigma_out, z_in, sigma_in)


# @njit
def P_pert(z_p, z, zbin_idx):
    return P(z_p, z, zbin_idx, z_pert, sigma_pert, z_in, sigma_in)


##################################################### R functions ######################################################
# @njit
def R_old(z_p, z, zbin_idx, nu_case, z_case, sigma_case, z_in, sigma_in):
    P_shortened = P(z_p, z, zbin_idx, z_case, sigma_case, z_in, sigma_in)  # just to make the equation more readable
    result = nu_case * sigma_in / sigma_case[zbin_idx] * np.exp(-0.5 * P_shortened / (sigma_in * (1 + z)) ** 2)
    return result


def R(z_p, z, zbin_idx, nu_case, c_case, z_case, sigma_case, rtol=1e-6):
    print(z_p, z, zbin_idx, 'osad', nu_in, c_in, z_in, sigma_in, base_gaussian(z_p, z, nu_in, c_in, z_in, sigma_in))
    print('the problem is that base_gaussian for z very far from z_p gives 0, at least I think')
    numerator = base_gaussian(z_p, z, nu_case[zbin_idx], c_case[zbin_idx], z_case[zbin_idx], sigma_case[zbin_idx])
    denominator = base_gaussian(z_p, z, nu_in, c_in, z_in, sigma_in)

    # if both are very close to 0, return 0
    # if np.allclose(numerator, 0, atol=0, rtol=rtol) and np.allclose(denominator, 0, atol=0, rtol=rtol):
    #     return 0

    # smarter alternatives
    # if abs(x - mean2) > (3 * std2 + abs(mean1 - mean2)):
    #     return 0
    # else:
    #     return ratio

    # or
    # log_ratio = np.log10(numerator) - np.log10((denominator))
    # if log_ratio < np.log10(zero_cut):
    #     return 0
    # return np.exp(log_ratio)

    # TODO these are probably too slow
    # either
    if np.abs(numerator) < zero_cut and np.abs(denominator) < zero_cut:
        return 0

    # or
    # try:
    #     return numerator / denominator
    # except ZeroDivisionError:
    #     print('inside the try statement', numerator, denominator)
    #     return np.nan  # should be 0

    # or
    # if denominator == 0:
    #     return 0

    return numerator / denominator




def R_test(z_p, z, zbin_idx, nu_case, c_case, z_case, sigma_case):
    """just to see what happens to the components of the ratio, i.e. where it explodes"""
    print(z_p, z, zbin_idx, 'osad', nu_in, c_in, z_in, sigma_in, base_gaussian(z_p, z, nu_in, c_in, z_in, sigma_in))
    print('the problem is that base_gaussian for z very far from z_p gives 0, at least I think')
    numerator = base_gaussian(z_p, z, nu_case[zbin_idx], c_case[zbin_idx], z_case[zbin_idx],
                              sigma_case[zbin_idx])
    denominator = base_gaussian(z_p, z, nu_in, c_in, z_in, sigma_in)
    return numerator, denominator


# @njit
def R_pert(z_p, z, zbin_idx):
    assert type(nu_pert) == np.ndarray, "nu_pert must be an array"
    return np.sum(R(z_p, z, zbin_idx, nu_pert, c_pert, z_pert, sigma_pert))


# @njit
def R_out(z_p, z, zbin_idx):
    # sigma_out and z_out are scalars, I vectorize them to make the function work with the P function
    return R(z_p, z, zbin_idx, nu_out * np.ones(N_pert), c_out * np.ones(N_pert), z_out * np.ones(N_pert),
             sigma_out * np.ones(N_pert))


# intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
# equal number of points per bin
zp_points = 500
zp_points_per_bin = int(zp_points / zbins)
zp_bin_grid = np.zeros((zbins, zp_points_per_bin))
for i in range(zbins):
    zp_bin_grid[i, :] = np.linspace(z_edges[i], z_edges[i + 1], zp_points_per_bin)

# more pythonic way of instantiating the same grid
# zp_bin_grid = np.linspace(z_min, z_max, zp_points)
# zp_bin_grid = np.append(zp_bin_grid, z_edges)  # add bin edges
# zp_bin_grid = np.sort(zp_bin_grid)
# zp_bin_grid = np.unique(zp_bin_grid)  # remove duplicates (first and last edges were already included)
# zp_bin_grid = np.tile(zp_bin_grid, (zbins, 1))  # repeat the grid for each bin (in each row)
# for i in range(zbins):  # remove all the points below the bin edge
#     zp_bin_grid[i, :] = np.where(zp_bin_grid[i, :] > z_edges[i], zp_bin_grid[i, :], 0)

# alternative: equispaced grid with z_edges added (does *not* work well, needs a lot of samples!!)
zp_grid = np.linspace(z_min, z_max, 4000)
zp_grid = np.concatenate((z_edges, zp_grid))
zp_grid = np.unique(zp_grid)
zp_grid = np.sort(zp_grid)
# indices of z_edges in zp_grid:
z_edges_idxs = np.array([np.where(zp_grid == z_edges[i])[0][0] for i in range(z_edges.shape[0])])


def niz_unnormalized_simps(z_grid, zbin_idx, pph):
    """numerator of Eq. (112) of ISTF, with simpson integration
    Not too fast (3.0980 s for 500 z_p points)"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'  # TODO check if these slow down the code using scalene
    niz_unnorm_integrand = np.array([pph(zp_bin_grid[zbin_idx, :], z) for z in z_grid])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[zbin_idx, :], axis=1)
    niz_unnorm_integral *= n(z_grid)
    return niz_unnorm_integral


# def niz_unnormalized_simps(z, zbin_idx, pph):
#     """numerator of Eq. (112) of ISTF, with simpson integration"""
#     assert type(zbin_idx) == int, 'zbin_idx must be an integer'
#     niz_unnorm_integrand = np.array([pph(z, zp_bin_grid[zbin_idx, :]) for z in z_grid])
#     niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[zbin_idx, :], axis=1)
#     niz_unnorm_integral *= n(z_grid)  # ! z_grid?
#     return niz_unnorm_integral


def niz_unnormalized_simps_fullgrid(z_grid, zbin_idx, pph):
    """numerator of Eq. (112) of ISTF, with simpson integration and "global" grid"""
    warnings.warn('this function does not work well, needs very high number of samples;'
                  ' the zp_bin_grid sampling is better')
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    z_minus = z_edges_idxs[zbin_idx]
    z_plus = z_edges_idxs[zbin_idx + 1]
    niz_unnorm_integrand = np.array([pph(zp_grid[z_minus:z_plus], z) for z in z_grid])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_grid[z_minus:z_plus], axis=1)
    return niz_unnorm_integral * n(z_grid)


def quad_integrand(z_p, z, pph):
    return n(z) * pph(z_p, z)


def niz_unnormalized_quad(z, zbin_idx, pph):
    """with quad - 0.620401143 s, faster than quadvec..."""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad(quad_integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=(z, pph))[0]


def niz_unnormalized_quadvec(z, zbin_idx, pph):
    """
    :param z: float, does not accept an array. Same as above, but with quad_vec.
    ! the difference is that the integrand can be a vector-valued function (in this case in z_p),
    so it's supposedly faster? -> no, it's slower - 5.5253 s
    """
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    niz_unnorm = quad_vec(quad_integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=(z, pph))[0]
    return niz_unnorm


def niz_unnormalized_analytical(z, zbin_idx):
    """the one used by Stefano in the PyCCL notebook
    by far the fastest, 0.009592 s"""
    addendum_1 = erf((z - z_out - c_out * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_out))
    addendum_2 = erf((z - z_out - c_out * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_out))
    addendum_3 = erf((z - z_in - c_in * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_in))
    addendum_4 = erf((z - z_in - c_in * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_in))

    result = n(z) / (2 * c_out * c_in) * \
             (c_in * omega_out * (addendum_1 - addendum_2) + c_out * (1 - omega_out) * (addendum_3 - addendum_4))
    return result


def niz_normalization_quad(niz_unnormalized_func, zbin_idx, pph):
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad(niz_unnormalized_func, z_min, z_max, args=(zbin_idx, pph))[0]


def normalize_niz_simps(niz_unnorm_arr, z_grid):
    """ much more convenient; uses simps, and accepts as input an array of shape (zbins, z_points)"""
    norm_factor = simps(niz_unnorm_arr, z_grid)
    niz_norm = (niz_unnorm_arr.T / norm_factor).T
    return niz_norm


def niz_normalized(z, zbin_idx, pph):
    """this is a wrapper function which normalizes the result.
    The if-else is needed not to compute the normalization for each z, but only once for each zbin_idx
    Note that the niz_unnormalized_quadvec function is not vectorized in z (its 1st argument)
    """
    warnings.warn("this function should be deprecated")
    if type(z) == float or type(z) == int:
        return niz_unnormalized_quadvec(z, zbin_idx, pph) / niz_normalization_quad(niz_unnormalized_quadvec, zbin_idx,
                                                                                   pph)

    elif type(z) == np.ndarray:
        niz_unnormalized_arr = np.asarray([niz_unnormalized_quadvec(z_value, zbin_idx, pph) for z_value in z])
        return niz_unnormalized_arr / niz_normalization_quad(niz_unnormalized_quadvec, zbin_idx, pph)

    else:
        raise TypeError('z must be a float, an int or a numpy array')


def mean_z(zbin_idx, pph):
    """mean redshift of the galaxies in the zbin_idx-th bin"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad_vec(lambda z: z * niz_normalized(z, zbin_idx, pph), z_min, z_max)[0]


def mean_z_simps(zbin_idx, pph, zsteps=500):
    """mean redshift of the galaxies in the zbin_idx-th bin; faster version with simpson integration"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    z_grid = np.linspace(z_min, z_max, zsteps)
    integrand = z_grid * niz_normalized(z_grid, zbin_idx, pph)
    return simps(y=integrand, x=z_grid)


# check: construct niz_true using R and P, that is, implement Eq. (5)
# TODO make this function more generic and/or use simps?
def niz_true_RP_integrand(z_p, z, zbin_idx):
    integrand = 1 + omega_out / (1 - omega_out) * R_out(z_p, z, zbin_idx) + (1 - omega_fid) / \
                ((1 - omega_out) * omega_fid) * R_pert(z_p, z, zbin_idx) * \
                base_gaussian(z_p, z, nu_case=1, c_case=c_in, z_case=z_in, sigma_case=sigma_in)
    return integrand


def niz_true_RP(z, zbin_idx):
    return omega_fid * (1 - omega_out) * n(z) * quad_vec(niz_true_RP_integrand, z_min, z_max, args=(z, zbin_idx))[0]


def loop_zbin_idx_ray(function, zbins=zbins, **kwargs):
    """
    convenience function; eg, shortens this line of code:
    zmean_fid_2 = np.asarray(ray.get([mean_z_simps_ray.remote(zbin_idx, pph_fid) for zbin_idx in range(zbins)]))
    """
    remote_function = ray.remote(function)
    return np.asarray(ray.get([remote_function.remote(zbin_idx=zbin_idx, **kwargs) for zbin_idx in range(zbins)]))


########################################################################################################################
########################################################################################################################
########################################################################################################################

z_num = 500
z_grid = np.linspace(0, manual_zmax, z_num)

# linspace from rainbow colormap
colors = np.linspace(0, 1, zbins)
colors = cm.get_cmap('rainbow')(colors)

# ! problem: niz_true_RP(0.001, 1) is nan, for example. Let's try with these parameters.

# compute n_i(z) for each zbin, for the various pph
# niz_fid = np.array([[niz_unnormalized_quad(z, zbin_idx, pph_fid) for z in z_grid] for zbin_idx in range(zbins)])
# niz_true = np.array([[niz_unnormalized_quad(z, zbin_idx, pph_true) for z in z_grid] for zbin_idx in range(zbins)])
# start = time.perf_counter()
# niz_true_RP_arr = np.array([[niz_true_RP(z, zbin_idx) for z in z_grid] for zbin_idx in range(zbins)])
# print('done!')

# normalize the arrays
# niz_fid = normalize_niz_simps(niz_fid, z_grid)
# niz_true = normalize_niz_simps(niz_true, z_grid)

# compute z shifts
zmean_fid = loop_zbin_idx_ray(mean_z_simps, pph=pph_fid)
zmean_true = loop_zbin_idx_ray(mean_z_simps, pph=pph_true)
delta_z = zmean_true - zmean_fid  # ! free to vary, in this case there will be zbins additional parameters

zmean_true = loop_zbin_idx_ray(mean_z_simps, pph=pph_true)
niz_shifted = np.asarray([[niz_unnormalized_quad(z - delta_z[zbin_idx], zbin_idx, pph_fid) for z in z_grid]
                          for zbin_idx in range(zbins)])

# niz_true_RP_arr[zbin_idx, :] = [ray.get(niz_true_RP_ray.remote(z, zbin_idx)) for z in z_grid]




# XXX 10jan working here

for z_test in (0.01, 0.1, 0.5, 1, 1.5,  2):
    R_test_arr = np.asarray([R_test(z_p, z_test, 0, nu_pert, c_pert, z_pert, sigma_pert) for z_p in z_grid])
    R_arr = np.asarray([R(z_p, z_test, 0, nu_pert, c_pert, z_pert, sigma_pert) for z_p in z_grid])

    plt.figure()
    plt.plot(z_grid, np.abs(R_test_arr[:, 0]), label='numerator')
    plt.plot(z_grid, np.abs(R_test_arr[:, 1]), label='denominator')
    plt.plot(z_grid, np.abs(R_arr), label='ratio')
    plt.yscale('log')
    plt.title('z_test = {}'.format(z_test))





assert 1 > 2

lnstl = ['-', '--', ':']
label_switch = 1  # this is to display the labels only for the first iteration
plt.figure()
for zbin_idx in range(zbins):
    if zbin_idx != 0:
        label_switch = 0
    # plt.plot(z_grid, niz_fid[zbin_idx, :], label='niz_fid' * label_switch, color=colors[zbin_idx], ls=lnstl[0])
    # plt.plot(z_grid, niz_true[zbin_idx, :], label='niz_true' * label_switch, color=colors[zbin_idx], ls=lnstl[1])
    plt.plot(z_grid, niz_true_RP_arr[zbin_idx, :], label='niz_true_RP_arr' * label_switch, color=colors[zbin_idx],
             ls='--')
    plt.plot(z_grid, niz_shifted[zbin_idx, :], label='niz_shifted' * label_switch, color=colors[zbin_idx], ls=lnstl[1])
    plt.axvline(zmean_fid[zbin_idx], label='zmean_fid' * label_switch, color=colors[zbin_idx], ls=lnstl[2])
    plt.axvline(zmean_true[zbin_idx], label='zmean_true' * label_switch, color=colors[zbin_idx], ls=lnstl[2])

plt.legend()

# nicer legend
# dummy_lines = []
# linestyle_labels = ['fiducial', 'shifted/true', 'zmean']
# for i in range(len(lnstl)):
#     dummy_lines.append(plt.plot([], [], c_case="black", ls=lnstl[i])[0])
#
# linestyles_legend = plt.legend(dummy_lines, linestyle_labels)
# plt.gca().add_artist(linestyles_legend)


ray.shutdown()

print(f'******* done in {(time.perf_counter() - start):.2f} s *******')

"""assert 1 > 2

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
"""
