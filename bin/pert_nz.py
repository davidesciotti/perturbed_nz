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
from numba import njit
from scipy.integrate import quad, quad_vec, simpson, dblquad, simps
from scipy.special import erf
import multiprocessing

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

sys.path.append(f'{project_path.parent}/cl_v2/bin')
import wf_cl_lib

# update plot paramseters
rcParams = mpl_cfg.mpl_rcParams_dict
plt.rcParams.update(rcParams)
matplotlib.use('Qt5Agg')

ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, dashboard_host='0.0.0.0')

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
# rng = np.random.default_rng(seed=42)
rng = np.random.default_rng()
nu_pert = rng.uniform(-1, 1, N_pert)
nu_pert /= np.sum(nu_pert)  # normalize the weights to 1
z_minus_pert = rng.uniform(-0.15, 0.15, N_pert)
z_plus_pert = rng.uniform(-0.15, 0.15, N_pert)
omega_fid = rng.uniform(0.69, 0.99)  # ! should this be called omega_fid_pert?
c_pert = np.ones(N_pert)
nu_out = 1.  # weights for Eq. (7)
nu_in = 1.

gaussian_zero_cut = 1e-26  # ! dangerous?
# TODO remove this cut on P? better at a higher level (aka R_with_P function)?
max_P_cut = 250  # ! dangerous?
exponent_cut = 50  # ! dangerous?
manual_zmax = 4.

assert type(nu_pert) == np.ndarray, "nu_pert must be an array"


# TODO z_edges[-1] = 2.5?? should it be 4 instead?
# TODO do not redefine functions here! Take as many as you can from wf.py
# TODO check different values of gaussian_zero_cut

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

# this is wrong: z_eff and sigma_eff are considered independent, we don't use these 2 relations:
# z_eff = z_case_func(z_minus_eff, z_plus_eff)
# sigma_eff = sigma_case_func(z_minus_eff, z_plus_eff, sigma_in)


####################################### function definition


n_of_z = wf_cl_lib.n_of_z  # n(z)


# TODO the problem is z_out in R_with_P


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


# ! not yet used!
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


##################################################### R_no/with_P functions ######################################################
R_with_p_list = []


# @njit
def R_with_P(z_p, z, zbin_idx, nu_case, z_case, sigma_case, z_in, sigma_in):
    """ In this case, I implement the factorization by Vincenzo, with the P convenience function.
    This does not work yet."""

    P_shortened = P(z_p, z, zbin_idx, z_case, sigma_case, z_in, sigma_in)  # just to make the equation more readable
    exponent = -0.5 * P_shortened / (sigma_in * (1 + z)) ** 2

    # cut the exponent to avoid numerical problems
    exponent = np.sign(exponent) * np.minimum(np.abs(exponent), exponent_cut)

    # if np.abs(exponent) > exponent_cut:
    #     exponent = np.sign(exponent) * exponent_cut

    result = nu_case * sigma_in / sigma_case[zbin_idx] * np.exp(exponent)
    R_with_p_list.append(result)
    return result


def R_no_P(z_p, z, zbin_idx, nu_case, c_case, z_case, sigma_case, gaussian_zero_cut=gaussian_zero_cut):
    """ In this case, I simply take the ratio of Gaussians, without using the P function"""
    numerator = base_gaussian(z_p, z, nu_case[zbin_idx], c_case[zbin_idx], z_case[zbin_idx], sigma_case[zbin_idx])
    denominator = base_gaussian(z_p, z, nu_in, c_in, z_in, sigma_in)

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

    # ! XXX this is quite delicate, check it a bit more
    if (np.abs(numerator) < gaussian_zero_cut and np.abs(denominator) < gaussian_zero_cut) or denominator == 0:
        return 0
    return numerator / denominator


# @njit
def R_pert(z_p, z, zbin_idx, R_func=R_no_P):
    if R_func == R_no_P:
        return np.sum(R_func(z_p, z, zbin_idx, nu_pert, c_pert, z_pert, sigma_pert))
    elif R_func == R_with_P:
        return np.sum(R_func(z_p, z, zbin_idx, nu_pert, z_pert, sigma_pert, z_in, sigma_in))


# @njit
def R_out(z_p, z, zbin_idx, R_func=R_no_P):  # is this supposed to have a sum?
    # sigma_out and z_out are scalars, I vectorize them to make the function work with the P function
    if R_func == R_no_P:
        return R_func(z_p, z, zbin_idx, nu_out * np.ones(N_pert), c_out * np.ones(N_pert), z_out * np.ones(N_pert),
                      sigma_out * np.ones(N_pert))
    elif R_func == R_with_P:
        return R_func(z_p, z, zbin_idx, nu_out * np.ones(N_pert), z_out*np.ones(N_pert), sigma_out * np.ones(N_pert),
                      z_in, sigma_in)[0] # this is not very nice, the function o   utputs an array with


#################################################### niz functions #####################################################


# with simpson integration
# intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
# equal number of points per bin
zp_bin_grid = wf_cl_lib.zp_bin_grid
niz_unnormalized_simps = wf_cl_lib.niz_unnormalized_simps

# alternative: equispaced grid with z_edges added (does *not* work well, needs a lot of samples!!)
zp_grid = wf_cl_lib.zp_grid
z_edges_idxs = wf_cl_lib.z_edges_idxs

niz_unnormalized_simps_fullgrid = wf_cl_lib.niz_unnormalized_simps_fullgrid

# other methods
quad_integrand = wf_cl_lib.quad_integrand
niz_unnormalized_quad = wf_cl_lib.niz_unnormalized_quad
niz_unnormalized_quadvec = wf_cl_lib.niz_unnormalized_quadvec
niz_unnormalized_analytical = wf_cl_lib.niz_unnormalized_analytical
niz_normalization_quad = wf_cl_lib.niz_normalization_quad
normalize_niz_simps = wf_cl_lib.normalize_niz_simps
niz_normalized = wf_cl_lib.niz_normalized  # ! wrong, does not actually accept pph as argument


def mean_z(zbin_idx, pph):
    warnings.warn("the use of niz_normalized should be deprecated")
    """mean redshift of the galaxies in the zbin_idx-th bin"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    warnings.warn("is it necessary to use niz_normalized?")
    return quad(lambda z: z * niz_normalized(z, zbin_idx, pph), z_min, z_max)[0]


def mean_z_simps(zbin_idx, pph, zsteps=500):
    warnings.warn("the use of niz_normalized should be deprecated")
    """mean redshift of the galaxies in the zbin_idx-th bin; faster version with simpson integration"""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    z_grid = np.linspace(z_min, z_max, zsteps)
    integrand = z_grid * niz_normalized(z_grid, zbin_idx, pph)
    return simps(y=integrand, x=z_grid)


# check: construct niz_true using R_no_P and P, that is, implement Eq. (5)
# TODO make this function more generic and/or use simps?
niz_true_RP_integrand_list = []


def niz_true_RP_integrand(z_p, z, zbin_idx, R_func=R_no_P):
    # try:
    # is it R_out(z_p, z, zbin_idx)? that is, with R_func=R_no_P?
    integrand = (1 + omega_out / (1 - omega_out) * R_out(z_p, z, zbin_idx, R_func) +
                 (1 - omega_fid) / ((1 - omega_out) * omega_fid) * R_pert(z_p, z, zbin_idx, R_func)) * \
                base_gaussian(z_p, z, nu_case=1, c_case=c_in, z_case=z_in, sigma_case=sigma_in)
    # except:
    #     print(R_pert(z_p, z, zbin_idx, R_func))
    #     print(R_func(z_p, z, zbin_idx, nu_pert, z_pert, sigma_pert, z_in, sigma_in))

    # niz_true_RP_integrand_list.append(integrand)
    return integrand


def niz_true_RP(z, zbin_idx, R_func=R_no_P):
    return omega_fid * (1 - omega_out) * n_of_z(z) * \
        quad(niz_true_RP_integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=(z, zbin_idx, R_func))[0]


def loop_zbin_idx_ray(function, zbins=zbins, **kwargs):
    """
    convenience function; eg, shortens this line of code:
    zmean_fid_2 = np.asarray(ray.get([mean_z_simps_ray.remote(zbin_idx, pph_fid) for zbin_idx in range(zbins)]))

    I can use it also for a doulbe loop, e.g. these two are equivalent:

    niz_fid = np.array([[niz_unnormalized_quad(z, zbin_idx, pph_fid) for z in z_grid] for zbin_idx in range(zbins)])
    niz_fid_ray = np.asarray([loop_zbin_idx_ray(niz_unnormalized_quad, z=z, pph=pph_fid) for z in z_grid]).T  # (you have to transpose!)

    however, the second case is slower, probably because I'm calling ray.get len(z_grid) times.
    It would probably be better to parallelize the z_grid loop...

    """
    remote_function = ray.remote(function)
    return np.asarray(ray.get([remote_function.remote(zbin_idx=zbin_idx, **kwargs) for zbin_idx in range(zbins)]))


niz_true_RP_ray = ray.remote(niz_true_RP)
########################################################################################################################
########################################################################################################################
########################################################################################################################

# z_num = 200
z_num = 500
if z_num != 500:
    warnings.warn("restore z_num = 500")
z_grid = np.linspace(0, manual_zmax, z_num)

# * compute n_i(z) for each zbin, for the various pph. This is quite fast.
niz_fid = np.array([[niz_unnormalized_quad(z, zbin_idx, pph_fid) for z in z_grid] for zbin_idx in range(zbins)])

# three ways to compute niz_true: just plugging the true pph, using R = phi/out/phi/in, using Eq.(7) (i.e, with the P
# convenience funcion). These 3 should match.
niz_true = np.array([[niz_unnormalized_quad(z, zbin_idx, pph_true) for z in z_grid] for zbin_idx in range(zbins)])
# with ray
# niz_true_RnoP_arr = np.array([ray.get([niz_true_RP_ray.remote(z=z, zbin_idx=zbin_idx) for z in z_grid])
#                               for zbin_idx in range(zbins)])
# without ray
niz_true_RnoP_arr = np.array([[niz_true_RP(z=z, zbin_idx=zbin_idx) for z in z_grid] for zbin_idx in range(zbins)])
# with ray
# niz_true_RwithP_arr = np.array([ray.get([niz_true_RP_ray.remote(z=z, zbin_idx=zbin_idx, R_func=R_with_P) for z in z_grid])
#                               for zbin_idx in range(zbins)])
# without ray
start_time = time.perf_counter()
niz_true_RwithP_arr = np.array([[niz_true_RP(z=z, zbin_idx=zbin_idx, R_func=R_with_P) for z in z_grid]
                                for zbin_idx in range(zbins)])
print(f"niz_true_RwithP_arr took {time.perf_counter() - start_time} seconds")
# with multiprocessing, just to check whether ray is the problem

# niz_true_RwithP_arr = np.zeros(niz_true_RnoP_arr.shape)
# for z_idx, z in enumerate(z_grid):
#     niz_true_RP_partial = partial(niz_true_RP, z=z, R_func=R_with_P)
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         niz_true_RwithP_arr[:, z_idx] = np.array(pool.map(niz_true_RP, range(zbins)))


plt.figure()
# plt.plot(R_with_p_list, label='R_with_p_list')
plt.plot(niz_true_RP_integrand_list, label='niz_true_RP_integrand_list')
plt.show()
plt.legend()

# * normalize the arrays
niz_fid = normalize_niz_simps(niz_fid, z_grid)
niz_true = normalize_niz_simps(niz_true, z_grid)
niz_true_RnoP_arr = normalize_niz_simps(niz_true_RnoP_arr, z_grid)
niz_true_RwithP_arr = normalize_niz_simps(niz_true_RwithP_arr, z_grid)

# * compute z shifts
# TODO fix ray
# zmean_fid = loop_zbin_idx_ray(mean_z_simps, pph=pph_fid)  # ray gives problems
# zmean_fid = np.array([mean_z_simps(zbin_idx, pph=pph_fid, zsteps=500) for zbin_idx in range(zbins)])  # no ray problems but mean_z_simps is wrong (no pph!!)
# zmean_true = loop_zbin_idx_ray(mean_z_simps, pph=pph_true)
zmean_fid = simps(z_grid * niz_fid, z_grid, axis=1)
zmean_true = simps(z_grid * niz_true, z_grid, axis=1)
delta_z = zmean_true - zmean_fid  # ! free to vary, in this case there will be zbins additional parameters

# * compute niz_shifted
niz_shifted = np.asarray([[niz_unnormalized_quad(z - delta_z[zbin_idx], zbin_idx, pph_fid) for z in z_grid]
                          for zbin_idx in range(zbins)])
niz_shifted = normalize_niz_simps(niz_shifted, z_grid)

# * plot everything
# linspace from rainbow colormap
colors = np.linspace(0, 1, zbins)
colors = cm.get_cmap('rainbow')(colors)

lnstl = ['-', '--', ':']
label_switch = 1  # this is to display the labels only for the first iteration
plt.figure()
for zbin_idx in range(zbins):
    if zbin_idx != 0:
        label_switch = 0
    # plt.plot(z_grid, niz_fid[zbin_idx, :], label='niz_fid' * label_switch, c=colors[zbin_idx], ls=lnstl[0])
    plt.plot(z_grid, niz_true[zbin_idx, :], label='niz_true' * label_switch, c=colors[zbin_idx], ls=lnstl[1])
    plt.plot(z_grid, niz_true_RnoP_arr[zbin_idx, :], label='niz_true_RnoP_arr' * label_switch, c=colors[zbin_idx],
             ls='-', alpha=0.6)
    plt.plot(z_grid, niz_true_RwithP_arr[zbin_idx, :], label='niz_true_RwithP_arr' * label_switch,
             c=colors[zbin_idx], ls='-.', alpha=0.6)
    # plt.plot(z_grid, niz_shifted[zbin_idx, :], label='niz_shifted' * label_switch, c=colors[zbin_idx], ls=lnstl[1])
    # plt.axvline(zmean_fid[zbin_idx], label='zmean_fid' * label_switch, c=colors[zbin_idx], ls=lnstl[2])
    # plt.axvline(zmean_true[zbin_idx], label='zmean_true' * label_switch, c=colors[zbin_idx], ls=lnstl[2])
plt.legend()

# nicer legend
# dummy_lines = []
# linestyle_labels = ['fiducial', 'shifted/true', 'zmean']
# for i in range(len(lnstl)):
#     dummy_lines.append(plt.plot([], [], c_case="black", ls=lnstl[i])[0])
#
# linestyles_legend = plt.legend(dummy_lines, linestyle_labels)
# plt.gca().add_artist(linestyles_legend)


print(f'******* done in {(time.perf_counter() - script_start):.2f} s *******')
ray.shutdown()

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
