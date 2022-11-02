import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import quadpy
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

omega_out = ISTF.photoz_pdf['f_out']
sigma_in = ISTF.photoz_pdf['sigma_b']
sigma_out = ISTF.photoz_pdf['sigma_o']
c_in = ISTF.photoz_pdf['c_b']
c_out = ISTF.photoz_pdf['c_o']
z_in = ISTF.photoz_pdf['z_b']
z_out = ISTF.photoz_pdf['z_o']

n_gal = ISTF.other_survey_specs['n_gal']

A_IA = ISTF.IA_free['A_IA']
eta_IA = ISTF.IA_free['eta_IA']
beta_IA = ISTF.IA_free['beta_IA']
C_IA = ISTF.IA_fixed['C_IA']

simps_z_step_size = 1e-4
sqrt2 = np.sqrt(2)
sqrt2pi = np.sqrt(2 * np.pi)

# various parameters for the perturbed photo-z PDF
N_pert = 20
rng = np.random.default_rng(seed=42)
nu_pert = rng.uniform(-1, 1, N_pert)
nu_pert /= np.sum(nu_pert)  # normalize the weights to 1
zminus_pert = rng.uniform(-0.15, 0.15, N_pert)
zplus_pert = rng.uniform(-0.15, 0.15, N_pert)
omega_fid_pert = rng.uniform(0.69, 0.99)
c_pert = np.ones((N_pert,))


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
    the name zparam is not the best, but I cannot smply sue z as it is a variable in the function
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


def pph_pert(z_p, z):
    result = base_gaussian(z_p, z, nu_pert, c_pert, zparam_pert, sigma_pert)
    return np.sum(result)


def pph_tot(z_p, z, omega_fid=omega_fid_pert):
    return omega_fid * pph_fid(z_p, z) + (1 - omega_fid) * pph_pert(z_p, z)


def P_out(z_p, z, z_in, z_out):
    return (z_in - z_out) * (2 * z_p - 2 * z + z_in + z_out)  # sigma_in == sigma_out


def R_out(z_p, z, z_in, z_out, sigma_in, sigma_out):
    print('sum is mising in this function')
    return sigma_in / sigma_out * np.exp(- P_out(z_p, z, z_in, z_out) / (2 * (sigma_in * (1 + z)) ** 2))


@njit
def n(z):
    # result = n_gal * (z / z_0) ** 2 * np.exp(-(z / z_0) ** 1.5)
    result = (z / z_0) ** 2 * np.exp(-(z / z_0) ** 1.5)
    # TODO normalize the distribution or not? which of the above expressions is correct?
    # result = result*(30/0.4242640687118783) # normalising the distribution?
    return result


################################## niz ##############################################

# choose the cut XXX
# n_i_import = np.genfromtxt("%s/input/Cij-NonLin-eNLA_15gen/niTab-EP10-RB00.dat" %path) # vincenzo (= davide standard, pare)
# n_i_import = np.genfromtxt(path.parent / "common_data/vincenzo/14may/InputNz/niTab-EP10-RB.dat") # vincenzo, more recent (= davide standard, anzi no!!!!)
# n_i_import = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/Cij_davide/output/WFs_v3_cut/niz_e-19cut.txt") # davide e-20cut
# n_i_import = np.load("%s/output/WF/WFs_v2/niz.npy" % project_path)  # davide standard


# n_i_import_2 = np.genfromtxt("%s/output/WF/%s/niz.txt" %(path, WFs_input_folder)) # davide standard with zcutVincenzo


# def n_i_old(z, i):
#     n_i_interp = interp1d(n_i_import[:, 0], n_i_import[:, i + 1], kind="linear")
#     result_array = n_i_interp(z)  # z is considered as an array
#     result = result_array.item()  # otherwise it would be a 0d array
#     return result
#
#
# z_values_from_nz = n_i_import[:, 0]
# i_array = np.asarray(range(zbins))
# n_i_import_cpy = n_i_import.copy()[:, 1:]  # remove redshift column

# note: this is NOT an interpolation in i, which are just the bin indices and will NOT differ from the values 0, 1, 2
# ... 9. The purpose is just to have a 2D vectorized callable.
# n_i_new = interp2d(i_array, z_values_from_nz, n_i_import_cpy, kind="linear")

# note: the normalization of n(z) should be unimportant, here I compute a ratio
# where n(z) is present both at the numerator and denominator!
# as a function, including (of not) the ie-20 cut
# def n_i(z, i):
#     integrand = lambda z_p, z: n(z) * pph(z_p, z)
#     numerator = quad(integrand, z_minus[i], z_plus[i], args=(z))
#     denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])
#     #    denominator = nquad(integrand, [[z_minus[i], z_plus[i]],[z_min, z_max]] )
#     #    return numerator[0]/denominator[0]*3 to have quad(n_i, 0, np.inf = nbar_b/20 = 3)
#     result = numerator[0] / denominator[0]
#     # if result < 6e-19: # remove these 2 lines if you don't want the cut
#     #     result = 0
#     return result


# compute n_i(z) with simpson


# define a grid passing through all the z_edges points, to have exact integration limits

"""
# 500 pts seems to be enough - probably more than what quad uses!
z_grid_norm = np.linspace(z_edges[0], z_edges[-1], 500)
def niz_normalization_simps(i):
    assert type(i) == int, 'i must be an integer'
    integrand = np.asarray([niz_unnorm(z, i) for z in z_grid_norm])
    return simps(integrand, z_grid_norm)
"""

# intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
zp_num = 2_000
zp_num_per_bin = int(zp_num / zbins)
zp_grid = np.empty(0)
zp_bin_grid = np.zeros((zbins, zp_num_per_bin))
for i in range(zbins):
    zp_bin_grid[i, :] = np.linspace(z_edges[i], z_edges[i + 1], zp_num_per_bin)


def niz_unnorm_simps(z, i):
    """numerator of Eq. (112) of ISTF, with simpson integration"""
    assert type(i) == int, 'i must be an integer'
    niz_unnorm_integrand = pph_fid(zp_bin_grid[i, :], z)
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[i, :])
    niz_unnorm_integral *= n(z)
    return niz_unnorm_integral


def niz_unnormalized(z, i, pph):
    """
    :param z: float, does not accept an array
    """
    assert type(i) == int, 'i must be an integer'
    niz_unnorm = quad_vec(pph, z_edges[i], z_edges[i + 1], args=z)[0]
    niz_unnorm *= n(z)
    return niz_unnorm


def niz_normalization(i, niz_unnormalized_func, pph):
    assert type(i) == int, 'i must be an integer'
    return quad(niz_unnormalized_func, z_edges[0], z_edges[-1], args=(i, pph))[0]


def niz_normalized(z, i, pph):
    if type(z) == float or type(z) == int:
        return niz_unnormalized(z, i, pph) / niz_normalization(i, niz_unnormalized, pph)

    elif type(z) == np.ndarray:
        niz_unnormalized_arr = np.asarray([niz_unnormalized(z_value, i, pph) for z_value in z])
        return niz_unnormalized_arr / niz_normalization(i, niz_unnormalized, pph)


def niz_unnorm_stef(z, i):
    """the one used by Stefano in the PyCCL notebook"""
    addendum_1 = erf((z - z_out - c_out * z_edges[i]) / sqrt2 / (1 + z) / sigma_out)
    addendum_2 = erf((z - z_out - c_out * z_edges[i + 1]) / sqrt2 / (1 + z) / sigma_out)
    addendum_3 = erf((z - z_in - c_in * z_edges[i]) / sqrt2 / (1 + z) / sigma_in)
    addendum_4 = erf((z - z_in - c_in * z_edges[i + 1]) / sqrt2 / (1 + z) / sigma_in)

    result = n(z) * 1 / 2 / c_out / c_in * \
             (c_in * omega_out * (addendum_1 - addendum_2) + c_out * (1 - omega_out) * (addendum_3 - addendum_4))
    return result


niz_normalization_arr = np.asarray([niz_normalization(i, niz_unnormalized, pph_fid) for i in range(zbins)])

z_num = 200
z_grid = np.linspace(z_min, z_max, z_num)
zbin_idx = 1

niz_fid_list = [niz_unnormalized(z, zbin_idx, pph_fid) for z in z_grid]
niz_unnorm_list = [niz_unnormalized(z, zbin_idx, pph_tot) for z in z_grid]

niz_norm_pphfid_list = [normalize_niz(z, zbin_idx, niz_unnormalized, pph_fid, niz_normalization_arr) for z in z_grid]
# niz_norm_pphtot_list = [normalize_niz(z, zbin_idx, niz_unnormalized, pph_tot) for z in z_grid]

niz_normalized_new_test = niz_normalized_new(z_grid, zbin_idx, pph_fid)
np.allclose(niz_normalized_new_test, niz_norm_pphfid_list, atol=0, rtol=0.00000001)
plt.figure()
plt.plot(z_grid, niz_norm_pphfid_list, label='niz_norm_pphfid_list')
plt.plot(z_grid, niz_normalized_new_test, label='niz_normalized_new_test')
plt.legend()

assert 1 > 2

# step 1 - let's try to build the perturbed n(z) as a sum of 20 Gaussians

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
