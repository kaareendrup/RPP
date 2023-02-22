import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from RPP.utils.maths.fits import gaussian_fit
from RPP.utils.maths.utils import fit_output


def bin_residual_width(bin_truths, bin_preds, verbose):
    
    # Calculate residual distribution and percentiles
    bin_res = bin_truths - bin_preds
    p84, p16 = np.percentile(bin_res, 84)/2, np.percentile(bin_res, 16)/2
    bin_w = p84 - p16

    # Fit gaussian and make error bars
    N, mu, sigma, converged = gaussian_fit(bin_res)
    p84_err = 1 / norm.pdf(p84, mu, sigma) * np.sqrt( (0.84*(1-0.84)) / len(bin_res) )
    p16_err = 1 / norm.pdf(p16, mu, sigma) * np.sqrt( (0.16*(1-0.16)) / len(bin_res) )
    w_err = np.sqrt(p84_err**2 + p16_err**2)

    # Plot for troubleshooting
    if verbose or (not converged and len(bin_res) > 1000):# or w_err > 100 * bin_w:
        fit_output(bin_truths, bin_res, w_err, p16, p84, N, mu, sigma)

    return np.mean(np.abs(bin_res)), bin_w, w_err, converged


def w_errorprop(w_compare, w_model, err_compare, err_model):
    return np.sqrt( w_model**2/w_compare**4 * err_compare + 1/w_compare**2 * err_model )

