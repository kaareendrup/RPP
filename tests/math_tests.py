import pytest
import numpy as np

from RPP.utils.maths.maths import bin_residual_width, polar_shift

# Units test(s)

def test_res_width():

    bin_truths = np.array([0,0,1,1,0,1,0,1,0,0,1,1,1,1,0])
    bin_preds = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9,0.9,0.9,0.8,0.7,0.1])

    mean, bin_w, w_err, converged = bin_residual_width(bin_truths, bin_preds, False)

    assert(np.round(mean,4)==0.4)
    assert(np.round(bin_w,4)==0.514)
    assert(np.round(w_err,4)==0.23)


def test_polar_shift():

    start_phi = np.array([1.5, -2, 1.3, -1.7, 2.2, -0.7])
    phi = start_phi + np.array([2, 3, 5, 1, 2, 4])*2*np.pi

    result = polar_shift(phi)
    assert sum(result - start_phi) < 1e-8
