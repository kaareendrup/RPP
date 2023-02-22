# %% 

import numpy as np
from scipy.stats import norm
from iminuit.util import make_func_code
from iminuit import Minuit, describe

from RPP.utils.maths.utils import gaussian_pdf, fit_output


# Utility functions 

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    

def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


# Chi2

class Chi2Regression: 
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2


# Gaussian fit

def gaussian_fit(data):

    counts, bin_edges = np.histogram(data, bins='fd')
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])

    attempts = 0
    converged = False

    #mu = np.mean(data)
    mu = np.median(data)
    sigma = np.std(data)
    N_approx = np.sqrt(len(data))

    # Try to fit a gaussian with different initial sigmas
    while (not converged) and (attempts < 4):
        k = 1/(2**attempts)

        chi2fit = Chi2Regression(gaussian_pdf, x, y, sy)
        minuit_chi2 = Minuit(chi2fit, N=N_approx, mu=mu, sigma=k*sigma)
        minuit_chi2.migrad(); 

        converged = minuit_chi2.fmin.is_valid
        attempts += 1

    if converged:
        p84, p16 = np.percentile(data, 84)/2, np.percentile(data, 16)/2
        N_fit, mu_fit, sigma_fit = minuit_chi2.values['N'], minuit_chi2.values['mu'], minuit_chi2.values['sigma']

        # Soft check if the distribution is actually Gaussian
        q = 1e-2 / sigma #1e-3
        if (norm.pdf(p84, mu_fit, sigma_fit) > q) and (norm.pdf(p16, mu_fit, sigma_fit) > q):
            #print('Good convergence')
            return N_fit, mu_fit, sigma_fit, converged
        else:
            #print('Fixed convergence')
            #fit_output(0, data, 0, p16, p84, N_fit, mu_fit, sigma_fit)
            converged = False
            return len(data), mu, sigma*2, converged
    else:
        #print("  WARNING: The ChiSquare fit DID NOT converge!!! ")
        return len(data), mu, sigma*2, converged
