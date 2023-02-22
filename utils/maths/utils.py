import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def gaussian_pdf(x, N, mu, sigma):
    return N * norm.pdf(x, mu, sigma)


def fit_output(bin_truths, bin_res, w_err, p16, p84, N, mu, sigma):

    print(np.mean(bin_truths))

    print('Found bad converence! Values:')
    print(w_err)
    print(p84)
    print(norm.pdf(p84, mu, sigma))
    print(p16)
    print(norm.pdf(p16, mu, sigma))
    print(N, mu, sigma)

    plt.close()
    fig, ax = plt.subplots(figsize=(9,6))
    ax.hist(bin_res, bins='fd', color='lightgray')
    x = np.linspace(np.max(bin_res), np.min(bin_res), 200)
    y = gaussian_pdf(x, N, mu, sigma)
    plt.plot(x,y)

    ax.set_xlabel('R', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlim(-170,100)

    plt.show()
    plt.close()
