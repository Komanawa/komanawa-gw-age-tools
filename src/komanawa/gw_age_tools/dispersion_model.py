"""
created matt_dumont 
on: 3/6/25
"""
import numpy as np


# todo

def dispersion_model(t, tm, dp):
    """
    calculate the pdf of the dispersion model

    :param t: time steps to calculate pdf for (yrs)
    :param tm: mean residence time (yrs)
    :param dp: the dispersion parameter (DP) is the inverse of the Peclet number or the ratio of the dispersion coefficient (D) to the  velocity (v) and outlet position (x). In practice, the dispersion parameter describes the relative width and height of the age distribution and is mainly a measure of the relative importance of dispersion (mixing) to advection (Zuber and Maloszewski, 2001).
    :return:
    """
    t = np.atleast_1d(t).astype(float)

    trel = t/tm
    exponent = -1 * ((1 - trel) ** 2 / (4 * dp * trel))
    out = (1 / tm) * (1 / (np.sqrt(4 * np.pi * dp * trel))) * np.e ** exponent
    return out


def dispersion_model_cdf(t, tm, dp):
    """
    calculate the cdf of the dispersion model

    :param t: time steps to calculate pdf for (yrs)
    :param tm: mean residence time (yrs)
    :param dp: the dispersion parameter (DP) is the inverse of the Peclet number or the ratio of the dispersion coefficient (D) to the  velocity (v) and outlet position (x). In practice, the dispersion parameter describes the relative width and height of the age distribution and is mainly a measure of the relative importance of dispersion (mixing) to advection (Zuber and Maloszewski, 2001).
    :return:
    """
    return np.cumsum(dispersion_model(t, tm, dp) * np.diff(t, prepend=0))

if __name__ == '__main__':
    # reproduce the distribution from TraceLPM
    import matplotlib.pyplot as plt

    t = np.arange(1, 201, 1)
    tm = 25.
    dp = 0.5
    dispersion_model(9.5, tm, dp)
    fig, ax = plt.subplots()
    ax.plot(t, dispersion_model(t, tm, dp))
    ax2 = ax.twinx()
    ax2.plot(t, dispersion_model_cdf(t, tm, dp), color='red')
    plt.show()