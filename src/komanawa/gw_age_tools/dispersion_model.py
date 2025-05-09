"""
created matt_dumont 
on: 3/6/25
"""
import numpy as np


# todo not sure why this doesn't match, but I think it has to do with the integration implementation... I need to better understand..

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

    t = np.arange(1, 201, 0.1).astype(float)
    tm = 25.
    dp = 0.5
    dispersion_model(12.5, tm, dp)
    # todo the peak I'm getting is too wide
    fig, ax = plt.subplots()
    dm_output = dispersion_model(t, tm, dp)
    t2= np.arange(1, 200, 1)
    dm_output2 = dispersion_model(t2, tm, 1)
    dm_output_cdf = dispersion_model_cdf(t, tm, dp)
    ax.plot(t/tm, dm_output)
    # ax2 = ax.twinx()
    # ax2.plot(t, dispersion_model_cdf(t, tm, dp), color='red')
    plt.show()