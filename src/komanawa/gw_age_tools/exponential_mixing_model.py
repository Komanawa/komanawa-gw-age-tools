"""
created matt_dumont 
on: 3/6/25
"""

import numpy as np

# todo looks right, but test against tracerlpm?

def exponential_mixing_model(t, tm):
    """
    :param t: time steps to calculate pdf for (yrs)
    :param tm: mean residence time (yrs)
    :return:
    """
    t = np.atleast_1d(t).astype(float)
    out = 1/tm * np.e**(-t/tm)
    return out

def exponential_mixing_model_cdf(t, tm):
    """
    :param t: time steps to calculate pdf for (yrs)
    :param tm: mean residence time (yrs)
    :return:
    """
    t = np.atleast_1d(t).astype(float)
    out = np.cumsum(exponential_mixing_model(t, tm) * np.diff(t, prepend=0))
    return out




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.arange(0, 150, 1)
    tm = 25
    fig, ax = plt.subplots()
    ax.plot(t, exponential_mixing_model(t, tm))
    ax2 = ax.twinx()
    ax2.plot(t, exponential_mixing_model_cdf(t, tm), color='red')
    plt.show()