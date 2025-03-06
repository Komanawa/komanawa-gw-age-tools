"""
created matt_dumont 
on: 3/6/25
"""
import numpy as np


# todo looks right, but test against tracerlpm?  very had to check as they don't calculate the pdf at the age, but instead the preportion within a window... it's messy...

def partial_exponential_model(t, ts, PEM_ratio):
    """
    calculate the pdf of the partial exponential model

    :param t: time steps to calculate pdf for (yrs)
    :param taq: mean age of young fraction (yrs)
    :param PEM_ratio: The PEM ratio is defined as the ratio of the unsampled thickness of the aquifer to  the sampled thickness z  z  ∗          and is used to calculate the  parameter n in the above equation, which is the ratio of the total thickness to the sampled thickness
    :return:
    """
    assert PEM_ratio >= 0, "PEM_ratio must be greater than or equal to 0"
    assert PEM_ratio <= 1, "PEM_ratio must be less than or equal to 1"
    t = np.atleast_1d(t).astype(float)
    out  = np.zeros(t.shape)
    zs = PEM_ratio  #Z* in the paper
    z = 1 #Z in the paper

    taq = ts/(1-np.log(1-zs/(z+zs)))
    n = PEM_ratio + 1

    idx = t>=(taq * np.log(n))

    out[idx] = n/taq * np.e**(-t[idx]/taq)
    return out

def partial_exponential_model_cdf(t, ts, PEM_ratio):
    """
    calculate the cdf of the partial exponential model

    :param t: time steps to calculate pdf for (yrs)
    :param taq: mean age of young fraction (yrs)
    :param PEM_ratio: The PEM ratio is defined as the ratio of the unsampled thickness of the aquifer to  the sampled thickness z  z  ∗          and is used to calculate the  parameter n in the above equation, which is the ratio of the total thickness to the sampled thickness
    :return:
    """

    return np.cumsum(partial_exponential_model(t, ts, PEM_ratio) * np.diff(t, prepend=0))

if __name__ == '__main__':
    # reproduce the distribution from TraceLPM
    import matplotlib.pyplot as plt
    t = np.arange(0, 100, 0.5)
    ts = 25
    PEM_ratio = 1
    fig, ax = plt.subplots()
    ax.plot(t, partial_exponential_model(t, ts, PEM_ratio))
    ax2 = ax.twinx()
    ax2.plot(t, partial_exponential_model_cdf(t, ts, PEM_ratio), color='red')
    plt.show()