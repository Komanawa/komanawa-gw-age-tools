"""
created matt_dumont 
on: 12/10/23
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gw_age_tools.source_predictions import estimate_source_conc_bepfm, _make_source, predict_future_conc_bepm, \
    make_age_dist, check_age_inputs


def test_from_whakauru(plot=False):
    """
    real world test
    test estimating source concentration from whakauru stream data in the pokaiwhenua catchment
    :return:
    """
    data_path = Path(__file__).parent.joinpath('test_data', 'Whakauru_conc.hdf')
    age = pd.read_hdf(data_path, 'age')
    conc_data = pd.read_hdf(data_path, 'data').set_index('datetime')
    conc_data = conc_data['conc'].sort_index()
    rolling_conc = conc_data.rolling(20, center=True).mean()

    start_conc = 0.1
    adder = -6
    inflect_xlim = [
        pd.to_datetime([f'{2008 + adder}-01-01', f'{2012 + adder}-01-01']),
        pd.to_datetime([f'{2012 + adder}-01-01', f'{2015 + adder}-01-01']),
        pd.to_datetime([f'{2015 + adder}-01-01', f'{2017 + adder}-01-01']),
        pd.to_datetime([f'{2017 + adder}-01-01', f'{2020 + adder}-01-01']),
        pd.to_datetime([f'{2021 + adder}-01-01', f'{2022 + adder}-01-01']),
        pd.to_datetime([f'{2021}-01-01', f'{2022}-01-01']),
    ]
    inflect_ylim = [
        [0.09, 0.11],
        [0.1, 1],
        [0.1, 1],
        [0.1, 5],
        [0.1, 5],
        [1, 5],
    ]
    age_kwargs = dict(
        mrt=12,
        mrt_p1=12,
        mrt_p2=None,
        frac_p1=1,
        precision=2,
        f_p1=0.7, f_p2=0.7,
    )

    pred_source, true_receptor, modelled_receptor = estimate_source_conc_bepfm(
        n_inflections=len(inflect_xlim),
        inflect_xlim=inflect_xlim,
        inflect_ylim=inflect_ylim,
        ts_data=conc_data,
        source_start_conc=start_conc,
        **age_kwargs,
        inflect_start_x=None, inflect_start_y=None
    )

    if plot:
        pred_receptor = modelled_receptor
        assert not pred_receptor.isna().any()
        assert not pred_source.isna().any()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(conc_data.index, conc_data, label='True receptor', color='blue')
        ax.plot(rolling_conc.index, rolling_conc, label='True receptor (rolling, 12)', color='blue', alpha=0.5, )
        ax.plot(pred_receptor.index, pred_receptor, label='Predicted receptor', color='red', alpha=0.5, )
        ax.plot(pred_source.index, pred_source, label='Predicted source', color='k', alpha=0.5, )
        for i in range(len(inflect_xlim)):
            # ys
            ax.plot([inflect_xlim[i][0], inflect_xlim[i][0]], [inflect_ylim[i][0], inflect_ylim[i][1]], color='k',
                    ls=':', alpha=0.5)
            ax.plot([inflect_xlim[i][1], inflect_xlim[i][1]], [inflect_ylim[i][0], inflect_ylim[i][1]], color='k',
                    ls=':', alpha=0.5)
            # xs
            ax.plot([inflect_xlim[i][0], inflect_xlim[i][1]], [inflect_ylim[i][0], inflect_ylim[i][0]], color='k',
                    ls=':', alpha=0.5)
            ax.plot([inflect_xlim[i][0], inflect_xlim[i][1]], [inflect_ylim[i][1], inflect_ylim[i][1]], color='k',
                    ls=':', alpha=0.5)

        ax.set_ylim(0, np.array(inflect_ylim).max() * 1.1)
        ax.set_xlim(conc_data.index.min() - pd.Timedelta(days=365 * 2), conc_data.index.max())

        ax.legend()
        plt.show()
        plt.close(fig)

    raise NotImplementedError
    # todo I got a nice result. it's a bit of a faff is there a better statistical technique
    #  unbounded doesn't work well... kinda need a non-linear solver to get the right answer... pest??? ugg

_test_start_date = pd.to_datetime('2000-01-01')
_test_start_conc = 0.1
_test_pred_max = 10
_test_inflection_pointx = np.array([-15, -10, -5, 0, 5, 15])
_test_inflection_pointy = np.array([_test_start_conc, 1, 2, 4, 5, 10])
assert _test_inflection_pointy.shape == _test_inflection_pointx.shape
_test_mrt = 20
_test_f_p1 = 0.7
_test_f_p2 = 0.7
_test_frac_p1 = 0.5
_test_mrt_p1 = 5
_test_mrt_p2 = None
_test_precision = 2
_test_mrt, _test_mrt_p2 = check_age_inputs(_test_mrt, _test_mrt_p1, _test_mrt_p2,
                                           _test_frac_p1, _test_precision,
                                           _test_f_p1, _test_f_p2)
_test_age_kwargs = dict(
    mrt=_test_mrt,
    mrt_p1=_test_mrt_p1,
    mrt_p2=_test_mrt_p2,
    frac_p1=_test_frac_p1,
    precision=_test_precision,
    f_p1=_test_f_p1, f_p2=_test_f_p2,
)


def _make_synteic_data(plot=False):
    """
    make syntheic data with 5 inflection points
    :return:
    """

    age_step, ages, age_fractions = make_age_dist(**_test_age_kwargs)
    t = np.arange(0, _test_pred_max + age_step, age_step)
    use_args = np.concatenate((_test_inflection_pointx[:, np.newaxis],
                               _test_inflection_pointy[:, np.newaxis]), axis=1).flatten()
    source = _make_source(ages, t, precision=_test_precision, source_start_conc=_test_start_conc,
                          n_inflections=6,
                          args=use_args)
    receptor = predict_future_conc_bepm(
        once_and_future_source_conc=source,
        predict_start=0, predict_stop=_test_pred_max,
        fill_value=_test_start_conc,
        fill_threshold=0.05, pred_step=age_step,
        **_test_age_kwargs)
    source.index = _test_start_date + pd.to_timedelta(source.index * 365.25, unit='d')
    receptor.index = _test_start_date + pd.to_timedelta(receptor.index * 365.25, unit='d')

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(source.index, source, label='source', color='orange')
        ax.plot(receptor.index, receptor, label='receptor', color='blue')
        ax.legend()
        plt.show()
        plt.close(fig)

    return source, receptor


def test_from_synthetic_data3(plot=False):
    source, receptor = _make_synteic_data()

    pred_source, true_receptor, modelled_receptor = estimate_source_conc_bepfm(
        n_inflections=3,
        inflect_xlim=[
            pd.to_datetime(['1990-01-01', '1995-01-01']),
            pd.to_datetime(['1998-01-01', '2003-01-01']),
            pd.to_datetime(['2008-01-01', '2010-01-01']),
        ],
        inflect_ylim=[
            [0.05, 0.15],
            [1, 5],
            [5, 12]
        ],
        ts_data=receptor,

        source_start_conc=_test_start_conc,
        **_test_age_kwargs,
        inflect_start_x=None, inflect_start_y=None
    )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(source.index, source, label='True source', color='orange')
        use_receptor = true_receptor
        pred_receptor = modelled_receptor
        pred_source = source
        ax.plot(use_receptor.index, use_receptor, label='True receptor', color='blue')
        ax.plot(pred_receptor.index, pred_receptor, label='Predicted receptor', color='red', alpha=0.5)
        ax.plot(pred_source.index, pred_source, label='Predicted source', color='k', alpha=0.5, ls='--')

        ax.legend()
        plt.show()
        plt.close(fig)


def test_from_synthetic_data3_unbound(plot=False):
    source, receptor = _make_synteic_data()

    pred_source, true_receptor, modelled_receptor = estimate_source_conc_bepfm(
        n_inflections=3,
        inflect_xlim=[
            pd.to_datetime(['1990-01-01', '1995-01-01']),
            pd.to_datetime(['1998-01-01', '2003-01-01']),
            pd.to_datetime(['2008-01-01', '2010-01-01']),
        ],
        inflect_ylim=[
            [0.05, 0.15],
            [1, 12],
            [1, 12]
        ],
        ts_data=receptor,

        source_start_conc=_test_start_conc,
        **_test_age_kwargs,
        inflect_start_x=None, inflect_start_y=None
    )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(source.index, source, label='True source', color='orange')
        use_receptor = true_receptor
        pred_receptor = modelled_receptor
        ax.plot(use_receptor.index, use_receptor, label='True receptor', color='blue')
        ax.plot(pred_receptor.index, pred_receptor, label='Predicted receptor', color='red', alpha=0.5)
        ax.plot(pred_source.index, pred_source, label='Predicted source', color='k', alpha=0.5, ls='--')

        ax.legend()
        plt.show()
        plt.close(fig)


def test_from_synthetic_data5(plot=False):
    source, receptor = _make_synteic_data()

    pred_source, true_receptor, modelled_receptor = estimate_source_conc_bepfm(
        n_inflections=5,
        inflect_xlim=[
            pd.to_datetime(['1978-01-01', '1995-01-01']),
            pd.to_datetime(['1990-01-01', '1998-01-01']),
            pd.to_datetime(['1998-01-01', '2000-01-01']),
            pd.to_datetime(['2004-01-01', '2006-01-01']),
            pd.to_datetime(['2009-01-01', '2010-01-01']),
        ],
        inflect_ylim=[
            [0.05, 0.15],
            [0.15, 3],
            [2, 4],
            [3, 6],
            [8, 12]
        ],
        ts_data=receptor,

        source_start_conc=_test_start_conc,
        **_test_age_kwargs,
        inflect_start_x=None, inflect_start_y=None
    )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(source.index, source, label='True source', color='orange')
        use_receptor = true_receptor
        pred_receptor = modelled_receptor
        ax.plot(use_receptor.index, use_receptor, label='True receptor', color='blue')
        ax.plot(pred_receptor.index, pred_receptor, label='Predicted receptor', color='red', alpha=0.5)
        ax.plot(pred_source.index, pred_source, label='Predicted source', color='k', alpha=0.5, ls='--')

        ax.legend()
        plt.show()
        plt.close(fig)


# todo make these into proper tests...
if __name__ == '__main__':
    test_from_whakauru_unbounded(True)
    test_from_whakauru(True)
    # _make_synteic_data(True)
    # test_from_synthetic_data3_unbound(True)
    # test_from_synthetic_data3(True)
    test_from_synthetic_data5(True)
