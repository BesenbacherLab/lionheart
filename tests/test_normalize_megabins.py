# test_normalize_megabins.py
import numpy as np
import pandas as pd
import pytest

from lionheart.features.correction.normalize_megabins import normalize_megabins


# ---------------------------------------------------------------------------
# Helper: build a tiny deterministic DataFrame
# ---------------------------------------------------------------------------
def _toy_df(covers):
    """chromosome=='1', start=[0,1,2,...], coverage=covers"""
    n = len(covers)
    return pd.DataFrame(
        {
            "chromosome": ["1"] * n,
            "start": np.arange(n, dtype=np.int64),
            "coverage": np.asarray(covers, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# No centre/scale → should raise (API forbids no-op)
# ---------------------------------------------------------------------------
def test_noop_raises():
    df = _toy_df([1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        normalize_megabins(df, mbin_size=3, stride=3, center=None, scale=None)


# ---------------------------------------------------------------------------
# Mean-centering works
# ---------------------------------------------------------------------------
def test_normmega_mean_centering():
    # Two megabins of length 3 each:
    #  bin-0 mean = 2, bin-1 mean = 5
    df = _toy_df([1, 2, 3, 4, 5, 6])
    out, _ = normalize_megabins(
        df, mbin_size=3, stride=3, center="mean", scale=None, copy=True
    )
    exp = np.array([-1, 0, 1, -1, 0, 1], dtype=float)  # coverage-mean
    assert np.allclose(out["coverage"].to_numpy(), exp)


# ---------------------------------------------------------------------------
# Mean-centre + std-scale gives per-bin std≈1
# ---------------------------------------------------------------------------
def test_normmega_zscore():
    df = _toy_df([1, 2, 3, 4, 5, 6])
    out, _ = normalize_megabins(
        df, mbin_size=3, stride=3, center="mean", scale="std", copy=True
    )
    # Check each megabin has std == 1
    g = out.groupby(np.digitize(out["start"], [0, 3, 6]))["coverage"]
    for _, arr in g:
        assert np.isclose(arr.std(ddof=0), 1.0)


# ---------------------------------------------------------------------------
# Median-centre + IQR-scale works (IQR of [1,2,3] = 1->2->3 => 2-1 =1)
# ---------------------------------------------------------------------------
def test_normmega_median_iqr():
    df = _toy_df([1, 2, 3])
    out, _ = normalize_megabins(
        df, mbin_size=3, stride=3, center="median", scale="iqr", copy=True
    )
    # After dividing by IQR==1 the numbers should be just centred:
    assert np.allclose(out["coverage"].to_numpy(), np.array([-1, 0, 1]))


# ---------------------------------------------------------------------------
# copy=True leaves the original untouched; copy=False mutates
# ---------------------------------------------------------------------------
def test_normmega_copy_flag():
    df = _toy_df([10, 20, 30])
    normalize_megabins(df, 3, 3, center="mean", copy=True)
    assert "coverage" in df.columns and (df["coverage"] == [10, 20, 30]).all()

    normalize_megabins(df, 3, 3, center="mean", copy=False)
    # now it *is* modified
    assert not np.allclose(df["coverage"].to_numpy(), np.array([10, 20, 30]))


# ---------------------------------------------------------------------------
# return_coverage delivers (np.ndarray, df) in correct order
# ---------------------------------------------------------------------------
def test_normmega_return_coverage_flag():
    df = _toy_df([1, 2, 3])
    cov, agg = normalize_megabins(
        df, 3, 3, center=None, scale="mean", return_coverage=True
    )
    assert isinstance(cov, np.ndarray) and isinstance(agg, pd.DataFrame)
    exp = df["coverage"].to_numpy() / df["coverage"].mean()
    assert np.allclose(cov, exp)


# ---------------------------------------------------------------------------
# stride < mbin_size produces overlapping megabins
# ---------------------------------------------------------------------------
def test_normmega_stride_smaller_than_mbin():
    df = _toy_df(np.arange(1, 10))  # 9 rows
    out, _ = normalize_megabins(  # mbin_size=4, stride=2
        df, mbin_size=4, stride=2, center=None, scale="mean"
    )
    # With two offsets the divisor for row-0 is (2.5 + 1.5)/2 = 2.0 → 1/2 = 0.5
    first_val = out["coverage"].iloc[0]
    assert np.isclose(first_val, 0.5)


# ---------------------------------------------------------------------------
# clip_above_quantile clamps outliers before stats
# ---------------------------------------------------------------------------
def test_normmega_clipping():
    df = _toy_df([1, 2, 100])  # extreme outlier
    _, agg = normalize_megabins(
        df, 3, 3, center=None, scale="mean", clip_above_quantile=0.66
    )
    naive_mean = np.mean([1, 2, 100])
    assert (agg["mbin_overall_mean"] < naive_mean).all()


# ---------------------------------------------------------------------------
# invalid parameter combinations raise
# ---------------------------------------------------------------------------
def test_normmega_invalid_combo():
    with pytest.raises(ValueError):
        normalize_megabins(_toy_df([1]), 1, 1, center="mean", scale="mean")


# ---------------------------------------------------------------------------
# Settings used in LIONHEART
# ---------------------------------------------------------------------------
def test_normmega_as_used():
    df = _toy_df(list(range(30)))
    (
        sample_cov,
        megabin_offset_combination_averages,
    ) = normalize_megabins(
        df=df,
        mbin_size=10,
        stride=3,
        old_col="coverage",
        new_col="coverage",
        center=None,
        scale="mean",
        copy=False,
        return_coverage=True,
    )
    np.testing.assert_allclose(
        sample_cov,
        np.asarray(
            [
                0.0,
                0.33333333,
                0.66666667,
                0.64864865,
                0.86486486,
                1.08108108,
                0.90566038,
                1.05660377,
                1.20754717,
                1.0,
                0.86956522,
                0.95652174,
                1.04347826,
                0.92857143,
                1.0,
                1.07142857,
                0.96969697,
                1.03030303,
                1.09090909,
                1.0,
                0.93023256,
                0.97674419,
                1.02325581,
                0.97354497,
                1.01587302,
                1.05820106,
                1.02463054,
                1.06403941,
                1.10344828,
                1.08411215,
            ]
        ),
    )

    # 12 rows in total (4 stridings × 3 or 4 megabins each)
    assert len(megabin_offset_combination_averages) == 12
    print(sample_cov)
    print(megabin_offset_combination_averages)
    # assert False, "Improve tests"
