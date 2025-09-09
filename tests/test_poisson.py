import math
import warnings

import numpy as np
import pytest
from scipy.stats import poisson as sp_poisson


from lionheart.features.correction.poisson import Poisson, ZIPoisson


# ---------- helpers --------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(seed=12345)


@pytest.fixture(scope="module")
def samp_small(rng):
    """A small Poisson(μ=3) sample used in several tests."""
    return rng.poisson(lam=3.0, size=10_000)


def _pmf_close_to_one(model, k_max=50, rtol=1e-4):
    """Utility: assert Σₖ pmf(k) ≈ 1."""
    ks = np.arange(0, k_max + 1)
    prob_sum = model.pmf(ks).sum()
    assert math.isclose(prob_sum, 1.0, rel_tol=rtol), prob_sum


# ---------- Poisson --------------------------------------------------


def test_poisson_fit_and_parameters(samp_small):
    p = Poisson().fit(samp_small)
    params = p.get_parameters()
    assert params["n"] == len(samp_small)
    assert math.isclose(params["mu"], samp_small.mean(), rel_tol=1e-9)


@pytest.mark.parametrize("k_values", [0, [0, 1, 2, 5], np.arange(0, 10)])
def test_poisson_pmf_matches_scipy(samp_small, k_values):
    p = Poisson().fit(samp_small)
    print(p.pmf(k_values))
    print(sp_poisson.pmf(k_values, mu=p.mu))
    np.testing.assert_allclose(
        p.pmf(k_values), sp_poisson.pmf(k_values, mu=p.mu), rtol=1e-12
    )


def test_poisson_cdf_monotone(samp_small):
    p = Poisson().fit(samp_small)
    ks = np.arange(0, 20)
    cdf_vals = p.cdf(ks)
    assert np.all(np.diff(cdf_vals) >= 0), cdf_vals
    assert cdf_vals[0] >= 0 and cdf_vals[-1] <= 1


def test_poisson_pmf_sums_to_one(samp_small):
    p = Poisson().fit(samp_small)
    # up to μ + 10√μ captures > 0.999999 of the mass
    k_max = int(p.mu + 10 * math.sqrt(p.mu))
    _pmf_close_to_one(p, k_max=k_max, rtol=1e-6)


def test_poisson_iterator_and_set_pos(samp_small):
    p = Poisson().fit(samp_small)
    start = 7
    p_iter = iter(p)
    p.set_iter_pos(start)
    k, pmf_val, cdf_val = next(p_iter)
    assert k == start
    np.testing.assert_allclose(pmf_val, p.pmf_one(k))
    np.testing.assert_allclose(cdf_val, p.cdf_one(k))


def test_poisson_set_iter_pos_validation(samp_small):
    p = Poisson().fit(samp_small)
    with pytest.raises(TypeError):
        p.set_iter_pos(-1)
    with pytest.raises(TypeError):
        p.set_iter_pos(3.7)


def test_poisson_negative_handling():
    bad = np.array([1, 2, -3, 4])

    # raise
    p = Poisson(handle_negatives="raise")
    with pytest.raises(ValueError):
        p.fit(bad)

    # warn_clip
    p = Poisson(handle_negatives="warn_clip", max_num_negatives=10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p.fit(bad.copy())
        assert any("negative numbers" in str(wi.message) for wi in w)
    assert (p.pmf(0) > 0).all()

    # Clip silently
    p = Poisson(handle_negatives="clip").fit(bad.copy())
    assert p.mu >= 0


def test_poisson_from_parameters(samp_small):
    p1 = Poisson().fit(samp_small)
    params = p1.get_parameters()
    p2 = Poisson.from_parameters(**params)
    np.testing.assert_allclose(p1.pmf(range(10)), p2.pmf(range(10)))
    np.testing.assert_allclose(p1.cdf(range(10)), p2.cdf(range(10)))


# ---------- ZIPoisson ------------------------------------------------


def test_zip_poisson_pmf_cdf_basics(samp_small):
    # create zero inflation by injecting zeros
    data = np.concatenate([np.zeros(5000, dtype=int), samp_small])
    model = ZIPoisson().fit(data)

    # pmf at 0 should be strictly greater than plain Poisson pmf(0; μ)
    base_pmf0 = sp_poisson.pmf(0, model.mu)
    assert model.pmf_one(0) > base_pmf0

    # cdf should start at that pmf
    assert math.isclose(model.cdf_one(0), model.pmf_one(0), rel_tol=1e-12)

    # monotone, bounded
    ks = np.arange(0, 30)
    c = model.cdf(ks)
    assert np.all(np.diff(c) >= 0) and c[-1] <= 1.0


def test_zip_poisson_probabilities_sum_to_one(samp_small):
    data = np.concatenate([np.zeros(3000, dtype=int), samp_small])
    zp = ZIPoisson().fit(data)
    k_max = int(zp.mu + 10 * math.sqrt(zp.mu))
    _pmf_close_to_one(zp, k_max=k_max, rtol=1e-6)


def test_zip_poisson_non_zero_counter(samp_small):
    data = np.concatenate([np.zeros(100, dtype=int), samp_small])
    n_non_zero_expected = (data > 0).sum()
    zp = ZIPoisson().fit(data)
    assert zp.non_zeros == n_non_zero_expected


def test_zip_poisson_from_parameters(samp_small):
    data = np.concatenate([np.zeros(2000, dtype=int), samp_small])
    zp1 = ZIPoisson().fit(data)
    zp2 = ZIPoisson.from_parameters(n=zp1.n, mu=zp1.mu, n_non_zero=zp1.non_zeros)
    np.testing.assert_allclose(zp1.pmf(range(15)), zp2.pmf(range(15)))
    np.testing.assert_allclose(zp1.cdf(range(15)), zp2.cdf(range(15)))


# ---------- both Poisson and ZIPoisson logic ------------------------


def test_scalar_helpers_match_vectorised(samp_small):
    p = Poisson().fit(samp_small)
    for k in [0, 3, 7]:
        assert math.isclose(p.pmf_one(k), p.pmf(k)[0])
        assert math.isclose(p.cdf_one(k), p.cdf(k)[0])

    zp = ZIPoisson().fit(np.concatenate([np.zeros(2000, dtype=int), samp_small]))
    for k in [0, 2, 6]:
        assert math.isclose(zp.pmf_one(k), zp.pmf(k)[0])
        assert math.isclose(zp.cdf_one(k), zp.cdf(k)[0])


# ---------- tail-probability clipping logic ------------------------


def test_tail_probability_cutoff(samp_small):
    """Replicates user-side clipping loop logic."""
    zp = ZIPoisson().fit(samp_small)

    THRESHOLD = 1.0 / 1000  # arbitrary
    start_k = int(np.floor(np.mean(samp_small)))
    zp.set_iter_pos(start_k)

    for k, _, Fk in zp:
        if 1.0 - Fk < THRESHOLD:  # tail prob P(X>k) below threshold
            cutoff = k
            break

    # There should be very few observations above the cutoff
    tail_count = (samp_small > cutoff).sum()
    tail_prob_empirical = tail_count / len(samp_small)
    assert tail_prob_empirical <= THRESHOLD * 1.5  # allow small sampling error
