from numbers import Number
from typing import List, Optional, Tuple, Union, Dict
import warnings
import numpy as np
from scipy.stats import poisson


class Poisson:
    def __init__(
        self, handle_negatives: str = "raise", max_num_negatives: Optional[int] = None
    ) -> None:
        """
        Calculate Poisson probabilities for nonnegative integers.

        Get the probability mass function (PMF) and the cumulative distribution function (CDF).

        Wrapper for `scipy.stats.poisson`. See its `.pmf()` and `.cdf()` methods.

        See `ZIPoisson` for handling zero-inflated data.

        Parameters
        ----------
        handle_negatives : str
            How to handle negative numbers (e.g., numeric versions of NaN).
            One of: {"raise", "warn_clip", "clip"}.
        max_num_negatives : int or `None`
            How many negative numbers to allow when
            `handle_negatives` is not `"raise"`.
        """
        self.handle_negatives = handle_negatives
        self.max_num_negatives = max_num_negatives
        self.n: int = 0
        self.mu: float = 0.0
        self._iter_n: int = 0

    def get_parameters(self) -> Dict[str, Number]:
        """
        Get the fitted parameters `n` and `mu` in a dict.
        """
        return {"n": self.n, "mu": self.mu}

    @staticmethod
    def from_parameters(
        n: int,
        mu: float,
        handle_negatives="raise",
        max_num_negatives: Optional[int] = None,
    ):
        """
        Create new `Poisson` model with existing parameters.

        Parameters
        ----------
        n
            Number of data points (positive integers).
        mu
            Mean of the data points.
        handle_negatives, max_num_negatives
            See `Poisson.__init__()`.

        Returns
        -------
        `Poisson`
            Poisson with the specified values.
        """
        m = Poisson(
            handle_negatives=handle_negatives,
            max_num_negatives=max_num_negatives,
        )
        m.n = n
        m.mu = mu
        return m

    def set_iter_pos(self, pos: int = 0) -> None:
        """
        Set the iterator position.
        This value will be returned on the next call to `next()`, unless changed in-between.

        Note: The position is reset to 0 when calling `__iter__()` (see ) or `fit()`.

        Parameters
        ----------
        pos : int
            A non-negative integer to set the current position to.
        """
        if not isinstance(pos, int) or pos < 0:
            raise TypeError("`pos` must be a non-negative integer.")
        self._iter_n = pos

    def __iter__(self):
        """
        Get iterator for generating integers from 0 -> inf
        along with their probability.

        Iteration is reset on `.fit()`.

        Tip: Use `.set_iter_pos()` to set a starting position
        after calling `iter()`.
        """
        self._check_is_fit()
        self._iter_n = 0
        return self

    def __next__(self) -> Tuple[int, float, float]:
        """
        Get next integer, its PMF probability, and it CDF probability.

        Returns
        -------
        tuple
            int
                k
            float
                PMF: probability of seeing k
            float
                CDF: probability of seeing <= k

        """
        self._check_is_fit()
        self._iter_n += 1
        return (
            self._iter_n - 1,
            self.pmf_one(self._iter_n - 1),
            self.cdf_one(self._iter_n - 1),
        )

    def _reset(self):
        """
        Reset all fitted parameters and the iterator position.
        """
        self.n = 0
        self.mu = 0
        self._iter_n = 0

    def reset(self):
        """
        Reset the distribution parameters.

        Returns
        -------
        self
        """
        self._reset()
        return self

    def fit(self, x: np.ndarray):
        """
        Fit the distribution.

        In case the distribution was already fitted, parameters are
        reset first. Use `.partial_fit()` instead to update an existing fit.

        Parameters
        ----------
        x : `numpy.ndarray`
            The 1D array to fit the Poisson distribution to.

        Returns
        -------
        self
        """
        self._reset()
        return self.partial_fit(x)

    def partial_fit(self, x: np.ndarray):
        """
        Partially fit the distribution. Previous fittings are respected.

        Parameters
        ----------
        x : `numpy.ndarray`
            The 1D array to fit the Poisson distribution to.
            All elements must be non-negative.

        Returns
        -------
        self
        """
        assert isinstance(x, np.ndarray) and x.ndim == 1
        if np.any(x < 0):
            if self.handle_negatives == "raise" or (
                self.max_num_negatives is not None
                and np.sum(x < 0) > self.max_num_negatives
            ):
                raise ValueError(self._str_negative_numbers(x=x))
            elif self.handle_negatives == "warn_clip":
                warnings.warn(self._str_negative_numbers(x=x))
                x[x < 0] = 0
            elif self.handle_negatives == "clip":
                x[x < 0] = 0

        self._update_mean(x=x, old_mu=self.mu, old_n=self.n)
        self.n += len(x)
        return self

    def _str_negative_numbers(self, x):
        """
        Format negative numbers message.
        """
        negative_indices = np.argwhere(x < 0)
        example_negs = x[x < 0].flatten()[:5]
        dots = ", ..." if len(example_negs) < 5 else ""
        examples_str = ", ".join([str(n) for n in example_negs]) + dots
        return (
            f"`x` contained {len(negative_indices)} negative numbers: "
            f"{examples_str} at indices: {negative_indices[:5]}{dots}"
        )

    def _update_mean(self, x: np.ndarray, old_mu: float, old_n: int) -> None:
        """
        Update mean with new data.
        """
        new_mu = np.mean(x)
        new_n = len(x)
        self.mu = (old_mu * old_n + new_mu * new_n) / (old_n + new_n)

    def pmf_one(self, k: int) -> float:
        """
        Probability Mass Function of a single value.

        Identical to `.pmf(ks=[k])[0]`.

        Parameters
        ----------
        k : int
            Must be non-negative.

        Returns
        -------
        float
            Probability of `k`.
        """
        return float(self.pmf(ks=[k])[0])

    def cdf_one(self, k: int) -> float:
        """
        Cumulative Distribution Function of a single value.

        Identical to `.cdf(ks=[k])[0]`.

        Parameters
        ----------
        k : int
            Must be non-negative.

        Returns
        -------
        float
            CDF probability of `k`.
        """
        return float(self.cdf(ks=[k])[0])

    def pmf(self, ks: Union[List[int], np.ndarray, range, int]) -> np.ndarray:
        """
        Probability Mass Function.

        Get probability of one or more values.

        Parameters
        ----------
        ks : int, range or list of ints
            Must be non-negative.

        Returns
        -------
        `numpy.ndarray` with floats
            Probability for each value in `ks`.
        """
        self._check_is_fit()
        ks = self._check_ks(ks=ks)
        return np.asarray(poisson.pmf(ks, mu=self.mu))

    def cdf(self, ks: Union[List[int], np.ndarray, range, int]) -> np.ndarray:
        """
        Cumulative Distribution Function.

        Get probability of <= k for one or more values.

        Use `1 - cdf` to get the tail probability p(X > k) for outlier detection.

        Parameters
        ----------
        ks : int, range or list of ints
            Must be non-negative.

        Returns
        -------
        `numpy.ndarray` with floats
            CDF probability for each value in `ks`.
        """
        self._check_is_fit()
        ks = self._check_ks(ks)
        return np.asarray(poisson.cdf(ks, mu=self.mu))

    def _check_ks(self, ks):
        """
        Check the `ks` argument.
        """
        if isinstance(ks, range):
            ks = list(ks)
        if isinstance(ks, Number):
            ks = [ks]
        ks = np.asarray(ks, dtype=int)
        if ks.ndim != 1 or np.any(ks < 0):
            raise ValueError(
                "`ks` must be scalar or 1-D list/array and "
                "contain non-negative integers."
            )
        return ks

    def _check_is_fit(self):
        """
        Check whether the class has been fitted.
        """
        if self.n == 0:
            raise RuntimeError(f"{self.__class__.__name__}: `.fit()` not called.")


class ZIPoisson(Poisson):
    def __init__(
        self, handle_negatives="raise", max_num_negatives: Optional[int] = None
    ) -> None:
        """
        Calculate zero-inflated Poisson probabilities for nonnegative integers.

        Get the probability mass function (PMF) and the cumulative distribution function (CDF).


        Probability mass function:

            P(X = k) = { (1 − p(X > 0)) + p(X > 0) * pmf(k, mean(X))   if k = 0
                       { p(X > 0) * pmf(k, mean(X))                    if k > 0


        Cumulative distribution function:

            p(X <= k) = (1 - p(X > 0)) + p(X > 0) * cdf(k, mean(X))    for k ≥ 0


        Parameters
        ----------
        handle_negatives : str
            How to handle negative numbers (e.g., numeric versions of NaN).
            One of: {"raise", "warn_clip", "clip"}.
        max_num_negatives : int or `None`
            How many negative numbers to allow when
            `handle_negatives` is not `"raise"`.
        """
        super().__init__(
            handle_negatives=handle_negatives, max_num_negatives=max_num_negatives
        )
        self.non_zeros: int = 0

    def get_parameters(self) -> Dict[str, Number]:
        """
        Get the fitted parameters `n`, `mu`, and `n_non_zero` in a dict.
        """
        parameters = super().get_parameters()
        parameters["n_non_zero"] = self.non_zeros
        return parameters

    @staticmethod
    def from_parameters(
        n: int,
        mu: float,
        n_non_zero: int,
        handle_negatives="raise",
        max_num_negatives: Optional[int] = None,
    ):
        """
        Create new `ZIPoisson` model with existing parameters.

        Parameters
        ----------
        n
            Number of data points (positive integers).
        mu
            Mean of the data points.
        n_non_zero
            Number on non-zero data points.
        handle_negatives, max_num_negatives
            See `ZIPoisson.__init__()`.

        Returns
        -------
        `ZIPoisson`
            Zero-inflated Poisson with the specified values.
        """
        m = ZIPoisson(
            handle_negatives=handle_negatives, max_num_negatives=max_num_negatives
        )
        m.n = n
        m.mu = mu
        m.non_zeros = n_non_zero
        return m

    def _reset(self):
        """
        Reset all fitted parameters and the iterator position.
        """
        super()._reset()
        self.non_zeros = 0

    def partial_fit(self, x: np.ndarray):
        """
        Partially fit the distribution. Previous fittings are respected.

        Parameters
        ----------
        x : `numpy.ndarray`
            The 1D array to fit the Poisson distribution to.
            All elements must be non-negative.

        Returns
        -------
        self
        """
        super().partial_fit(x=x)
        self.non_zeros += np.count_nonzero(x)
        return self

    def pmf(self, ks: Union[List[int], np.ndarray, range, int]) -> np.ndarray:
        """
        Probability Mass Function.

        Get zero-inflated probability of one or more values.

        Parameters
        ----------
        ks : int, range or list of ints
            Must be non-negative.

        Returns
        -------
        `numpy.ndarray` with floats
            Probability for each value in `ks`.
        """
        self._check_is_fit()
        ks = self._check_ks(ks=ks)
        prob_non_zero = self.non_zeros / self.n
        poiss_pmf = np.asarray(poisson.pmf(ks, mu=self.mu))
        prob_pmf = prob_non_zero * poiss_pmf
        prob_pmf[ks == 0] += 1.0 - prob_non_zero
        return prob_pmf

    def cdf(self, ks: Union[List[int], np.ndarray, range, int]) -> np.ndarray:
        """
        Cumulative Distribution Function.

        Get zero-inflated probability of <= k for one or more values.

        Use `1 - cdf` to get the tail probability p(X > k) for outlier detection.

        Parameters
        ----------
        ks : int, range or list of ints
            Must be non-negative.

        Returns
        -------
        `numpy.ndarray` with floats
            CDF probability for each value in `ks`.
        """
        self._check_is_fit()
        ks = self._check_ks(ks)
        prob_non_zero = self.non_zeros / self.n
        pois_cdf = np.asarray(poisson.cdf(ks, mu=self.mu))
        prob_cdf = (1 - prob_non_zero) + pois_cdf * prob_non_zero
        return prob_cdf
