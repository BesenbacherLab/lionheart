from numbers import Number
from typing import List, Optional, Tuple, Union, Dict
import warnings
import numpy as np
from scipy.stats import poisson


class PoissonPMF:
    def __init__(
        self, handle_negatives: str = "raise", max_num_negatives: Optional[int] = None
    ) -> None:
        """
        Poisson probability mass function.

        Wrapper for `scipy.stats.poisson`. See its `.pmf()` method.

        See `ZIPoissonPMF` for handling zero-inflated data.

        Parameters
        ----------
        handle_negatives : str
            How to handle negative numbers (e.g. numeric versions of NaN).
            One of: {"raise", "warn_truncate", "truncate"}.
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
        Create new `PoissonPMF` model with existing parameters.

        Parameters
        ----------
        n
            Number of data points (positive integers).
        mu
            Mean of the data points.
        handle_negatives, max_num_negatives
            See `PoissonPMF.__init__()`.

        Returns
        -------
        `PoissonPMF`
            Poisson with the specified values.
        """
        m = PoissonPMF(
            handle_negatives=handle_negatives, max_num_negatives=max_num_negatives
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
        if not isinstance(pos, int) and pos >= 0:
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
        if self.n == 0:
            raise RuntimeError(f"{self.__class__.__name__}: `.fit()` not called.")
        self._iter_n = 0
        return self

    def __next__(self) -> Tuple[int, float]:
        """
        Get next integer and its probability.

        Returns
        -------
        tuple
            int
                k
            float
                probability of k
        """
        if self.n == 0:
            raise RuntimeError(f"{self.__class__.__name__}: `.fit()` not called.")
        self._iter_n += 1
        return self._iter_n - 1, self.pmf(self._iter_n - 1)

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
            elif self.handle_negatives == "warn_truncate":
                warnings.warn(self._str_negative_numbers(x=x))
                x[x < 0] = 0
            elif self.handle_negatives == "truncate":
                x[x < 0] = 0

        self._update_mean(x=x, old_mu=self.mu, old_n=self.n)
        self.n += len(x)
        return self

    def _str_negative_numbers(self, x):
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

    def pmf(self, ks: Union[List[int], np.ndarray, range, int]) -> List[float]:
        """
        Probability Mass Function.

        Get probability of one or more values.

        Parameters
        ----------
        ks : int, range or list of ints

        Returns
        -------
        `numpy.ndarray` with floats
            Probability for each value in `ks`.
        """
        ks = self._check_ks(ks=ks)
        return np.asarray(poisson.pmf(ks, mu=self.mu))

    def _check_ks(self, ks):
        """
        Check the `ks` argument.
        """
        if isinstance(ks, range):
            ks = list(ks)
        if isinstance(ks, Number):
            ks = [ks]
        if isinstance(ks, np.ndarray):
            assert ks.ndim == 1
        return ks


class ZIPoissonPMF(PoissonPMF):
    def __init__(
        self, handle_negatives="raise", max_num_negatives: Optional[int] = None
    ) -> None:
        """
        Zero-inflated Poisson probability mass function.

        p(X == k | k != 0) = p(X != 0) * pmf(k, mean(X))

        p(X == k | k == 0) = p(X == 0) + p(X != 0) * pmf(k, mean(X))

        Parameters
        ----------
        handle_negatives : str
            How to handle negative numbers (e.g. numeric versions of NaN).
            One of: {"raise", "warn_truncate", "truncate"}.
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
        Create new `ZIPoissonPMF` model with existing parameters.

        Parameters
        ----------
        n
            Number of data points (positive integers).
        mu
            Mean of the data points.
        n_non_zero
            Number on non-zero data points.
        handle_negatives, max_num_negatives
            See `ZIPoissonPMF.__init__()`.

        Returns
        -------
        `ZIPoissonPMF`
            Zero-inflated Poisson with the specified values.
        """
        m = ZIPoissonPMF(
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

    def pmf(self, ks: Union[List[int], np.ndarray, range, int]) -> List[float]:
        """
        Probability Mass Function.

        Get probability of one or more values.

        Parameters
        ----------
        ks : int, range or list of ints

        Returns
        -------
        `numpy.ndarray` with floats
            Probability for each value in `ks`.
        """
        ks = self._check_ks(ks=ks)
        prob_non_zero = self.non_zeros / self.n
        poiss_probs = poisson.pmf(ks, mu=self.mu)
        return np.asarray(
            [
                int(k == 0) * (1 - prob_non_zero) + prob_non_zero * p
                for p, k in zip(poiss_probs, ks)
            ]
        )
