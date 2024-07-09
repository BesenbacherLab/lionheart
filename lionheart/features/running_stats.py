
from numbers import Number
from typing import Dict, Tuple
import numpy as np

# TODO perhaps add() is more standard than add_data() ?


class RunningStats():

    def __init__(self, ignore_nans: bool = True) -> None:
        """
        Running statistics about chunks of arrays.

        Only summary statistics are saved in the object.

        Variance and standard deviation:
            Chunked unbiased (1 delta degrees of freedom) estimator of variance, based on the final formula at:
                https://math.stackexchange.com/a/2971522
            Same as:
                `numpy.var(x, ddof=1)` ; `numpy.std(x, ddof=1)`
                for the complete data.
        """
        self.ignore_nans = ignore_nans
        self.dtype = None
        self.n = 0
        self.mean = 0.0
        self.var = 0.0
        self.min = None
        self.max = None

    @property
    def std(self) -> float:
        return np.sqrt(self.var)

    @property
    def stats(self) -> Dict[str, Number]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max
        }

    def add_data(self, x: np.ndarray) -> None:
        """
        Update statistics with more data. Only summary statistics are saved in the object.

        :param x: 1D `numpy.ndarray` with numbers to add to the running summarization.
        """
        # Check input data
        x = self._check_data(x)
        new_n = len(x)
        if new_n == 0:
            return
        if self.n == 0:
            self.n = new_n
            self.mean, self.var, self.min, self.max = self._compute_stats(x=x)
            return

        new_mean, new_var, new_min, new_max = self._compute_stats(x=x)

        # Variance calculations
        # Chunked unbiased estimator of variance, based on the final formula at:
        #   https://math.stackexchange.com/a/2971522
        # Continued after updating of mean and count
        # (here self.* are the 'old' values)
        q1 = (self.n - 1) * self.var + self.n * self.mean**2
        q2 = (new_n - 1) * new_var + new_n * new_mean**2
        qc = q1 + q2

        # Update mean
        old_mean_expanded = self.n * self.mean
        new_mean_expanded = new_n * new_mean
        self.mean = (old_mean_expanded + new_mean_expanded) / (self.n + new_n)

        # Update count
        self.n += new_n

        # Calculate new variance
        self.var = (qc - (self.n) * self.mean**2) / (self.n - 1)

        # Update min and max
        self.min = min(self.min, new_min)
        self.max = max(self.max, new_max)

    def _compute_stats(self, x: np.ndarray) -> Tuple[float, float, Number, Number]:
        """
        Compute statistics for new data subset.
        """
        return np.mean(x), np.var(x, ddof=1), np.min(x), np.max(x)

    def _check_data(self, x: np.ndarray) -> np.ndarray:

        # Ensure x is an ndarray
        x = np.asarray(x)
        assert x.ndim == 1, "`x` must be an 1-dimensional array."

        # Remove elements that are NaN in either array
        if self.ignore_nans:
            x = x[~np.isnan(x)]

        # dtype is the data type for the calculations.  This expression ensures
        # that the data type is at least 64 bit floating point.  It might have
        # more precision if the input is, for example, np.longdouble.
        dtype = type(1.0 + x[0])
        if self.dtype is None:
            self.dtype = dtype
        elif self.dtype != dtype:
            # TODO: Probably is if the `type(...)` call doesn't fail
            raise TypeError(
                "The data type of `x` and/or `y` was not compatible "
                "with previously recorded data."
            )

        return x
