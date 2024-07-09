
from numbers import Number
from typing import Dict, Tuple
import numpy as np
from scipy import special


class RunningPearsonR():

    def __init__(self, ignore_nans: bool = True) -> None:
        """
        Calculates Pearson R from multiple subsets of data.
        Add data multiple times and get the current pearson R and
        p-value from the `.pearson_r` property.

        Also calculates the cosine similarity.

        NOTE: No effort has been made with regards to 
        numerical stability.

        Based on:
            1) The formula at https://mathoverflow.net/a/57914
            2) `pearsonr()` from `scipy.stats`
                See this for details on the p-value calculation.

        Parameters
        ----------
        ignore_nans : bool
            Whether to remove elements that are NaN in either array 
            prior to calculation of sums.
        """
        self.ignore_nans = ignore_nans
        self.dtype = None
        self.n = 0
        self.x_sum = 0.0
        self.x_squared_sum = 0.0
        self.y_sum = 0.0
        self.y_squared_sum = 0.0
        self.xy_sum = 0.0

    @property
    def state(self) -> Dict[str, Number]:
        """
        Get current state as a dict.
        """
        return {
            "n": self.n,
            "x_sum": self.x_sum,
            "x_squared_sum": self.x_squared_sum,
            "y_sum": self.y_sum,
            "y_squared_sum": self.y_squared_sum,
            "xy_sum": self.xy_sum
        }

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Add data. The length and a set of sums
        are added to the existing `n` and sums.
        """
        # `._check_data` removes NaNs when specified
        x, y, n = self._check_data(x, y)
        if n == 0:
            return
        self.n += n
        self.x_sum += x.sum(dtype=self.dtype)
        self.y_sum += y.sum(dtype=self.dtype)
        self.x_squared_sum += (x**2).sum(dtype=self.dtype)
        self.y_squared_sum += (y**2).sum(dtype=self.dtype)
        self.xy_sum += (x * y).sum(dtype=self.dtype)

    @property
    def pearson_r(self) -> Tuple[float, float]:
        """
        Calculates the Pearson R with the current data statistics.

        NOTE: Recalculated on every call, so consider saving 
        output to variable.

        Returns
        -------
        float
            Pearson R
        float
            P-value
        """
        if self.n < 2:
            raise ValueError(
                'At least two elements are required to calculate pearson_r. '
                'Please add more data.'
            )

        numerator = (self.n * self.xy_sum - self.x_sum * self.y_sum)
        denominator = np.sqrt(self.n * self.x_squared_sum - self.x_sum**2) * \
            np.sqrt(self.n * self.y_squared_sum - self.y_sum**2)
        r = numerator / denominator

        # Presumably, if abs(r) > 1, then it is only some small artifact of
        # floating point arithmetic.
        r = max(min(r, 1.0), -1.0)

        # FROM scipy.stats.pearsonr docs:
        #   As explained in the docstring, the p-value can be computed as
        #       p = 2*dist.cdf(-abs(r))
        #   where dist is the beta distribution on [-1, 1] with shape parameters
        #   a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
        #   on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
        #   shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
        #   becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
        #   to avoid a TypeError raised by btdtr when r is higher precision.)
        ab = self.n / 2 - 1
        prob = 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))

        return float(r), float(prob)

    @property
    def cosine_similarity(self) -> float:
        """
        Calculates the cosine similarity.
        `cosine similarity = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}`
        """
        if self.n < 2:
            raise ValueError(
                'At least two elements are required to calculate cosine_similarity. '
                'Please add more data.'
            )

        denominator = np.sqrt(self.x_squared_sum) * np.sqrt(self.y_squared_sum)
        cos_sim = self.xy_sum / denominator

        # Presumably, if abs(cos_sim) > 1, then it is only some
        # small artifact of floating point arithmetic
        cos_sim = max(min(cos_sim, 1.0), -1.0)

        return float(cos_sim)

    def _check_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:

        # Ensure x and y are ndarrays
        x = np.asarray(x)
        y = np.asarray(y)

        # Remove elements that are NaN in either array
        if self.ignore_nans:
            not_nan_mask = (np.isnan(x).astype(np.int32) +
                            np.isnan(y).astype(np.int32)) == 0
            x = x[not_nan_mask]
            y = y[not_nan_mask]

        n = len(x)
        if n != len(y):
            raise ValueError('x and y must have the same length.')

        # dtype is the data type for the calculations.  This expression ensures
        # that the data type is at least 64 bit floating point.  It might have
        # more precision if the input is, for example, np.longdouble.
        dtype = type(1.0 + x[0] + y[0])
        if self.dtype is None:
            self.dtype = dtype
        elif self.dtype != dtype:
            # TODO: Probably is if the `type(...)` call doesn't fail
            raise TypeError(
                "The data type of `x` and/or `y` was not compatible "
                "with previously recorded data."
            )

        return x, y, n
