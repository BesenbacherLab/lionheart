from typing import Union, Optional
import numpy as np


class BinaryPreparer:
    """
    Methods for preparing inputs to binary evaluation.
    """

    @staticmethod
    def prepare_targets(
        targets: Optional[Union[list, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """
        Ensure targets have the right format for binary classification evaluation.

        Parameters
        ----------
        targets : list or `numpy.ndarray` or `None`
            The binary target classes.

        Returns
        -------
        np.ndarray (or None)
            Targets in the right format.
        """
        if targets is not None:
            targets = np.asarray(targets, dtype=np.int32)
            assert targets.ndim <= 2
            if targets.ndim == 2:
                if targets.shape[1] > 1:
                    raise ValueError(
                        (
                            "`targets` must be array with 1 scalar "
                            f"per observation but had shape ({targets.shape})."
                        )
                    )

                # Remove singleton dimension
                targets = targets.squeeze()

        return targets

    @staticmethod
    def prepare_probabilities(
        probabilities: Optional[Union[list, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """
        Ensure probabilities have the right format for binary classification evaluation.

        Parameters
        ----------
        probabilities : list or `numpy.ndarray` or `None`
            The predicted probabilities.

        Returns
        -------
        np.ndarray (or None)
            Predicted probabilities in the right format.
        """
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float32)

            assert probabilities.ndim <= 2
            if probabilities.ndim == 2:
                if probabilities.shape[1] not in [1, 2]:
                    raise ValueError(
                        (
                            "Second dimension of `probabilities` must have size "
                            f"(1) or (2), but had size ({probabilities.shape[1]})."
                        )
                    )

                if probabilities.shape[1] == 2:
                    # Get probabilities of second class
                    probabilities = probabilities[:, 1]
                else:
                    # Remove singleton dimensions
                    probabilities = probabilities.squeeze()

        return probabilities

    @staticmethod
    def prepare_predictions(
        predictions: Optional[Union[list, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """
        Ensure predictions have the right format for binary classification evaluation.

        Parameters
        ----------
        predictions : list or `numpy.ndarray` or `None`
            The predicted classes.

        Returns
        -------
        np.ndarray (or None)
            Predicted classes in the right format.
        """
        if predictions is not None:
            predictions = np.asarray(predictions, dtype=np.int32)

            assert predictions.ndim <= 2
            if predictions.ndim == 2:
                if predictions.shape[1] > 1:
                    raise ValueError(
                        (
                            "`predictions` must be array with 1 scalar "
                            f"per observation but had shape ({predictions.shape})."
                        )
                    )

                # Remove singleton dimension
                predictions = predictions.squeeze()

        return predictions
