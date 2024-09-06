import pathlib
from typing import Callable, List, Tuple, Union
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.pipeline import Pipeline
from utipy import move_column_inplace


class FeatureContributionAnalyzer:
    def __init__(
        self,
        X: np.ndarray,
        pipeline: Pipeline,
        feature_names: List[str],
        groups: List[str],
        get_coefs_fn: Callable = lambda pipe: pipe.named_steps["model"].coef_,
        get_components_fn: Callable = lambda pipe: pipe.named_steps["pca"].components_,
    ) -> None:
        """
        Class for calculating and plotting feature contributions and the
        effects of changing each feature as input to a fitted pipeline on
        the predicted probability.
        """
        self.X = X.copy()
        self.pipeline = deepcopy(pipeline)
        self.feature_names = feature_names
        self.groups = groups
        self.get_coefs_fn = get_coefs_fn
        self.get_components_fn = get_components_fn
        self.feature_contributions = None
        self.feature_effects = None

        # Call the calculators
        self()

    def __call__(self):
        """
        Run the calculations of feature contributions and effects on the probability.
        """
        self.feature_contributions = (
            FeatureContributionAnalyzer._calculate_feature_contributions(
                lasso_coefficients=self.get_coefs_fn(self.pipeline),
                pca_components=self.get_components_fn(self.pipeline),
                feature_names=self.feature_names,
                groups=self.groups,
            )
        )
        self.feature_effects = FeatureContributionAnalyzer._analyze_feature_effects(
            pipeline=self.pipeline,
            X=self.X,
            feature_names=self.feature_names,
            groups=self.groups,
        )
        return self

    def save_effects(self, path: Union[pathlib.Path, str]):
        """
        Save the feature effects as a .csv file.

        Parameters
        ------------
        path : str, optional
          The path to save the data frame at. Should have the ".csv" extension.

        Returns
        -------
        self
        """
        self.feature_effects.to_csv(path, index=False)
        return self

    def save_contributions(self, path: Union[pathlib.Path, str]):
        """
        Save the feature contributions as a .csv file.

        Parameters
        ------------
        path : str, optional
          The path to save the data frame at. Should have the ".csv" extension.

        Returns
        -------
        self
        """
        self.feature_contributions.to_csv(path, index=False)
        return self

    def plot_effects(
        self,
        save_path: str = None,
        fig_size: tuple = (10, 8),
        dpi: int = 300,
    ) -> Figure:
        """
        Plot the feature effects as a heatmap, ordered by the predefined groups and
        clustered within groups. Optionally save the plot.

        Parameters
        ------------
        save_path : str, optional
          The path where the figure will be saved. If None, the figure will not be saved.
        fig_size : tuple, optional
          The size of the figure (default is (10, 8)).
        dpi : int, optional
          The resolution of the figure in dots per inch (default is 300).

        Returns
        --------
        matplotlib.figure.Figure
          The figure object, allowing for further modification.
        """
        return FeatureContributionAnalyzer._plot_feature_effects(
            feature_effects=self.feature_effects,
            save_path=save_path,
            fig_size=fig_size,
            dpi=dpi,
        )

    def plot_contributions(
        self,
        save_path: str = None,
        fig_size: tuple = (10, 8),
        dpi: int = 300,
    ) -> Figure:
        """
        Plot the feature contributions as a horizontal bar plot,
        ordered by the magnitude of contributions.
        Optionally save the plot.

        Parameters
        ------------
        save_path : str, optional
          The path where the figure will be saved. If None, the figure will not be saved.
        fig_size : tuple, optional
          The size of the figure (default is (10, 8)).
        dpi : int, optional
          The resolution of the figure in dots per inch (default is 300).

        Returns
        --------
        matplotlib.figure.Figure
          The figure object, allowing for further modification.
        """
        return FeatureContributionAnalyzer._plot_feature_contributions(
            contributions=self.feature_contributions,
            save_path=save_path,
            fig_size=fig_size,
            dpi=dpi,
        )

    @staticmethod
    def _calculate_feature_contributions(
        lasso_coefficients: np.ndarray,
        pca_components: np.ndarray,
        feature_names: Union[List[str], np.ndarray],
        groups: Union[List[str], np.ndarray],
    ) -> pd.DataFrame:
        feature_contribution_df = pd.DataFrame(
            {
                "Feature Idx": range(pca_components.shape[1]),
                "Feature": feature_names,
                "Group": groups,
                "Contribution": np.dot(lasso_coefficients, pca_components).flatten(),
            }
        )
        # Sort by absolute contribution
        feature_contribution_df["AbsContribution"] = feature_contribution_df[
            "Contribution"
        ].abs()
        feature_contribution_df = feature_contribution_df.sort_values(
            by="AbsContribution", ascending=False
        ).reset_index(drop=True)
        return feature_contribution_df

    @staticmethod
    def _plot_feature_contributions(
        contributions: pd.DataFrame,
        save_path: str = None,
        fig_size: tuple = (10, 8),
        dpi: int = 300,
    ) -> Figure:
        """
        Plot the feature contributions as a horizontal bar plot,
        ordered by the magnitude of contributions.
        Optionally save the plot.

        Parameters
        ------------
        contributions : pandas.DataFrame
            The data frame with the contributions per feature.
        save_path : str, optional
          The path where the figure will be saved. If None, the figure will not be saved.
        fig_size : tuple, optional
          The size of the figure (default is (10, 8)).
        dpi : int, optional
          The resolution of the figure in dots per inch (default is 300).

        Returns
        --------
        matplotlib.figure.Figure
          The figure object, allowing for further modification.
        """
        # Step 7: Plot the feature contributions
        plt.figure(figsize=fig_size)
        plt.barh(
            contributions["Feature"],
            contributions["Contribution"],
            color="skyblue",
        )
        plt.xlabel("Contribution")
        plt.ylabel("Feature")
        plt.title("Feature Contributions to the Classifier")
        plt.gca().invert_yaxis()

        # Save the plot if a path is given
        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
            )

        # Return the figure object to allow further changes
        return plt.gcf()

    @staticmethod
    def _analyze_feature_effects(
        pipeline: Pipeline,
        X: np.ndarray,
        feature_names: list[str],
        groups: list[str],
        step: float = 0.05,
        class_index: int = 1,
    ) -> pd.DataFrame:
        """
        Analyze how changes in each feature affect the predicted probability (predict_proba)
        and return a DataFrame with the results, including grouping information.

        Note that when the pipeline has row-scaling (row / mean(row)) changing a
        single feature affects the mean and thus all the other features as well
        (although this effect should be negible with a large feature set).

        Parameters
        ------------
        pipeline
            The scikit-learn pipeline containing the model.
        X
            The test data (features).
        feature_names
            List of feature names.
        groups
            List or array mapping each feature to a group.
        step
            The step size to vary each feature.
        class_index
            Index of the class to examine in the predict_proba output (1 for positive class in binary classification).

        Returns
        --------
        pandas.DataFrame
            DataFrame containing feature effects, feature names, and group labels.
        """
        n_samples, n_features = X.shape
        original_probas = pipeline.predict_proba(X)[
            :, class_index
        ]  # Base probabilities for the positive class
        feature_effects = []

        # Vary each feature and calculate the effect on predict_proba
        # Variation range is based on the input features
        # being correlations and experimentation with ranges
        variations = np.arange(-0.5, 0.5, step)

        for feature_idx in range(n_features):
            prob_diffs = []
            for var in variations:
                X_mod = X.copy()
                X_mod[:, feature_idx] += var
                new_probas = pipeline.predict_proba(X_mod)[:, class_index]
                prob_diff = (
                    new_probas - original_probas
                )  # Calculate the difference in predicted probabilities
                prob_diffs.append(
                    np.mean(prob_diff)
                )  # Average effect across all test samples

            feature_effects.append(prob_diffs)

        # Convert the results into a DataFrame for plotting
        feature_effects_df = pd.DataFrame(
            feature_effects, columns=[f"{round(var, 2)}" for var in variations]
        )

        # Add feature and feature group names
        feature_effects_df["Feature"] = feature_names
        feature_effects_df["Group"] = groups  # Use the provided group labels

        # Reorder to have name columns first
        move_column_inplace(feature_effects_df, "Group", 0)
        move_column_inplace(feature_effects_df, "Feature", 0)

        return feature_effects_df

    @staticmethod
    def _cluster_feature_effects(
        feature_effects: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], List[tuple[int, int]]]:
        """
        Cluster the feature effects within pre-defined groups.

        Parameters
        ------------
        feature_effects
            The DataFrame returned by analyze_feature_effects_all containing feature effects, groups, and feature names.
        """
        # Create a copy to avoid modifying the original DataFrame
        feature_effects = feature_effects.copy()

        # Get the list of unique groups
        # Sorted by largest -> smallest
        groups, group_counts = np.unique(
            feature_effects["Group"].to_numpy(),
            return_counts=True,
        )
        groups = groups[np.flip(np.argsort(group_counts))].tolist()

        # Prepare a DataFrame to store the reordered features
        ordered_effects_dfs = []

        # Loop over each group and cluster features within each group
        for group in groups:
            # Subset the features of the current group
            group_data = feature_effects[feature_effects["Group"] == group].copy()
            if len(group_data) < 2:
                ordered_effects_dfs.append(group_data)
                continue

            # Perform clustering within the group
            effect_data = group_data.drop(columns=["Feature", "Group"])
            clustering = linkage(effect_data, method="ward")
            order = leaves_list(clustering)

            # Reorder the group features based on clustering
            # Append the ordered group data to the result DataFrame
            ordered_effects_dfs.append(group_data.iloc[order])

        ordered_effects = pd.concat(ordered_effects_dfs)

        # Get group ID per feature
        group_ids = ordered_effects["Group"].to_numpy()

        # Use np.where to find where the group changes
        group_changes = (
            np.where(group_ids[:-1] != group_ids[1:])[0] + 1
        )  # Add 1 to shift to the right index

        # Group positions will be pairs of (start, end) indices
        group_positions = [(0, group_changes[0])]
        group_positions += [
            (group_changes[i], group_changes[i + 1])
            for i in range(len(group_changes) - 1)
        ]
        group_positions.append(
            (group_changes[-1], len(group_ids))
        )  # Add the last group

        return ordered_effects, groups, group_positions

    @staticmethod
    def _plot_feature_effects(
        feature_effects: pd.DataFrame,
        save_path: str = None,
        fig_size: tuple = (12, 8),
        dpi: int = 300,
    ) -> Figure:
        """
        Plot the feature effects as a heatmap, grouped by the predefined groups and
        clustered within groups.

        Parameters
        ------------
        feature_effects
          The DataFrame returned by _analyze_feature_effects containing feature effects,
          groups, and feature names.
        save_path : str, optional
          The path where the figure will be saved. If None, the figure will not be saved.
        fig_size : tuple, optional
          The size of the figure (default is (10, 8)).
        dpi : int, optional
          The resolution of the figure in dots per inch (default is 300).

        Returns
        --------
        matplotlib.figure.Figure
          The figure object, allowing for further modification.
        """

        ordered_effects, groups, group_positions = (
            FeatureContributionAnalyzer._cluster_feature_effects(feature_effects)
        )

        # Remove the Feature and Group columns for heatmap plotting
        del ordered_effects["Feature"]
        del ordered_effects["Group"]

        # Plot as a heatmap
        plt.figure(figsize=fig_size)
        sns.heatmap(
            ordered_effects, cmap="coolwarm", annot=False, center=0, yticklabels=False
        )

        # Add lines between groups and a single group label per group
        for start, end in group_positions:
            plt.hlines([end], *plt.xlim(), colors="black", lw=1)

        # Add group names in the middle of each group's section
        for i, (start, end) in enumerate(group_positions):
            if i > 15:
                # Don't name the smallest groups
                continue
            plt.text(
                -1.2,
                (start + end) / 2,
                groups[i],
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
            )

        plt.title("Effect of Feature Changes on Predicted Probability")
        plt.xlabel("Change in Feature Value")
        plt.ylabel("Feature")

        # Save the plot if a path is given
        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
            )

        # Return the figure object to allow further changes
        return plt.gcf()
