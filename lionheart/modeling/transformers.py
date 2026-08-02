from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from generalize.model.transformers import PCAByExplainedVariance
from generalize.model.pipeline.pipeline_designer import PipelineDesigner
from sklearn.impute import SimpleImputer

def default_min_var_thresh():
    return [0.0]


def default_scale_rows():
    return ["mean", "std"]


def prepare_transformers_fn(
    pca_target_variance: List[float],
    min_var_thresh: Optional[List[float]] = None,
    scale_rows: Optional[List[str]] = None,
    post_scale_feature_indices: Optional[List[int]] = None,
    standardize: bool = True,
    add_dim_transformer_wrappers: bool = False,
):
    # Set defaults for mutable types
    if min_var_thresh is None:
        min_var_thresh = default_min_var_thresh()
    if scale_rows is None:
        scale_rows = default_scale_rows()

    def transformers_fn(
        model_dict: dict,
    ) -> Tuple[List[Tuple[str, BaseEstimator]], dict]:
        designer = PipelineDesigner()

        split_designer = True
        if split_designer:

            feature_splits = split_indices={
                "CT_Pearson": list(range(0,898)),
                "CT_LengthRatios": list(range(898,1796)),
                "CNA_Pearson": list(range(1796,1833)),
                "CNA_LengthContrasts": list(range(1833,1944)),
                "METH_CGN_NCG": list(range(1944,2409)),
            }

            feature_pca_vars = split_indices={
                "CT_Pearson": 0.993,
                "CT_LengthRatios": 0.997,
                "CNA_Pearson": 0.993,
                "CNA_LengthContrasts": 0.993,
                "METH_CGN_NCG": 0.8,
            }

            # hack since split() checks for NaN
            designer.add_step(
                name="replace_nan_with_sentinel",
                transformer=SimpleImputer,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                kwargs={
                    "strategy": "constant",
                    "fill_value": -100.0,
                },
            )

            designer.split(
                name="split_feature_sets",
                split_indices=feature_splits,
            )

            for feature_group_name in feature_splits.keys():

                kwargs = {}
                kwargs["threshold"] = 0.0
                designer.add_step(
                    name=f"{feature_group_name}__zero_variance",
                    transformer=VarianceThreshold,
                    add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                    kwargs=kwargs,
                )
                del kwargs

                """ Handled by impute -> variance selection
                designer.add_step(
                    name=f"{feature_group_name}__rm_all_nan_columns",
                    transformer=AllNaNFeatureRemover,
                    add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                    kwargs={},
                )
                """

                if "Pearson" in feature_group_name and scale_rows:
                    if len(scale_rows) != 2:
                        raise ValueError(
                            "When `--scale_rows` is specified, it must have length 2."
                        )
                    kwargs = {}
                    kwargs["center"], kwargs["scale"] = [
                        metric if metric != "none" else None for metric in scale_rows
                    ]
                    designer.add_step(
                        name=f"{feature_group_name}__row_standardize",
                        transformer="scale_rows",
                        add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                        kwargs=kwargs,
                    )
                    del kwargs

                if False and feature_group_name == "CNA_LengthContrasts":
                    designer.add_step(
                        name=f"{feature_group_name}__rm_all_nan_columns",
                        transformer=AllNaNFeatureRemover,
                        add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                        kwargs={},
                    )

                kwargs = {}
                kwargs["target_variance"] =  feature_pca_vars[feature_group_name]
                designer.add_step(
                    name=f"{feature_group_name}__pre_pca_standardize",
                    transformer=StandardScaler,
                    add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                )
                designer.add_step(
                    name=f"{feature_group_name}__pca",
                    transformer=PCAByExplainedVariance,
                    add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                    kwargs=kwargs,
                )
                del kwargs



        """
        designer.add_step(
            name="rm_all_nan_columns",
            transformer=AllNaNFeatureRemover,
            add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            kwargs={},
        )

        # Add minimum variance feature selector
        if min_var_thresh:
            if len(min_var_thresh) != 1:
                if add_dim_transformer_wrappers:
                    model_dict["grid"]["near_zero_variance__kwargs"] = [
                        {"threshold": vt} for vt in min_var_thresh
                    ]
                else:
                    model_dict["grid"]["near_zero_variance__threshold"] = min_var_thresh
            kwargs = {}
            if len(min_var_thresh) == 1:
                kwargs["threshold"] = min_var_thresh[0]
            designer.add_step(
                name="near_zero_variance",
                transformer=VarianceThreshold,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                kwargs=kwargs,
            )
            del kwargs

        # Add row-scaling
        # NOTE: Must come after the variance-based selection steps
        if scale_rows:
            if len(scale_rows) != 2:
                raise ValueError(
                    "When `--scale_rows` is specified, it must have length 2."
                )
            kwargs = {}
            kwargs["center"], kwargs["scale"] = [
                metric if metric != "none" else None for metric in scale_rows
            ]
            designer.add_step(
                name="row_standardize",
                transformer="scale_rows",
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                kwargs=kwargs,
            )
            del kwargs

        # NOTE: Cannot have selection based on min_var_thresh or max_corr_thresh
        # prior to this
        if post_scale_feature_indices is not None:
            if min_var_thresh:
                raise ValueError(
                    "Cannot perform `post_scale_feature_indices` selection when "
                    "`--min_var_thresh` is specified."
                )
            designer.add_step(
                name="select_features_post_scaling",
                transformer="feature_selector",
                add_dim_transformer_wrapper=True,
                kwargs={"feature_indices": post_scale_feature_indices},
            )

        if pca_target_variance:
            if len(pca_target_variance) != 1:
                if add_dim_transformer_wrappers:
                    model_dict["grid"]["pca__kwargs"] = [
                        {"target_variance": tv} for tv in pca_target_variance
                    ]
                else:
                    model_dict["grid"]["pca__target_variance"] = pca_target_variance
            kwargs = {}
            if len(pca_target_variance) == 1:
                kwargs["target_variance"] = pca_target_variance[0]
            designer.add_step(
                name="pre_pca_standardize",
                transformer=StandardScaler,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            )
            designer.add_step(
                name="pca",
                transformer=PCAByExplainedVariance,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                kwargs=kwargs,
            )
            del kwargs

        """

        # Combine pipelines
        if split_designer:
            designer.collect()

        # Add standard scaler
        if standardize:
            designer.add_step(
                name="standardize",
                transformer=StandardScaler,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            )

        kwargs = {}
        kwargs["target_variance"] = 0.997
        designer.add_step(
            name="pca_combined",
            transformer=PCAByExplainedVariance,
            add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            kwargs=kwargs,
        )
        del kwargs

        # Add standard scaler
        if standardize:
            designer.add_step(
                name="standardize_again",
                transformer=StandardScaler,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            )


        # If no non-split/collect steps were added
        if not designer.n_transformers:
            designer.add_step(
                name="identity",
                transformer="identity",
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            )

        return designer.build(), model_dict

    return transformers_fn


def prepare_benchmark_transformers_fn(
    feature_type: str,
    pca_target_variance: List[float],
    min_var_thresh: Optional[List[float]] = None,
    standardize: bool = True,
    add_dim_transformer_wrappers: bool = False,
):
    # Set defaults for mutable types
    if min_var_thresh is None:
        min_var_thresh = default_min_var_thresh()

    def transformers_fn(
        model_dict: dict,
    ) -> Tuple[List[Tuple[str, BaseEstimator]], dict]:
        designer = PipelineDesigner()

        designer.add_step(
            name="rm_all_nan_columns",
            transformer=AllNaNFeatureRemover,
            add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            kwargs={},
        )

        if feature_type in ["bin_depths"]:
            # Divide by row-mean (1-centered but keep proportional variance)
            designer.add_step(
                name="row_scale",
                transformer="scale_rows",
                add_dim_transformer_wrapper=True,
                kwargs={
                    "center": None,
                    "scale": "mean",
                },
            )

        if feature_type in ["lengths"]:
            # Divide by row-sum (--> distribution)
            designer.add_step(
                name="row_scale",
                transformer="scale_rows",
                add_dim_transformer_wrapper=True,
                kwargs={
                    "center": None,
                    "scale": "sum",
                },
            )

        # Add minimum variance feature selector
        if min_var_thresh:
            if len(min_var_thresh) != 1:
                if add_dim_transformer_wrappers:
                    model_dict["grid"]["near_zero_variance__kwargs"] = [
                        {"threshold": vt} for vt in min_var_thresh
                    ]
                else:
                    model_dict["grid"]["near_zero_variance__threshold"] = min_var_thresh
            kwargs = {}
            if len(min_var_thresh) == 1:
                kwargs["threshold"] = min_var_thresh[0]
            designer.add_step(
                name="near_zero_variance",
                transformer=VarianceThreshold,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                kwargs=kwargs,
            )
            del kwargs

        if pca_target_variance:
            if len(pca_target_variance) != 1:
                if add_dim_transformer_wrappers:
                    model_dict["grid"]["pca__kwargs"] = [
                        {"target_variance": tv} for tv in pca_target_variance
                    ]
                else:
                    model_dict["grid"]["pca__target_variance"] = pca_target_variance
            kwargs = {}
            if len(pca_target_variance) == 1:
                kwargs["target_variance"] = pca_target_variance[0]
            designer.add_step(
                name="pre_pca_standardize",
                transformer=StandardScaler,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            )
            designer.add_step(
                name="pca",
                transformer=PCAByExplainedVariance,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
                kwargs=kwargs,
            )
            del kwargs

        # Add standard scaler
        if standardize:
            designer.add_step(
                name="standardize",
                transformer=StandardScaler,
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            )

        # If no non-split/collect steps were added
        if not designer.n_transformers:
            designer.add_step(
                name="identity",
                transformer="identity",
                add_dim_transformer_wrapper=add_dim_transformer_wrappers,
            )

        return designer.build(), model_dict

    return transformers_fn


# Tmp NaN removing transformer

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class AllNaNFeatureRemover(BaseEstimator, TransformerMixin):
    """Remove features that contain only NaN values in the training data."""

    def fit(self, X, y=None):
        X = self._validate_data(X, force_all_finite="allow-nan")
        self.keep_mask_ = ~np.isnan(X).all(axis=0)

        if not self.keep_mask_.any():
            raise ValueError("All features contain only NaN values.")

        return self

    def transform(self, X):
        check_is_fitted(self, "keep_mask_")
        X = self._validate_data(
            X,
            reset=False,
            force_all_finite="allow-nan",
        )
        return X[:, self.keep_mask_]

    def get_support(self):
        """Return the Boolean mask of retained features."""
        check_is_fitted(self, "keep_mask_")
        return self.keep_mask_.copy()
