import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generalize.model.transformers import DimTransformerWrapper, IndexFeatureSelector
from lionheart.modeling.feature_contribution import FeatureContributionAnalyzer
from lionheart.modeling.transformers import prepare_transformers_fn


def test_feature_contributions_use_post_scaling_selected_feature_metadata():
    X = np.array(
        [
            [0.0, 2.0, 1.0, 4.0, 0.5],
            [1.0, 1.0, 2.0, 3.0, 1.5],
            [2.0, 0.0, 3.0, 2.0, 2.5],
            [3.0, 1.0, 4.0, 1.0, 3.5],
            [4.0, 2.0, 5.0, 0.0, 4.5],
            [5.0, 3.0, 6.0, 1.0, 5.5],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    selected_feature_indices = [0, 2, 4]
    feature_names = pd.Series(
        ["Neutrophils", "Colon", "Lung", "Liver", "Monocytes"]
    )
    feature_groups = pd.Series(
        [
            "Blood/Immune",
            "Digestive System",
            "Respiratory System",
            "Digestive System",
            "Blood/Immune",
        ]
    )

    pipeline = Pipeline(
        [
            ("row_standardize", StandardScaler()),
            (
                "select_features_post_scaling",
                DimTransformerWrapper(
                    IndexFeatureSelector,
                    kwargs={"feature_indices": selected_feature_indices},
                ),
            ),
            ("pre_pca_standardize", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=1)),
            ("standardize", StandardScaler()),
            ("model", LogisticRegression(solver="liblinear", random_state=1)),
        ]
    ).fit(X, y)

    analyser = FeatureContributionAnalyzer(
        X=X,
        pipeline=pipeline,
        feature_names=feature_names,
        groups=feature_groups,
    )

    assert set(analyser.feature_contributions["Feature"]) == {
        "Neutrophils",
        "Lung",
        "Monocytes",
    }
    assert set(analyser.feature_contributions["Group"]) == {
        "Blood/Immune",
        "Respiratory System",
    }
    assert len(analyser.feature_contributions) == len(selected_feature_indices)
    assert len(analyser.feature_effects) == len(selected_feature_indices)


def test_post_scaling_feature_selection_rejects_preceding_variance_filtering():
    transformers_fn = prepare_transformers_fn(
        pca_target_variance=[0.99],
        min_var_thresh=[0.0],
        post_scale_feature_indices=[0, 2, 4],
    )

    with pytest.raises(ValueError, match="post_scale_feature_indices"):
        transformers_fn({"grid": {}})
