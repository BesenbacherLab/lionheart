import argparse

import pytest

from lionheart.commands import cross_validate, train_model
from lionheart.modeling.prepare_modeling_command import get_category_indices


def test_feature_categories_parser_accepts_include_and_exclude_modes():
    for setup_parser in [
        lambda parser: cross_validate.setup_parser(parser, show_advanced=False),
        train_model.setup_parser,
    ]:
        parser = argparse.ArgumentParser()
        setup_parser(parser)

        args = parser.parse_args(
            [
                "--out_dir",
                "out",
                "--resources_dir",
                "resources",
                "--feature_categories",
                "Blood/Immune",
                "Digestive System",
            ]
        )
        assert args.feature_categories == ["Blood/Immune", "Digestive System"]

        args = parser.parse_args(
            [
                "--out_dir",
                "out",
                "--resources_dir",
                "resources",
                "--feature_categories",
                "exclude",
                "Blood/Immune",
                "Digestive System",
            ]
        )
        assert args.feature_categories == [
            "exclude",
            "Blood/Immune",
            "Digestive System",
        ]


def test_get_category_indices_include_and_exclude(tmp_path):
    category_path = tmp_path / "feature_names_and_grouping.csv"
    category_path.write_text(
        "idx\tcell_type\tcategory\n"
        "0\tNeutrophils\tBlood/Immune\n"
        "1\tColon\tDigestive System\n"
        "2\tLung\tRespiratory System\n"
    )

    args = argparse.Namespace(feature_categories=["Blood/Immune", "Digestive System"])
    assert get_category_indices(args, category_path) == [0, 1]

    args = argparse.Namespace(
        feature_categories=["exclude", "Blood/Immune", "Digestive System"]
    )
    assert get_category_indices(args, category_path) == [2]


def test_get_category_indices_rejects_misplaced_exclude(tmp_path):
    category_path = tmp_path / "feature_names_and_grouping.csv"
    category_path.write_text(
        "idx\tcell_type\tcategory\n"
        "0\tNeutrophils\tBlood/Immune\n"
        "1\tColon\tDigestive System\n"
    )

    args = argparse.Namespace(feature_categories=["Blood/Immune", "exclude"])
    with pytest.raises(ValueError, match="must be the first value"):
        get_category_indices(args, category_path)


def test_get_category_indices_rejects_exclude_without_categories(tmp_path):
    category_path = tmp_path / "feature_names_and_grouping.csv"
    category_path.write_text(
        "idx\tcell_type\tcategory\n"
        "0\tNeutrophils\tBlood/Immune\n"
    )

    args = argparse.Namespace(feature_categories=["exclude"])
    with pytest.raises(ValueError, match="must be followed by at least one category"):
        get_category_indices(args, category_path)


def test_get_category_indices_rejects_old_dash_prefix(tmp_path):
    category_path = tmp_path / "feature_names_and_grouping.csv"
    category_path.write_text(
        "idx\tcell_type\tcategory\n"
        "0\tNeutrophils\tBlood/Immune\n"
    )

    args = argparse.Namespace(feature_categories=["-Blood/Immune"])
    with pytest.raises(ValueError, match="no longer accepts '-' prefixes"):
        get_category_indices(args, category_path)


def test_get_category_indices_rejects_excluding_all_categories(tmp_path):
    category_path = tmp_path / "feature_names_and_grouping.csv"
    category_path.write_text(
        "idx\tcell_type\tcategory\n"
        "0\tNeutrophils\tBlood/Immune\n"
        "1\tColon\tDigestive System\n"
    )

    args = argparse.Namespace(
        feature_categories=["exclude", "Blood/Immune", "Digestive System"]
    )
    with pytest.raises(ValueError, match="excluded all feature categories"):
        get_category_indices(args, category_path)
