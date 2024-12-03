import pathlib
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utipy import Messenger, StepTimer, check_messenger, mk_dir


def plot_inner_scores(
    inner_results: pd.DataFrame,
    catplot_kwargs: dict = {"height": 5, "aspect": 1.5, "kind": "strip"},
    metric_name="Balanced Accuracy",
    save_dir: Union[str, pathlib.Path] = None,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """

    catplot_args : dict
        Arguments to `searborn.catplot`.
        See https://seaborn.pydata.org/generated/seaborn.catplot.html
    messenger : `utipy.Messenger` or None
        A `utipy.Messenger` instance used to print/log/... information.
        When `None`, no printing/logging is performed.
        The messenger determines the messaging function (e.g., `print` or `log.info`)
        and potential indentation.
    """

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(verbose=messenger.verbose, msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Detect hyperparameter columns
    hparam_cols = [col for col in inner_results.columns if col.startswith("param_")]

    # Create pretty names for the hyperparameters
    col_to_name = _extract_hparam_names(hparam_cols)
    messenger(f"Detected the following hyperparameters: {col_to_name}.")

    # Aggregate by the different hyperparameters' settings
    messenger("Aggregating inner results by each hyperparameter's settings")
    aggregated_dfs = {
        col: _aggregate_folds_by_hparam(
            inner_results, col=col, col_to_name=col_to_name, messenger=messenger
        )
        for col in hparam_cols
    }

    # Prepare saving of plot
    save_path = None
    if save_dir is not None:
        # Create save_dir directory if necessary
        mk_dir(path=save_dir, arg_name="save_dir")
        save_dir = pathlib.Path(save_dir)

    messenger("Plotting results for each hyperparameter")
    for metric in ["Mean Test Score", "Mean Fit Time"]:
        title = (
            metric_name.replace("_", "").title()
            if metric != "Mean Fit Time"
            else "Training Time"
        )
        filename_metric = metric.split(" ")[-1]
        for hparam, agg_df in aggregated_dfs.items():
            if save_dir is not None:
                save_path = (
                    save_dir
                    / f"inner_cv_{filename_metric}_HP_{col_to_name[hparam].replace(' ', '_')}.png"
                )
            _plot_hparam(
                agg_df,
                col=col_to_name[hparam],
                catplot_kwargs=catplot_kwargs,
                y=metric,
                title=title,
                save_path=save_path,
            )

    timer.stamp()
    messenger(
        "Finished plotting scores per hyperparameter. "
        f"Function took: {timer.get_total_time()}"
    )


def _extract_hparam_names(cols):
    split_cols = [col.replace("param_", "").split("__") for col in cols]
    i = 1
    while True:
        terms = ["__".join(col_terms[-i:]) for col_terms in split_cols]
        if len(terms) == len(set(terms)):
            break
        i += 1
    return {
        col: name.replace("__", " ").replace("_", " ").title()
        for col, name in zip(cols, terms)
    }


def _aggregate_folds_by_hparam(inner_results, col, col_to_name, messenger):
    # Ensure dtypes are numeric
    _col_to_float(df=inner_results, col="mean_test_score", messenger=messenger)
    _col_to_float(df=inner_results, col="mean_fit_time", messenger=messenger)
    try:
        agg_df = (
            inner_results.groupby(["Repetition", col])
            .agg({"mean_test_score": "mean", "mean_fit_time": "mean"})
            .reset_index()
        )
        agg_df.columns = [
            "Repetition",
            col_to_name[col],
            "Mean Test Score",
            "Mean Fit Time",
        ]
    except Exception as e:
        messenger(f"Failed to aggregate folds by hparam: {e}")
        messenger(f"`inner_results` data frame: {inner_results}")
        messenger(f"  with columns: {inner_results.columns}")
        raise
    return agg_df


def _col_to_float(df, col, messenger):
    try:
        df[col] = df[col].astype(float)
    except Exception as e:
        messenger(f"Failed to convert `{col}` to float. Was {df[col].dtype}.")
        messenger(df[col])
        raise e


def _plot_hparam(
    agg_df, col, catplot_kwargs, y="Mean Test Score", title="", save_path=""
):
    # Clear figure
    plt.clf()

    # Set figure theme
    sns.set_style("whitegrid")

    pl = sns.catplot(data=agg_df, x=col, y=y, **catplot_kwargs)
    plt.xticks(rotation=45)

    if title:
        pl.set(title=title)

    # Add caption
    fig = pl.figure

    # Save plot
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")

    return fig
