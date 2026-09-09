from __future__ import annotations

from typing import Any, Mapping, Sequence
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import time


def _validate_heatmap_inputs(results_df: pd.DataFrame, metrics: list[str], model_col: str, preset_col: str) -> None:

    '''
    It validates that the input DataFrame contains the required model, preset and metric columns.

    Parameters
    ----------
     results_df:pd.DataFrame
      Benchmark results DataFrame to validate.

     metrics:list[str]
      List of metric column names to verify.

     model_col:str
      Column name for model identifiers.

     preset_col:str
      Column name for dataset preset identifiers.
    '''

    if len(metrics) == 0:
        raise ValueError("At least one metric must be provided for heatmap visualization.")

    required_cols = [model_col, preset_col] + metrics

    for col in required_cols:
        if col not in results_df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame columns: {list(results_df.columns)}")


def _compute_sorted_pivots(results_df: pd.DataFrame, metrics: list[str], model_col: str, preset_col: str) -> list[pd.DataFrame]:

    '''
    It builds pivot tables for each metric and orders models by their average score on the primary metric.

    Parameters
    ----------
     results_df:pd.DataFrame
      Benchmark results DataFrame.

     metrics:list[str]
      List of metric names to pivot.

     model_col:str
      Column name for model identifiers.

     preset_col:str
      Column name for dataset preset identifiers.

    Returns
    -------
     list[pd.DataFrame]
      List of pivoted DataFrames ordered by primary metric performance.
    '''

    pivots: list[pd.DataFrame] = []
    for m in metrics:

        p = results_df.pivot(index = model_col, columns = preset_col, values = m)
        pivots.append(p)

    # Sort models by their average score on the primary metric (descending)
    primary_pivot = pivots[0]
    avg_order = primary_pivot.mean(axis = 1).sort_values(ascending = False).index

    return [p.loc[avg_order] for p in pivots]


def _render_heatmap_subplot(ax: Axes, p_df: pd.DataFrame, metric_name: str, cmap: str, annot_size: int, is_first: bool) -> None:

    '''
    It renders a single annotated heatmap on the designated subplot axis.

    Parameters
    ----------
     ax:Axes
      Subplot axis to render upon.

     p_df:pd.DataFrame
      Pivoted DataFrame containing metric values.

     metric_name:str
      Name of the metric displayed.

     cmap:str
      Colormap name for the heatmap.

     annot_size:int
      Font size for cell annotations.

     is_first:bool
      Whether this is the leftmost subplot in the figure.
    '''

    sns.heatmap(
        p_df,
        annot = True,
        fmt = ".4f",
        cmap = cmap,
        cbar = True,
        linewidths = 0.5,
        ax = ax,
        annot_kws = {"size": annot_size, "weight": "bold"},
    )

    ax.set_title(f"{metric_name} across Model & Dataset Presets", fontsize = 13, fontweight = "bold", pad = 12)
    ax.set_xlabel("Dataset Preset", fontsize = 11, fontweight = "bold")
    ax.set_ylabel("Model Architecture" if is_first else "", fontsize = 11, fontweight = "bold")
    ax.tick_params(axis = "x", rotation = 30)


def plot_benchmark_heatmaps(results_df: pd.DataFrame, metrics: tuple[str, ...] | list[str] = ("Accuracy", "ROC-AUC"), model_col: str = "Model",
                            preset_col: str = "Preset", cmaps: tuple[str, ...] | list[str] = ("YlGnBu", "mako"), figsize: tuple[int, int] = (18, 6),
                            annot_size: int = 10, show: bool = True) -> tuple[Figure, np.ndarray]:

    '''
    It creates side-by-side heatmaps displaying model performance metrics across dataset presets.

    Parameters
    ----------
     results_df:pd.DataFrame
      Benchmark results DataFrame containing model, preset, and metric columns.

     metrics:tuple[str, ...] | list[str]
      Metrics to visualize as separate heatmaps [Default = ('Accuracy', 'ROC-AUC')].

     model_col:str
      Column name for model names [Default = 'Model'].

     preset_col:str
      Column name for dataset preset names [Default = 'Preset'].

     cmaps:tuple[str, ...] | list[str]
      Colormap names for each metric heatmap [Default = ('YlGnBu', 'mako')].

     figsize:tuple[int, int]
      Figure size dimensions (width, height) [Default = (18, 6)].

     annot_size:int
      Font size for heatmap cell annotations [Default = 10].

     show:bool
      Whether to call plt.show() [Default = True].

    Returns
    -------
     tuple[Figure, np.ndarray]
      Tuple containing the matplotlib Figure and Axes array.
    '''

    metrics_list = list(metrics)
    _validate_heatmap_inputs(results_df = results_df, metrics = metrics_list, model_col = model_col, preset_col = preset_col)
    pivots = _compute_sorted_pivots(results_df = results_df, metrics = metrics_list, model_col = model_col, preset_col = preset_col)

    fig, axes = plt.subplots(1, len(metrics_list), figsize = figsize, squeeze = False)
    axes_flat: np.ndarray = axes[0]

    for idx, (metric_name, p_df) in enumerate(zip(metrics_list, pivots)):

        ax: Axes = axes_flat[idx]
        cmap = cmaps[idx % len(cmaps)]
        _render_heatmap_subplot(ax = ax, p_df = p_df, metric_name = metric_name, cmap = cmap, annot_size = annot_size, is_first = (idx == 0))

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes_flat


def plot_model_sensitivity(results_df: pd.DataFrame, metric: str = "Accuracy", model_col: str = "Model", preset_col: str = "Preset", 
                           palette: str | list[Any] | None = None, figsize: tuple[int, int] = (14, 6), ylim: tuple[float, float] | None = (0.70, 0.83),
                           benchmark_line: float | None = 0.80, benchmark_label: str = "80% Accuracy Benchmark", show: bool = True) -> tuple[Figure, Axes]:

    '''
    It visualizes model metric sensitivity across dataset presets using a grouped bar chart.

    Parameters
    ----------
     results_df:pd.DataFrame
      Benchmark results DataFrame containing model, preset, and metric columns.

     metric:str
      Performance metric to plot on the y-axis [Default = 'Accuracy'].

     model_col:str
      Column name for model identifiers [Default = 'Model'].

     preset_col:str
      Column name for dataset preset identifiers [Default = 'Preset'].

     palette:str | list[Any] | None
      Color palette for presets. If None, uses seaborn 'deep' palette [Default = None].

     figsize:tuple[int, int]
      Figure size dimensions (width, height) [Default = (14, 6)].

     ylim:tuple[float, float] | None
      Y-axis lower and upper limits [Default = (0.70, 0.83)].

     benchmark_line:float | None
      Horizontal reference threshold line value [Default = 0.80].

     benchmark_label:str
      Label for the benchmark line in the plot [Default = '80% Accuracy Benchmark'].

     show:bool
      Whether to call plt.show() [Default = True].

    Returns
    -------
     tuple[Figure, Axes]
      Tuple containing the matplotlib Figure and Axes object.
    '''

    if metric not in results_df.columns:
        raise KeyError(f"Metric '{metric}' not found in DataFrame columns: {list(results_df.columns)}")

    n_presets = results_df[preset_col].nunique()
    color_palette = palette if palette is not None else sns.color_palette("deep", n_presets)

    fig, ax = plt.subplots(figsize = figsize)

    sns.barplot(data = results_df, x = model_col, y = metric, hue = preset_col, palette = color_palette, ax = ax)

    ax.set_title(f"Model {metric} Sensitivity Across Dataset Presets", fontsize = 14, fontweight = "bold", pad = 15)
    ax.set_xlabel("Model Architecture", fontsize = 12, fontweight = "bold")
    ax.set_ylabel(f"Test {metric}", fontsize = 12, fontweight = "bold")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(title = "Dataset Preset", bbox_to_anchor = (1.02, 1), loc = "upper left", frameon = True)
    ax.tick_params(axis = "x", rotation = 15)
    plt.setp(ax.get_xticklabels(), ha = "right")
    ax.grid(axis = "y", linestyle = "--", alpha = 0.7)

    if benchmark_line is not None:
        ax.axhline(benchmark_line, color = "crimson", linestyle = "--", linewidth = 1.2, alpha = 0.7, label = benchmark_label)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def _prepare_predictions_df(predictions: pd.DataFrame | Mapping[Any, Any]) -> pd.DataFrame:

    '''
    It converts a dictionary or DataFrame of model predictions into a standardized binary integer DataFrame.

    Parameters
    ----------
     predictions:pd.DataFrame | Mapping[Any, Any]
      Container with model predictions.

    Returns
    -------
     pd.DataFrame
      Standardized DataFrame with binary integer predictions.
    '''

    if isinstance(predictions, pd.DataFrame):
        df = predictions.copy()

    elif isinstance(predictions, Mapping):
        df = pd.DataFrame(predictions)

    else:
        raise TypeError(f"Unsupported predictions type '{type(predictions).__name__}'. Expected pd.DataFrame or dict.")

    if df.empty:
        raise ValueError("Predictions container is empty.")

    return df.astype(int)


def _render_correlation_heatmap(ax: Axes, corr_matrix: pd.DataFrame) -> None:

    '''
    It renders a pairwise prediction correlation heatmap across models.

    Parameters
    ----------
     ax:Axes
      Subplot axis to render upon.

     corr_matrix:pd.DataFrame
      Pairwise Pearson correlation matrix between model predictions.
    '''

    sns.heatmap(corr_matrix, annot = True, fmt = ".3f", cmap = "Blues", cbar = True, linewidths = 0.5, ax = ax, annot_kws = {"size": 11, "weight": "bold"})
    ax.set_title("Pairwise Prediction Correlation Between Models", fontsize = 12, fontweight = "bold", pad = 12)


def _render_vote_distribution(ax: Axes, votes: pd.Series, n_models: int, min_votes: int) -> None:

    '''
    It renders the distribution of ensemble positive votes with threshold highlighting.

    Parameters
    ----------
     ax:Axes
      Subplot axis to render upon.

     votes:pd.Series
      Series containing the count of positive votes per sample.

     n_models:int
      Total number of models in the ensemble.

     min_votes:int
      Minimum votes required for a positive classification.
    '''

    vote_counts = votes.value_counts().sort_index()
    bar_colors = ["#d9534f" if v < min_votes else "#5cb85c" for v in vote_counts.index]

    ax.bar(vote_counts.index, vote_counts.to_numpy(), color = bar_colors, edgecolor = "#333", linewidth = 0.8, alpha = 0.85)
    ax.set_title(f"Distribution of Ensemble Votes (Out of {n_models} Models)", fontsize = 12, fontweight = "bold", pad = 12)
    ax.set_xlabel("Number of Models Voting 'Transported = True'", fontsize = 11, fontweight = "bold")
    ax.set_ylabel("Number of Test Passengers", fontsize = 11, fontweight = "bold")
    ax.set_xticks(range(n_models + 1))
    ax.grid(axis = "y", linestyle = "--", alpha = 0.6)

    max_count = vote_counts.max() if len(vote_counts) > 0 else 1
    ax.set_ylim(0, max_count * 1.20)
    label_offset = max(max_count * 0.015, 1)

    for x_val, y_val in zip(vote_counts.index, vote_counts.to_numpy()):
        label = f"{y_val:,}\n({y_val / len(votes) * 100:.1f}%)"
        ax.text(x_val, y_val + label_offset, label, ha = "center", va = "bottom", fontsize = 9, fontweight = "bold")

    threshold_x = min_votes - 0.5
    ax.axvline(threshold_x, color = "black", linestyle = "--", linewidth = 1.5, label = f"Majority Threshold (≥ {min_votes} Votes)")
    ax.legend(loc = "upper center", frameon = True)


def _print_consensus_summary(votes: pd.Series, ensemble_predictions: pd.Series, n_models: int) -> None:

    '''
    It prints a formatted summary table of ensemble voting consensus and prediction rates.

    Parameters
    ----------
     votes:pd.Series
      Series of vote counts per sample.

     ensemble_predictions:pd.Series
      Final boolean ensemble predictions.

     n_models:int
      Total number of models in the ensemble.
    '''

    unanimous_count = (votes == 0).sum() + (votes == n_models).sum()
    total = len(votes)
    pos_count = ensemble_predictions.sum()
    neg_count = (~ensemble_predictions).sum()

    print(f"\n📊 Ensemble Consensus Summary:")
    print(f"  • Total Competition Test Passengers: {total:,}")
    print(f"  • Unanimous Agreement (0 or {n_models} votes): {unanimous_count:,} ({unanimous_count / total * 100:.1f}%)")
    print(f"  • Final Ensemble Transported = True:  {pos_count:,} ({pos_count / total * 100:.2f}%)")
    print(f"  • Final Ensemble Transported = False: {neg_count:,} ({neg_count / total * 100:.2f}%)\n")


def plot_ensemble_consensus(predictions: pd.DataFrame | Mapping[Any, Any], min_votes: int | None = None,
                            figsize: tuple[int, int] = (16, 5), show: bool = True,
                            verbose: bool = True, return_fig: bool = False) -> pd.Series | tuple[pd.Series, Figure, np.ndarray]:

    '''
    It evaluates pairwise prediction correlation, computes majority voting, visualizes vote distributions, and returns ensemble predictions.

    Parameters
    ----------
     predictions:pd.DataFrame | Mapping[Any, Any]
      DataFrame or dictionary containing binary model predictions for each test instance.

     min_votes:int | None
      Minimum votes required to classify an instance as positive. If None, uses strict majority (n_models // 2 + 1) [Default = None].

     figsize:tuple[int, int]
      Figure dimensions (width, height) for the side-by-side subplots [Default = (16, 5)].

     show:bool
      Whether to call plt.show() [Default = True].

     verbose:bool
      Whether to print consensus summary statistics to stdout [Default = True].

     return_fig:bool
      Whether to return (ensemble_predictions, fig, axes) instead of just ensemble_predictions [Default = False].

    Returns
    -------
     pd.Series | tuple[pd.Series, Figure, np.ndarray]
      Boolean Series containing final ensemble predictions (or tuple with Figure and Axes if return_fig is True).
    '''

    preds_df = _prepare_predictions_df(predictions = predictions)
    n_models = preds_df.shape[1]
    required_votes = min_votes if min_votes is not None else (n_models // 2) + 1

    votes = preds_df.sum(axis = 1)
    ensemble_predictions = (votes >= required_votes).astype(bool)

    corr_matrix = preds_df.corr()

    fig, axes = plt.subplots(1, 2, figsize = figsize)
    axes_flat: np.ndarray = axes

    _render_correlation_heatmap(ax = axes_flat[0], corr_matrix = corr_matrix)
    _render_vote_distribution(ax = axes_flat[1], votes = votes, n_models = n_models, min_votes = required_votes)

    plt.tight_layout()

    if show:
        plt.show()

    if verbose:
        _print_consensus_summary(votes = votes, ensemble_predictions = ensemble_predictions, n_models = n_models)

    ensemble_predictions.attrs["fig"] = fig
    ensemble_predictions.attrs["axes"] = axes_flat
    ensemble_predictions.attrs["votes"] = votes
    ensemble_predictions.attrs["corr_matrix"] = corr_matrix

    if return_fig:
        return ensemble_predictions, fig, axes_flat

    return ensemble_predictions


def _resolve_test_passenger_ids(test_csv_path: str | Path = "dataset/test.csv", passenger_ids: pd.Series | np.ndarray | Sequence[str] | None = None) -> pd.Series:

    '''
    It resolves and extracts competition test PassengerIds from an array or a CSV file.

    Parameters
    ----------
     test_csv_path:str | Path
      Path to competition test or sample submission CSV [Default = 'dataset/test.csv'].

     passenger_ids:pd.Series | np.ndarray | Sequence[str] | None
      Optional explicit sequence of passenger IDs [Default = None].

    Returns
    -------
     pd.Series
      Series of test PassengerIds.
    '''

    if passenger_ids is not None:
        return pd.Series(passenger_ids, name = "PassengerId")

    p_test = Path(test_csv_path)

    if not p_test.exists():

        fallback = Path("dataset/sample_submission.csv") if p_test.name == "test.csv" else Path("dataset/test.csv")
        
        if fallback.exists():
            p_test = fallback

        else:
            raise FileNotFoundError(f"Test dataset not found at '{p_test}'. Unable to load PassengerId column.")

    raw_test = pd.read_csv(p_test)

    if "PassengerId" not in raw_test.columns:
        raise KeyError(f"File at '{p_test}' does not contain a 'PassengerId' column.")

    return raw_test["PassengerId"]


def _prepare_submission_predictions(predictions: pd.Series | np.ndarray | pd.DataFrame | Sequence[Any],
                                    n_expected: int) -> np.ndarray:

    '''
    It extracts, formats and validates binary boolean predictions matching the expected passenger count.

    Parameters
    ----------
     predictions:pd.Series | np.ndarray | pd.DataFrame | Sequence[Any]
      Predicted binary labels (True/False or 1/0).

     n_expected:int
      Expected number of predictions.

    Returns
    -------
     np.ndarray
      1D boolean array of predictions.
    '''

    if isinstance(predictions, pd.DataFrame):

        if "Transported" in predictions.columns:
            preds_series = predictions["Transported"]

        elif predictions.shape[1] == 1:
            preds_series = predictions.iloc[:, 0]

        else:
            raise ValueError("Predictions DataFrame must contain a 'Transported' column or exactly one column.")

    elif isinstance(predictions, pd.Series):
        preds_series = predictions

    elif isinstance(predictions, (np.ndarray, list, tuple)):
        preds_series = pd.Series(predictions)

    else:
        preds_series = pd.Series(list(predictions))

    if len(preds_series) != n_expected:
        raise ValueError(f"Length mismatch: {n_expected:,} Passenger IDs vs {len(preds_series):,} predictions.")

    return preds_series.to_numpy().astype(bool)


def _print_submission_summary(output_path: Path, sub_df: pd.DataFrame) -> None:

    '''
    It prints confirmation and summary statistics of the exported submission DataFrame.

    Parameters
    ----------
     output_path:Path
      Path where the submission file was saved.

     sub_df:pd.DataFrame
      Exported submission DataFrame.
    '''

    null_count = int(sub_df.isna().sum().sum())

    print(f"💾 Official Ensemble Submission saved to: '{output_path}'")
    print(f"• Total Rows: {len(sub_df):,}")
    print(f"• Null Values: {null_count}")
    print("\nFirst 10 submission rows:")


def save_submission(predictions: pd.Series | np.ndarray | pd.DataFrame | Sequence[Any], output_path: str | Path = "submissions/submission.csv",
                    test_csv_path: str | Path = "dataset/test.csv", passenger_ids: pd.Series | np.ndarray | Sequence[str] | None = None,
                    verbose: bool = True) -> pd.DataFrame:

    '''
    It constructs, validates and saves a Kaggle SpaceShip Titanic submission CSV file.

    Parameters
    ----------
     predictions:pd.Series | np.ndarray | pd.DataFrame | Sequence[Any]
      Predicted binary labels (True/False or 1/0) for competition test passengers.

     output_path:str | Path
      Destination path for the exported submission CSV [Default = 'submissions/submission.csv'].

     test_csv_path:str | Path
      Path to competition test or sample submission CSV used to extract PassengerId if not provided [Default = 'dataset/test.csv'].

     passenger_ids:pd.Series | np.ndarray | Sequence[str] | None
      Optional explicit sequence of passenger IDs. If None, loaded from test_csv_path [Default = None].

     verbose:bool
      Whether to print confirmation and summary statistics to stdout [Default = True].

    Returns
    -------
     pd.DataFrame
      Submission DataFrame containing 'PassengerId' and 'Transported' columns.
    '''

    p_ids = _resolve_test_passenger_ids(test_csv_path = test_csv_path, passenger_ids = passenger_ids)
    transported_bool = _prepare_submission_predictions(predictions = predictions, n_expected = len(p_ids))

    sub_df = pd.DataFrame({
                                "PassengerId": p_ids.to_numpy(),
                                "Transported": transported_bool
                            })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents = True, exist_ok = True)
    sub_df.to_csv(out_file, index = False)

    if verbose:
        _print_submission_summary(output_path = out_file, sub_df = sub_df)

    return sub_df


def _fit_single_ensemble_model(spec: Mapping[str, Any], full_datasets: Mapping[str, Mapping[str, Any]]) -> tuple[str, Any, np.ndarray, np.ndarray, float, float]:

    '''
    It instantiates, trains and evaluates competition test predictions for a single ensemble model specification.

    Parameters
    ----------
     spec:Mapping[str, Any]
      Dictionary defining model name, trainer class, dataset preset, threshold, and fit kwargs.

     full_datasets:Mapping[str, Mapping[str, Any]]
      Dictionary mapping preset names to full train and competition test datasets.

    Returns
    -------
     tuple[str, Any, np.ndarray, np.ndarray, float, float]
      Tuple containing (model_name, fitted_trainer, probabilities, binary_predictions, threshold, elapsed_time).
    '''

    m_name = str(spec.get("name", "Model"))
    p_name = str(spec["preset"])
    thr = float(spec.get("threshold", 0.50))
    fit_kwargs = spec.get("fit_kwargs", {})

    if p_name not in full_datasets:
        raise KeyError(f"Preset '{p_name}' not found in full_datasets. Available presets: {list(full_datasets.keys())}")

    ds = full_datasets[p_name]
    trainer_cls = spec["trainer_cls"]

    t_start = time.time()

    trainer = trainer_cls(X_train = ds["X_train"], y_train = ds["y_train"], X_competition_test = ds["X_test"])
    trainer.fit(**fit_kwargs)

    probs = trainer.predict_proba(ds["X_test"])
    preds = (probs > thr).astype(int)

    elapsed = time.time() - t_start

    return m_name, trainer, probs, preds, thr, elapsed


def _print_ensemble_model_status(m_name: str, p_name: str, thr: float, elapsed: float, preds: np.ndarray) -> None:

    '''
    It prints the training duration and predicted positive class distribution for an ensemble model.

    Parameters
    ----------
     m_name:str
      Model identifier name.

     p_name:str
      Dataset preset name.

     thr:float
      Decision threshold.

     elapsed:float
      Training and prediction elapsed time in seconds.

     preds:np.ndarray
      Binary predictions array.
    '''

    n_pos = int(preds.sum())
    total = len(preds)
    pct = (n_pos / total * 100) if total > 0 else 0.0

    print(f"  ✓ {m_name:<18} | Preset: {p_name:<16} | Threshold: {thr:.2f} | Time: {elapsed:5.2f}s | Predicted Transported: {n_pos:4d}/{total} ({pct:.1f}%)")


def train_ensemble_models(ensemble_specs: Sequence[Mapping[str, Any]], full_datasets: Mapping[str, Mapping[str, Any]],
                          verbose: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:

    '''
    It trains ensemble models on full datasets without validation splitting and generates competition test predictions.

    Parameters
    ----------
     ensemble_specs:Sequence[Mapping[str, Any]]
      Sequence of dictionaries containing model configuration specs.

     full_datasets:Mapping[str, Mapping[str, Any]]
      Dictionary of full train and competition test datasets keyed by preset name.

     verbose:bool
      Whether to print progress banners and status lines [Default = True].

    Returns
    -------
     tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]
      Tuple containing:
        - fitted_ensemble_trainers: Dictionary mapping model names to fitted trainer instances.
        - test_probabilities: Dictionary mapping model names to positive class probability arrays.
        - test_predictions: Dictionary mapping model names to binary integer prediction arrays.
    '''

    fitted_ensemble_trainers: dict[str, Any] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}

    n_models = len(ensemble_specs)

    if verbose:
        print(f"🚀 Training {n_models} Ensemble Models on Full Datasets & Generating Test Predictions...")
        print("=" * 120)

    for spec in ensemble_specs:

        m_name, trainer, probs, preds, thr, elapsed = _fit_single_ensemble_model(spec = spec, full_datasets = full_datasets)

        fitted_ensemble_trainers[m_name] = trainer
        test_probabilities[m_name] = probs
        test_predictions[m_name] = preds

        if verbose:
            _print_ensemble_model_status(m_name = m_name, p_name = str(spec.get("preset", "custom")), thr = thr, elapsed = elapsed, preds = preds)

    if verbose:

        print("=" * 120)
        print(f"✅ All {n_models} ensemble models successfully trained on full datasets!")

    return fitted_ensemble_trainers, test_probabilities, test_predictions