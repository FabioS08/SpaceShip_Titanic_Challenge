from .base import ModelTrainer
from typing import Any
import pandas as pd


def resolve_trainer_thresholds(trainers: list[ModelTrainer], threshold: float | dict[str, float] | list[float] | tuple[float, ...] = 0.5) -> list[float]:

    '''
    It resolves the classification threshold for each trainer from a scalar, dictionary or sequence.

    Parameters
    ----------
     trainers:list[ModelTrainer]
      List of ModelTrainer instances.

     threshold:float | dict[str, float] | list[float] | tuple[float, ...]
      Global threshold, mapping of model name/key to threshold, or sequence of thresholds.

    Returns
    -------
     list[float]
      List of resolved thresholds matching the order of trainers.
    '''

    if isinstance(threshold, (int, float)):
        return [float(threshold)] * len(trainers)

    if isinstance(threshold, dict):

        resolved = []

        for t in trainers:
            th = None

            for key in (t.model_name, t.model_name.lower(), t.model_preset_key, t.model_preset_key.lower(), "default"):
                if key in threshold:
                    th = threshold[key]
                    break

            resolved.append(th if th is not None else 0.5)

        return resolved

    if isinstance(threshold, (list, tuple)):

        if len(threshold) != len(trainers):
            raise ValueError(f"Number of thresholds ({len(threshold)}) must match number of trainers ({len(trainers)}).")

        return [th for th in threshold]

    raise TypeError(f"Unsupported threshold type '{type(threshold).__name__}'. Expected float, dict or list/tuple.")


def evaluate_trainers(trainers: list[ModelTrainer], thresholds: list[float]) -> list[dict[str, Any]]:

    '''
    It evaluates each trainer on its shared test holdout set with its corresponding threshold.

    Parameters
    ----------
     trainers:list[ModelTrainer]
      List of fitted ModelTrainer instances.

     thresholds:list[float]
      Sequence of decision thresholds corresponding to each trainer.

    Returns
    -------
     list[dict[str, Any]]
      List of metric dictionaries including model name and threshold.
    '''

    rows: list[dict[str, Any]] = []

    for t, th in zip(trainers, thresholds):

        metrics = t.evaluate_test(threshold = th, verbose = False)
        row: dict[str, Any] = {"Model": t.model_name, "Threshold": th, **metrics}
        rows.append(row)

    return rows


def format_leaderboard(rows: list[dict[str, Any]], sort_by: str = "Accuracy") -> pd.DataFrame:

    '''
    It formats and sorts the evaluation rows into a structured leaderboard DataFrame.

    Parameters
    ----------
     rows:list[dict[str, Any]]
      List of evaluation metric dictionaries.

     sort_by:str
      Metric column to sort by in descending order [Default = 'Accuracy'].

    Returns
    -------
     pd.DataFrame
      Formatted and sorted leaderboard DataFrame.
    '''

    df = pd.DataFrame(rows)
    cols = ["Model", "Threshold", "Accuracy", "ROC-AUC", "F1-Score", "Precision", "Recall", "Log-Loss"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    if sort_by in df.columns:
        df = df.sort_values(by = sort_by, ascending = False).reset_index(drop = True)

    return df


def print_leaderboard(df: pd.DataFrame) -> None:

    '''
    It prints the formatted leaderboard table to the console.

    Parameters
    ----------
     df:pd.DataFrame
      Leaderboard DataFrame to display.
    '''

    w = 80
    print("\n" + "=" * w)
    print("🏆 Shared Test Set Multi-Model Leaderboard".center(w))
    print("=" * w)
    print(df.to_string(index = False, justify = "center"))
    print("=" * w + "\n")


def compare_trainers(trainers: list[ModelTrainer], threshold: float | dict[str, float] | list[float] | tuple[float, ...] = 0.5,  
                     sort_by: str = "Accuracy") -> pd.DataFrame:
    
    '''
    It evaluates a collection of fitted ModelTrainer instances on their shared test holdout set, returning a side-by-side leaderboard DataFrame.

    Parameters
    ----------
     trainers:list[ModelTrainer]
      List of fitted ModelTrainer instances to compare.

     threshold:float | dict[str, float] | list[float] | tuple[float, ...]
      Decision threshold(s) for converting probabilities to binary predictions.
      Accepts a single float (applied to all models), a dict mapping model names
      or preset keys to thresholds, or a sequence matching the trainers list [Default = 0.5].

     sort_by:str
      Metric column to sort by in descending order [Default = 'Accuracy'].

    Returns
    -------
     pd.DataFrame
      Formatted and sorted leaderboard DataFrame.
    '''

    thresholds = resolve_trainer_thresholds(trainers, threshold)
    rows = evaluate_trainers(trainers, thresholds)
    df = format_leaderboard(rows, sort_by = sort_by)
    print_leaderboard(df)

    return df
