from __future__ import annotations


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split
from ..utils.model_presets import get_model_preset
from ..utils.data_split import DataSplit
from matplotlib.figure import Figure
from abc import ABC, abstractmethod
from matplotlib.axes import Axes
from typing import Any, Literal
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time


class ModelTrainer(ABC):

    '''
    Abstract Base Class for all the model training pipelines in the SpaceShip Titanic Challenge.

    It provides foundational workflows for:

      - Unified dataset splitting with a guaranteed, identical shared test holdout.
      - Training curve and evaluation curve tracking and plotting.
      - Standardized model persistence (save_model / load_model).
      - Parameter tuning driven by external configuration presets.
      - Standardized inference, evaluation and Kaggle submission generation.
    '''

    model_preset_key: str = "base"
    default_name: str = "ModelTrainer"

    def __init__(self, data_split: DataSplit | None = None, X_train: pd.DataFrame | None = None, y_train: pd.Series | None = None,
                 X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None, X_test: pd.DataFrame | None = None, y_test: pd.Series | None = None,
                 X_competition_test: pd.DataFrame | None = None, params: dict[str, Any] | None = None, random_state: int = 42):

        self.random_state = random_state

        if data_split is not None:
            self._set_data_split(data_split)

        else:

            if X_train is None or y_train is None:
                raise ValueError("Either 'data_split' or explicit ('X_train', 'y_train') must be provided.")

            self._set_data(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test, 
                           X_competition_test = X_competition_test)

        # Resolve hyperparameters
        preset_cfg = self._get_preset_config()
        self.params: dict[str, Any] = preset_cfg.get("default_params", {}).copy()
        if params is not None:
            self.params.update(params)

        self.model_name = self.default_name
        self.model: Any = None
        self.is_fitted: bool = False
        self.history: dict[str, list[float] | list[int]] = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": [], "iterations": [],}
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = 0.0

        # Instantiate the initial model
        self.model = self._init_model(self.params)

    # -------------------------------------------------------------------------
    # Data Management
    # -------------------------------------------------------------------------

    def _set_data(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None, 
                  X_test: pd.DataFrame | None = None, y_test: pd.Series | None = None, X_competition_test: pd.DataFrame | None = None) -> None:

        '''
        It stores and formats train, validation, test and optional competition test partitions.

        Parameters
        ----------
         X_train:pd.DataFrame
          Training feature matrix.

         y_train:pd.Series
          Training target labels.
         
         X_val:pd.DataFrame | None
          Optional validation feature matrix.
         
         y_val:pd.Series | None
          Optional validation target labels.
         
         X_test:pd.DataFrame | None
          Optional test holdout feature matrix.
         
         y_test:pd.Series | None
          Optional test holdout target labels.
         
         X_competition_test:pd.DataFrame | None
          Optional unlabeled competition test feature matrix.
        '''

        self.X_train = X_train.copy()
        self.y_train = y_train.copy().astype(int)
        self.X_val = X_val.copy() if X_val is not None else pd.DataFrame()
        self.y_val = y_val.copy().astype(int) if y_val is not None else pd.Series(dtype = int)
        self.X_test = X_test.copy() if X_test is not None else pd.DataFrame()
        self.y_test = y_test.copy().astype(int) if y_test is not None else pd.Series(dtype = int)
        self.X_competition_test = X_competition_test.copy() if X_competition_test is not None else None


    def _set_data_split(self, data_split: DataSplit) -> None:

        '''
        It unpacks and assigns train, validation, test and optional competition test
        partitions from a DataSplit instance.

        Parameters
        ----------
         data_split:DataSplit
          Container holding partitioned feature matrices and target series.
        '''

        self._set_data(**vars(data_split))

    # -------------------------------------------------------------------------
    # Preset Configuration Resolution
    # -------------------------------------------------------------------------

    def _get_preset_config(self) -> dict[str, Any]:

        '''
        It retrieves hyperparameters and settings from the active preset. If the
        preset key is "base" or not found, an empty dict is returned, allowing
        the trainer to fall back to defaults.

        Returns
        -------
         dict[str, Any]
          Model preset configuration.
        '''

        if self.model_preset_key == "base":
            return {}
        
        try:

            return get_model_preset(self.model_preset_key)
        
        except KeyError:
            return {}
            
    # -------------------------------------------------------------------------
    # Factory: Data Splitting (Shared Test Holdout Guarantee)
    # -------------------------------------------------------------------------

    @staticmethod
    def create_data_splits(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, val_size: float = 0.20, X_competition_test: pd.DataFrame | None = None,
                           stratify: bool = True, random_state: int = 42) -> DataSplit:
        
        '''
        It partitions the labeled data into a Train set, a Validation set and a clean, untouched Test holdout set.

        Parameters
        ----------
         X:pd.DataFrame
          Full labeled feature matrix.

         y:pd.Series
          Full labeled target vector.
         
         test_size:float
          Proportion of labeled data reserved as the identical shared test set [Default = 0.20].
         
         val_size:float
          Proportion of remaining data reserved for validation monitoring [Default = 0.20].
         
         X_competition_test:pd.DataFrame | None
          Optional unlabeled competition test set.
         
         stratify:bool
          Whether to stratify splits according to target class proportions [Default = True].
         
         random_state:int
          Random seed ensuring identical splits across all model instances [Default = 42].

        Returns
        -------
         DataSplit
          Dataclass containing the partitioned splits.
        '''

        y_int = y.copy().astype(int)

        # Extract the shared holdout test set
        strat_col = y_int.to_numpy() if stratify else None
        X_remaining, X_test, y_remaining, y_test = train_test_split(X, y_int, test_size = test_size, stratify = strat_col, random_state = random_state)

        # Split remaining data into Train and Validation sets
        val_rel_size = val_size / (1.0 - test_size)
        strat_rem = y_remaining.to_numpy() if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(X_remaining, y_remaining, test_size = val_rel_size, stratify = strat_rem, random_state = random_state)

        return DataSplit(
                            X_train              = X_train.reset_index(drop = True),
                            y_train              = y_train.reset_index(drop = True),
                            X_val                = X_val.reset_index(drop = True),
                            y_val                = y_val.reset_index(drop = True),
                            X_test               = X_test.reset_index(drop = True),
                            y_test               = y_test.reset_index(drop = True),
                            X_competition_test   = X_competition_test.copy() if X_competition_test is not None else None
                        )

    # -------------------------------------------------------------------------
    # Abstract Hooks
    # -------------------------------------------------------------------------

    @abstractmethod
    def _init_model(self, params: dict[str, Any]) -> Any:
        
        '''
        It instantiates the model with the given parameter dictionary.
        '''

        pass


    @abstractmethod
    def fit(self, **kwargs: Any) -> ModelTrainer:
        
        '''
        It fits the model on self.X_train and monitors performance on self.X_val.
        Populates self.history with training and evaluation curves.
        '''

        pass

    # -------------------------------------------------------------------------
    # Visualization: Train & Evaluation Curves
    # -------------------------------------------------------------------------

    def _resolve_curve_x_axis(self) -> list[int]:
        
        '''
        It resolves or constructs the iteration/epoch sequence for evaluation curves.

        Returns
        -------
         list[int]
          Sequence of iterations or epochs matching history data points.
        '''

        x_axis = self.history.get("iterations")
        train_loss = self.history.get("train_loss", [])
        val_loss = self.history.get("val_loss", [])

        if not x_axis or len(x_axis) != len(train_loss):
            return list(range(1, len(train_loss or val_loss) + 1))

        return [int(x) for x in x_axis]


    def _plot_loss_curve(self, ax: Axes, x_axis: list[int]) -> None:
        
        '''
        It plots training and validation loss curves on the provided subplot axis.

        Parameters
        ----------
         ax:Axes
          Subplot axis for loss visualization.

         x_axis:list[int]
          Sequence of iterations or epochs.
        '''

        if self.history.get("train_loss"):
            ax.plot(x_axis, self.history["train_loss"], label = "Train Loss", color = "#1f77b4", lw = 2)

        if self.history.get("val_loss"):

            val_loss = self.history["val_loss"]
            ax.plot(x_axis, val_loss, label = "Val Loss", color = "#ff7f0e", lw = 2)
            best_iter_idx = int(np.argmin(val_loss))
            best_val_loss = val_loss[best_iter_idx]
            best_x = x_axis[best_iter_idx]
            ax.axvline(best_x, color = "red", linestyle = "--", alpha = 0.7, label = f"Min Val Loss ({best_val_loss:.4f})")

        ax.set_title("Loss / Objective Function", fontsize = 11, fontweight = "semibold")
        ax.set_xlabel("Iteration / Epoch", fontsize = 10)
        ax.set_ylabel("Loss", fontsize = 10)
        ax.grid(True, linestyle = ":", alpha = 0.6)
        ax.legend(loc = "best", frameon = True, facecolor = "white", edgecolor = "#ddd")


    def _plot_metric_curve(self, ax: Axes, x_axis: list[int], metric_name: str) -> None:
        
        '''
        It plots training and validation metric progression curves on the provided subplot axis.

        Parameters
        ----------
         ax:Axes
          Subplot axis for metric visualization.

         x_axis:list[int]
          Sequence of iterations or epochs.

         metric_name:str
          Name of evaluation metric displayed.
        '''

        if self.history.get("train_metric"):
            ax.plot(x_axis, self.history["train_metric"], label = f"Train {metric_name}", color = "#2ca02c", lw = 2)

        if self.history.get("val_metric"):

            val_metric = self.history["val_metric"]
            ax.plot(x_axis, val_metric, label = f"Val {metric_name}", color = "#d62728", lw = 2)
            best_m_idx = int(np.argmax(val_metric))
            best_val_m = val_metric[best_m_idx]
            best_m_x = x_axis[best_m_idx]
            ax.axvline(best_m_x, color = "blue", linestyle = "--", alpha = 0.7, label = f"Peak Val {metric_name} ({best_val_m:.4f})")

        ax.set_title(f"{metric_name} Progression", fontsize = 11, fontweight = "semibold")
        ax.set_xlabel("Iteration / Epoch", fontsize = 10)
        ax.set_ylabel(metric_name, fontsize = 10)
        ax.grid(True, linestyle = ":", alpha = 0.6)
        ax.legend(loc = "best", frameon = True, facecolor = "white", edgecolor = "#ddd")


    def _create_empty_curve_figure(self, show: bool = True) -> Figure:
        
        '''
        It generates and displays a placeholder figure when no curve history is available.

        Parameters
        ----------
         show:bool
          Whether to call plt.show() [Default = True].

        Returns
        -------
         Figure
          Matplotlib figure object.
        '''

        fig, ax = plt.subplots(figsize = (6, 3))
        ax.text(0.5, 0.5, f"No iteration curve history recorded for {self.model_name}.",
                ha = "center", va = "center", fontsize = 11, color = "gray")
        ax.axis("off")

        if show:
            plt.show()

        return fig


    def _save_eval_figure(self, fig: Figure, save_path: str | Path) -> None:
        
        '''
        It saves figure to disk ensuring target directories exist.

        Parameters
        ----------
         fig:Figure
          Matplotlib figure to save.

         save_path:str | Path
          Target file path.
        '''

        p = Path(save_path)
        p.parent.mkdir(parents = True, exist_ok = True)
        fig.savefig(p, bbox_inches = "tight", dpi = 300)


    def plot_eval_curve(self, metric_name: str = "Accuracy", title: str | None = None, save_path: str | Path | None = None, show: bool = True, 
                        figsize: tuple[int, int] = (12, 5)) -> Figure:
        
        '''
        It visualizes the progression of training loss and evaluation metrics across
        iterations or epochs, allowing clear diagnosis of overfitting and optimal early stopping.

        Parameters
        ----------
         metric_name:str
          Name of evaluation metric displayed on secondary subplot [Default = 'Accuracy'].

         title:str | None
          Plot title. If None, defaults to the model name.
         
         save_path:str | Path | None
          Optional filepath to save figure to disk.
         
         show:bool
          Whether to call plt.show() [Default = True].
         
         figsize:tuple[int, int]
          Figure dimensions (width, height) [Default = (12, 5)].

        Returns
        -------
         Figure
          Matplotlib figure object.
        '''

        if not self.history or (not self.history.get("train_loss") and not self.history.get("val_loss")):
            return self._create_empty_curve_figure(show = show)

        has_loss = bool(self.history.get("train_loss") or self.history.get("val_loss"))
        has_metric = bool(self.history.get("train_metric") or self.history.get("val_metric"))

        n_plots = 2 if (has_loss and has_metric) else 1
        fig, axes = plt.subplots(1, n_plots, figsize = figsize, squeeze = False)
        fig_title = title or f"{self.model_name}: Learning & Evaluation Curves"
        fig.suptitle(fig_title, fontsize = 13, fontweight = "bold", y = 1.02)

        x_axis = self._resolve_curve_x_axis()

        ax_idx = 0
        if has_loss:
            self._plot_loss_curve(axes[0, ax_idx], x_axis)
            ax_idx += 1

        if has_metric:
            self._plot_metric_curve(axes[0, ax_idx], x_axis, metric_name)

        plt.tight_layout()

        if save_path is not None:
            self._save_eval_figure(fig, save_path)

        if show:
            plt.show()

        return fig

    # -------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------

    def tune_hyperparameters(self, preset: Literal["fast", "thorough"] = "fast", search_type: Literal["grid", "random"] = "grid",
                             n_iter: int = 10, cv: int = 3, scoring: str = "accuracy", verbose: bool = True) -> dict[str, Any]:

        '''
        It optimizes teh hyperparameters using search grids defined in the external model presets file.

        Parameters
        ----------
         preset:Literal['fast', 'thorough']
          Tuning preset tier defined in model_presets.py [Default = 'fast'].
         
         search_type:Literal['grid', 'random']
          GridSearchCV vs RandomizedSearchCV [Default = 'grid'].
         
         n_iter:int
          Number of iterations if search_type is 'random' [Default = 10].
         
         cv:int
          Stratified CV folds used during parameter evaluation [Default = 3].
         
         scoring:str
          Optimization metric string [Default = 'accuracy'].
         
         verbose:bool
          Whether to print search results [Default = True].

        Returns
        -------
         dict[str, Any]
          Best parameters discovered.
        '''
        
        preset_cfg = self._get_preset_config()
        grids = preset_cfg.get("tuning_grids", {})
        param_grid = grids.get(preset, {})

        if not param_grid:
            raise ValueError(f"No tuning grid found for model '{self.model_preset_key}' with preset '{preset}'.")

        base_estimator = self._init_model(self.params)
        skf = StratifiedKFold(n_splits = cv, shuffle = True, random_state = self.random_state)

        if search_type == "grid":
            search = GridSearchCV(base_estimator, param_grid, cv = skf, scoring = scoring, n_jobs = -1, verbose = 1 if verbose else 0)
        
        else:
            search = RandomizedSearchCV(base_estimator, param_grid, n_iter = n_iter, cv = skf, scoring = scoring,
                                        n_jobs = -1, random_state = self.random_state, verbose = 1 if verbose else 0)

        if verbose:
            print(f"⚙️ Running {search_type.upper()} search for {self.model_name} (Preset = '{preset}', Number of Folds = {cv})...")

        search.fit(self.X_train, self.y_train)
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

        # Update model to best estimator
        self.params.update(self.best_params_)
        self.model = search.best_estimator_
        self.is_fitted = True

        if verbose:
            print(f"  🏆 Best Score ({scoring}): {self.best_score_:.4f}")
            print(f"  🔧 Best Parameters: {self.best_params_}")

        return self.best_params_

    # -------------------------------------------------------------------------
    # Evaluation & Inference
    # -------------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        
        '''
        It computes the calibrated class probabilities for positive class (Transported = True).
        '''

        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} must be fitted before calling predict_proba().")

        if hasattr(self.model, "predict_proba"):

            probs = self.model.predict_proba(X)
            if probs.ndim == 2:
                return probs[:, 1]
            
            return probs
        
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-df))
        
        else:
            return self.model.predict(X).astype(float)


    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        
        '''
        It predicts the binary classification labels (True/False or 1/0).
        '''

        probs = self.predict_proba(X)
        return (probs > threshold).astype(bool)


    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:

        '''
        It computes standardized classification metrics from true labels, predictions and probabilities.

        Parameters
        ----------
         y_true:np.ndarray
          Ground truth binary labels.

         y_pred:np.ndarray
          Predicted binary class labels.

         y_prob:np.ndarray
          Predicted class probabilities.

        Returns
        -------
         dict[str, float]
          Dictionary containing standardized classification metrics.
        '''

        acc = float(accuracy_score(y_true, y_pred))

        try:
            roc = float(roc_auc_score(y_true, y_prob))
        
        except ValueError:
            roc = 0.5

        f1 = float(f1_score(y_true, y_pred, zero_division = 0))
        prec = float(precision_score(y_true, y_pred, zero_division = 0))
        rec = float(recall_score(y_true, y_pred, zero_division = 0))

        try:
            loss = float(log_loss(y_true, y_prob, labels = [0, 1]))

        except ValueError:
            loss = 0.0

        return {
                    "Accuracy": acc,
                    "ROC-AUC": roc,
                    "F1-Score": f1,
                    "Precision": prec,
                    "Recall": rec,
                    "Log-Loss": loss,
                }


    def _print_evaluation_summary(self, metrics: dict[str, float], split_name: str, threshold: float) -> None:

        '''
        It prints a formatted summary table of evaluation metrics.

        Parameters
        ----------
         metrics:dict[str, float]
          Dictionary of evaluation metrics.

         split_name:str
          Name of dataset split being evaluated.

         threshold:float
          Classification decision threshold.
        '''

        w = 70
        print("-" * w)
        print(f"📊 {self.model_name} — {split_name} Results (Threshold: {threshold:.3f})".center(w))
        print("-" * w)

        for k, v in metrics.items():
            print(f"  • {k:15s}: {v:.4f}" + (f" ({v * 100:.2f}%)" if k in ["Accuracy", "Precision", "Recall"] else ""))

        print("-" * w)


    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, split_name: str = "Evaluation", threshold: float = 0.5,
                 verbose: bool = True) -> dict[str, float]:
        
        '''
        It computes standardized classification metrics and prints a formatted summary table.

        Parameters
        ----------
         X:pd.DataFrame | np.ndarray
          Feature matrix for evaluation.

         y:pd.Series | np.ndarray
          Ground truth target labels.

         split_name:str
          Label used in the summary header [Default = 'Evaluation'].

         threshold:float
          Decision threshold for converting probabilities into class labels [Default = 0.5].

         verbose:bool
          Whether to print the formatted metrics table [Default = True].

        Returns
        -------
         dict[str, float]
          Dictionary containing calculated evaluation metrics.
        '''

        y_true = np.asarray(y).astype(int)
        y_prob = self.predict_proba(X)
        y_pred = (y_prob > threshold).astype(int)

        metrics = self._compute_metrics(y_true, y_pred, y_prob)

        if verbose:
            self._print_evaluation_summary(metrics, split_name, threshold)

        return metrics


    def evaluate_test(self, threshold: float = 0.5, verbose: bool = True) -> dict[str, float]:
        
        '''
        It evaluates model performance on the shared, untouched test holdout set.
        '''

        if self.X_test.empty or self.y_test.empty:
            raise ValueError("Test holdout split (X_test, y_test) is empty.")
        
        return self.evaluate(self.X_test, self.y_test, split_name = "Shared Test Holdout", threshold = threshold, verbose = verbose)

    # -------------------------------------------------------------------------
    # Submission Generation
    # -------------------------------------------------------------------------

    def _load_test_passenger_ids(self, test_csv_path: str | Path) -> pd.Series:

        '''
        It loads and validates passenger identifiers from the raw competition test dataset.

        Parameters
        ----------
         test_csv_path:str | Path
          Path to the raw test CSV file.

        Returns
        -------
         pd.Series
          Series of passenger identifiers.
        '''

        p_test = Path(test_csv_path)

        if not p_test.exists():
            raise FileNotFoundError(f"Test file not found at '{p_test}'.")

        raw_test = pd.read_csv(p_test)

        if "PassengerId" not in raw_test.columns:
            raise KeyError("The raw test dataset does not contain a 'PassengerId' column.")

        return raw_test["PassengerId"]


    def _get_competition_features(self) -> pd.DataFrame:

        '''
        It retrieves and validates competition test features.

        Returns
        -------
         pd.DataFrame
          Competition test feature matrix.
        '''

        if self.X_competition_test is not None:
            return self.X_competition_test

        raise ValueError("No competition test features (X_competition_test) provided.")


    def _save_submission(self, sub_df: pd.DataFrame, output_path: str | Path) -> None:

        '''
        It saves the submission dataframe to disk ensuring target directories exist.

        Parameters
        ----------
         sub_df:pd.DataFrame
          Submission dataframe containing predictions.

         output_path:str | Path
          Target CSV file path.
        '''

        out = Path(output_path)
        out.parent.mkdir(parents = True, exist_ok = True)
        sub_df.to_csv(out, index = False)
        print(f"💾 Kaggle submission saved to '{out}' ({len(sub_df):,} records).")


    def generate_submission(self, test_csv_path: str | Path = "dataset/test.csv", output_path: str | Path = "submission.csv", 
                            threshold: float = 0.5) -> pd.DataFrame:
        
        '''
        It generates competition predictions for Kaggle submission.

        Parameters
        ----------
         test_csv_path:str | Path
          Path to the raw test CSV file [Default = 'dataset/test.csv'].

         output_path:str | Path
          Path where the submission file will be saved [Default = 'submission.csv'].

         threshold:float
          Decision threshold for converting probabilities into class labels [Default = 0.5].
        
        Returns
        -------
         pd.DataFrame
          Submission dataframe containing PassengerId and predicted Transported labels.
        '''

        passenger_ids = self._load_test_passenger_ids(test_csv_path)
        features = self._get_competition_features()
        preds = self.predict(features, threshold = threshold)

        sub_df = pd.DataFrame({"PassengerId": passenger_ids, "Transported": preds})
        self._save_submission(sub_df, output_path)

        return sub_df

    # -------------------------------------------------------------------------
    # Model Persistence
    # -------------------------------------------------------------------------

    def save_model(self, path: str | Path) -> Path:

        '''
        It serializes the trained model state, parameters and training history to disk.
        '''

        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before saving.")

        p = Path(path)
        p.parent.mkdir(parents = True, exist_ok = True)

        payload = {
                    "model_name": self.model_name,
                    "model_preset_key": self.model_preset_key,
                    "params": self.params,
                    "model": self.model,
                    "history": self.history,
                    "best_params_": self.best_params_,
                    "best_score_": self.best_score_,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
        joblib.dump(payload, p)
        print(f"💾 {self.model_name} successfully saved to '{p}'.")
        return p

    def load_model(self, path: str | Path) -> 'ModelTrainer':
        
        '''
        It restores model state from serialized checkpoint.
        '''
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{p}'.")

        payload = joblib.load(p)
        self.model = payload["model"]
        self.params = payload.get("params", {})
        self.history = payload.get("history", {})
        self.best_params_ = payload.get("best_params_", {})
        self.best_score_ = payload.get("best_score_", 0.0)
        self.is_fitted = True
        print(f"📂 {self.model_name} successfully loaded from '{p}'.")

        return self