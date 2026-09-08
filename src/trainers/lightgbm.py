from __future__ import annotations

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from .base import ModelTrainer
from typing import Any
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"


class LightGBMTrainer(ModelTrainer):

    '''
    Specialized trainer for LightGBM (LGBMClassifier).
    '''

    model_preset_key: str = "lightgbm"
    default_name: str = "LightGBM"


    def _init_model(self, params: dict[str, Any]) -> LGBMClassifier:

        '''
        It initializes the LGBMClassifier.

        Parameters
        ----------
        params: dict[str, Any]
            Hyperparameters for the model.

        Returns
        -------
        LGBMClassifier
            Initialized model.
        '''
        p = params.copy()

        if "random_state" not in p:
            p["random_state"] = self.random_state

        if "n_jobs" not in p:
            p["n_jobs"] = 1

        if "verbosity" not in p:
            p["verbosity"] = -1

        return LGBMClassifier(**p)


    def fit(self, early_stopping_rounds: int | None = 30, eval_metric: str = "binary_logloss", verbose: bool = False, **kwargs: Any) -> LightGBMTrainer:

        '''
        It trains LGBMClassifier with evaluation sets and extracts iteration history.

        Parameters
        ----------
        early_stopping_rounds: int | None
            Number of rounds for early stopping.

        eval_metric: str
            Evaluation metric to track during training.

        verbose: bool
            Whether to print training progress.

        **kwargs: Any
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        LightGBMTrainer
            The fitted trainer instance.
        '''

        fit_kwargs = self._prepare_fit_kwargs(early_stopping_rounds = early_stopping_rounds, eval_metric = eval_metric, verbose = verbose, **kwargs)

        self.model.fit(self.X_train, self.y_train, **fit_kwargs)
        self.is_fitted = True

        self._extract_history(eval_metric = eval_metric)

        return self


    def _prepare_callbacks(self, early_stopping_rounds: int | None, verbose: bool) -> list[Any]:

        '''
        It prepares early stopping and logging callbacks for LGBMClassifier.

        Parameters
        ----------
        early_stopping_rounds: int | None
            Number of rounds for early stopping.

        verbose: bool
            Whether to print training evaluation progress.

        Returns
        -------
        list[Any]
            List of LightGBM callbacks.
        '''

        callbacks = []

        if early_stopping_rounds is not None and not self.X_val.empty:
            callbacks.append(early_stopping(stopping_rounds = early_stopping_rounds, verbose = verbose))

        if verbose:
            callbacks.append(log_evaluation(period = 50))

        return callbacks


    def _prepare_fit_kwargs(self, early_stopping_rounds: int | None, eval_metric: str, verbose: bool, **kwargs: Any) -> dict[str, Any]:

        '''
        It prepares keyword arguments for LGBMClassifier.fit.

        Parameters
        ----------
        early_stopping_rounds: int | None
            Number of rounds for early stopping.

        eval_metric: str
            Evaluation metric name for evaluation sets.

        verbose: bool
            Whether to print training evaluation progress.

        **kwargs: Any
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        dict[str, Any]
            Prepared keyword arguments dictionary for fitting the model.
        '''

        callbacks = self._prepare_callbacks(early_stopping_rounds = early_stopping_rounds, verbose = verbose)

        eval_set = []
        eval_names = []

        if not self.X_train.empty:

            eval_set.append((self.X_train, self.y_train))
            eval_names.append("train")

        if not self.X_val.empty:

            eval_set.append((self.X_val, self.y_val))
            eval_names.append("val")

        fit_kwargs: dict[str, Any] = {}

        if eval_set:

            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["eval_names"] = eval_names
            fit_kwargs["eval_metric"] = eval_metric

        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        fit_kwargs.update(kwargs)

        return fit_kwargs


    def _extract_history(self, eval_metric: str) -> None:

        '''
        It extracts learning curves from evals_result_ into self.history.

        Parameters
        ----------
        eval_metric: str
            Evaluation metric name to retrieve from evals_result_.
        '''

        evals_result = getattr(self.model, "evals_result_", {})

        if not evals_result:
            return

        train_losses = evals_result.get("train", {}).get(eval_metric, [])
        val_losses = evals_result.get("val", {}).get(eval_metric, [])

        if train_losses:
            self.history["train_loss"] = [float(x) for x in train_losses]

        if val_losses:
            self.history["val_loss"] = [float(x) for x in val_losses]

        n_iters = len(train_losses) or len(val_losses)
        self.history["iterations"] = list(range(1, n_iters + 1))