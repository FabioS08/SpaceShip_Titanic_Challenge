from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from .base import ModelTrainer
from typing import Any
import numpy as np


class RandomForestTrainer(ModelTrainer):

    '''
    Specialized trainer for RandomForestClassifier.
    '''

    model_preset_key: str = "random_forest"
    default_name: str = "RandomForest"


    def _init_model(self, params: dict[str, Any]) -> RandomForestClassifier:

        p = params.copy()

        if "random_state" not in p:
            p["random_state"] = self.random_state

        if "n_jobs" not in p:
            p["n_jobs"] = -1

        return RandomForestClassifier(**p)


    def fit(self, compute_tree_curve: bool = True, step: int = 15, **kwargs: Any) -> RandomForestTrainer:
        
        '''
        It fits the random forest and evaluates the tree-by-tree ensemble convergence curve.

        Parameters
        ----------
        compute_tree_curve: bool
            Whether to compute the tree-by-tree convergence curve [Default = True].

        step: int
            Step size in number of trees between evaluation checkpoints [Default = 15].

        **kwargs: Any
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        RandomForestTrainer
            The fitted trainer instance.
        '''

        self.model.fit(self.X_train, self.y_train, **kwargs)
        self.is_fitted = True

        if compute_tree_curve and hasattr(self.model, "estimators_") and len(self.model.estimators_) > 1:
            self._compute_tree_curve(step = step)

        return self


    def _compute_tree_curve(self, step: int = 15) -> None:

        '''
        It computes the tree-by-tree ensemble convergence curve across tree count without retraining.

        Parameters
        ----------
        step: int
            Step size in number of trees between evaluation checkpoints [Default = 15].
        '''

        steps = self._get_step_checkpoints(step = step)
        train_preds, val_preds = self._precompute_tree_probabilities()
        self._evaluate_ensemble_steps(steps = steps, train_preds = train_preds, val_preds = val_preds)


    def _get_step_checkpoints(self, step: int) -> list[int]:

        '''
        It calculates the tree counts at which the ensemble is evaluated.

        Parameters
        ----------
        step: int
            Step size in number of trees between evaluation checkpoints.

        Returns
        -------
        list[int]
            List of tree counts to evaluate.
        '''

        n_trees = len(self.model.estimators_)
        steps = list(range(max(5, step), n_trees + 1, step))

        if n_trees not in steps:
            steps.append(n_trees)

        return steps


    def _precompute_tree_probabilities(self) -> tuple[np.ndarray, np.ndarray | None]:

        '''
        It precomputes class-1 probability predictions from each tree estimator in the ensemble.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            Tuple containing train predictions and optional validation predictions arrays.
        '''

        train_preds = np.array([e.predict_proba(self.X_train)[:, 1] for e in self.model.estimators_])
        val_preds = np.array([e.predict_proba(self.X_val)[:, 1] for e in self.model.estimators_]) if not self.X_val.empty else None

        return train_preds, val_preds


    def _evaluate_ensemble_steps(self, steps: list[int], train_preds: np.ndarray, val_preds: np.ndarray | None) -> None:

        '''
        It evaluates cumulative ensemble predictions across step checkpoints and records metrics into self.history.

        Parameters
        ----------
        steps: list[int]
            List of tree count checkpoints.

        train_preds: np.ndarray
            Precomputed tree-level probability predictions on the training set.

        val_preds: np.ndarray | None
            Precomputed tree-level probability predictions on the validation set, or None.
        '''

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        has_val = val_preds is not None and not self.X_val.empty

        for k in steps:

            p_tr = np.mean(train_preds[:k], axis = 0)
            train_losses.append(float(log_loss(self.y_train, p_tr)))
            train_accs.append(float(accuracy_score(self.y_train, p_tr > 0.5)))

            if has_val and val_preds is not None:
                p_val = np.mean(val_preds[:k], axis = 0)
                val_losses.append(float(log_loss(self.y_val, p_val)))
                val_accs.append(float(accuracy_score(self.y_val, p_val > 0.5)))

        self.history["train_loss"] = train_losses
        self.history["val_loss"] = val_losses
        self.history["train_metric"] = train_accs
        self.history["val_metric"] = val_accs
        self.history["iterations"] = steps