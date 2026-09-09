from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from .base import ModelTrainer
from typing import Any


class GradientBoostingTrainer(ModelTrainer):

    '''
    Specialized trainer for GradientBoostingClassifier.
    '''

    model_preset_key: str = "gradient_boosting"
    default_name: str = "GradientBoosting"


    def _init_model(self, params: dict[str, Any]) -> GradientBoostingClassifier:
        
        '''
        It initializes the GradientBoostingClassifier.

        Parameters
        ----------
        params: dict[str, Any]
            Hyperparameters for the model.
        
        Returns
        -------
        GradientBoostingClassifier
            Initialized model.
        '''

        p = params.copy()

        if "random_state" not in p:
            p["random_state"] = self.random_state
        
        return GradientBoostingClassifier(**p)


    def fit(self, **kwargs: Any) -> GradientBoostingTrainer:
        
        '''
        It fits the GradientBoostingClassifier and evaluates staged loss and metric trajectories.

        Parameters
        ----------
        **kwargs: Any
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        GradientBoostingTrainer
            The fitted trainer instance.
        '''

        self.model.fit(self.X_train, self.y_train, **kwargs)
        self.is_fitted = True

        self._extract_history()

        return self


    def _extract_history(self) -> None:
        
        '''
        It extracts staged training deviance and validation metrics into self.history.
        '''

        self._extract_training_history()
        self._extract_validation_history()


    def _extract_training_history(self) -> None:
        
        '''
        It extracts staged training deviance into self.history.
        '''

        if not hasattr(self.model, "train_score_"):
            return

        self.history["train_loss"] = [float(x) for x in self.model.train_score_]
        n_stages = len(self.model.train_score_)
        self.history["iterations"] = list(range(1, n_stages + 1))


    def _extract_validation_history(self) -> None:
        
        '''
        It evaluates and extracts staged validation loss and accuracy trajectories into self.history.
        '''

        if self.X_val.empty or not hasattr(self.model, "staged_predict_proba"):
            return

        y_val_np = self.y_val.to_numpy()
        val_losses: list[float] = []
        val_accs: list[float] = []

        for staged_probs in self.model.staged_predict_proba(self.X_val):

            p_val = staged_probs[:, 1]
            val_losses.append(float(log_loss(y_val_np, p_val)))
            val_accs.append(float(accuracy_score(y_val_np, p_val > 0.5)))

        self.history["val_loss"] = val_losses
        self.history["val_metric"] = val_accs