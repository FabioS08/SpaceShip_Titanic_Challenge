from __future__ import annotations

from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from .base import ModelTrainer
from typing import Any


class CatBoostTrainer(ModelTrainer):

    '''
    Specialized trainer for CatBoostClassifier.
    '''

    model_preset_key: str = "catboost"
    default_name: str = "CatBoost"


    def _init_model(self, params: dict[str, Any]) -> CatBoostClassifier:
        
        '''
        Initializes the CatBoostClassifier model.

        Parameters
        ----------
        params: dict[str, Any]
            Model parameters.
        '''
        p = params.copy()
        
        if "random_seed" not in p:
            p["random_seed"] = self.random_state
        
        if "verbose" not in p:
            p["verbose"] = False
        
        return CatBoostClassifier(**p)


    def fit(self, early_stopping_rounds: int | None = 50, eval_metric: str = "Logloss", verbose: bool = False, **kwargs: Any) -> CatBoostTrainer:
        
        '''
        It trains CatBoostClassifier on self.X_train, logging evaluation curves on self.X_val.
        
        Parameters
        ----------
        early_stopping_rounds: int | None
            Number of rounds for early stopping.

        eval_metric: str
            Evaluation metric to use.
        
        verbose: bool
            Whether to print verbose output.
        
        **kwargs: Any
            Additional keyword arguments to pass to the fit method.
        '''

        fit_kwargs = self._prepare_fit_kwargs(early_stopping_rounds = early_stopping_rounds, verbose = verbose, **kwargs)

        self.model.fit(self.X_train, self.y_train, **fit_kwargs)
        self.is_fitted = True

        self._extract_history(eval_metric = eval_metric)

        return self


    def _prepare_fit_kwargs(self, early_stopping_rounds: int | None, verbose: bool, **kwargs: Any) -> dict[str, Any]:
        
        '''
        It prepares keyword arguments for CatBoostClassifier.fit.

        Parameters
        ----------
        early_stopping_rounds: int | None
            Number of rounds for early stopping.

        verbose: bool
            Whether to print verbose output.

        **kwargs: Any
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        dict[str, Any]
            Prepared keyword arguments dictionary for fitting the model.
        '''
        eval_set = (self.X_val, self.y_val) if not self.X_val.empty else None

        fit_kwargs: dict[str, Any] = {"verbose": verbose}

        if eval_set is not None:
        
            fit_kwargs["eval_set"] = eval_set
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        
        fit_kwargs.update(kwargs)

        return fit_kwargs


    def _extract_history(self, eval_metric: str = "Logloss") -> None:
        
        '''
        It extracts native learning curves and validation metrics into self.history.

        Parameters
        ----------
        eval_metric: str
            Evaluation metric name to retrieve from evals_result.
        '''

        evals_result = self.model.get_evals_result()
        
        if not evals_result:
            return

        learn_loss = evals_result.get("learn", {}).get(eval_metric, evals_result.get("learn", {}).get("Logloss", []))
        val_loss = evals_result.get("validation", {}).get(eval_metric, evals_result.get("validation", {}).get("Logloss", []))
        self.history["train_loss"] = [float(x) for x in learn_loss]
        self.history["val_loss"] = [float(x) for x in val_loss]
        self.history["iterations"] = list(range(1, len(learn_loss) + 1))

        if not self.X_val.empty:
        
            val_probs = self.predict_proba(self.X_val)
            final_val_acc = accuracy_score(self.y_val.to_numpy(), val_probs > 0.5)
            self.history["val_metric"] = [float(final_val_acc)] * len(val_loss)