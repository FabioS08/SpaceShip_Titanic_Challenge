from xgboost import XGBClassifier
from .base import ModelTrainer
from typing import Any


class XGBoostTrainer(ModelTrainer):

    '''
    Specialized trainer for XGBoost (XGBClassifier).
    '''

    model_preset_key: str = "xgboost"
    default_name: str = "XGBoost"


    def _init_model(self, params: dict[str, Any]) -> XGBClassifier:

        p = params.copy()

        if "random_state" not in p:
            p["random_state"] = self.random_state

        if "eval_metric" not in p:
            p["eval_metric"] = "logloss"

        if "n_jobs" not in p:
            p["n_jobs"] = 1

        return XGBClassifier(**p)


    def fit(self, verbose: bool = False, **kwargs: Any) -> 'XGBoostTrainer':

        '''
        It fits XGBoostClassifier with train and validation evaluation monitoring.

        Parameters
        ----------
        verbose: bool
            Whether to print evaluation messages during boosting [Default = False].

        **kwargs: Any
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        XGBoostTrainer
            Fitted trainer instance.
        '''
        fit_kwargs = self._prepare_fit_kwargs(verbose = verbose, **kwargs)

        self.model.fit(self.X_train, self.y_train, **fit_kwargs)
        self.is_fitted = True

        self._extract_history()

        return self


    def _prepare_fit_kwargs(self, verbose: bool = False, **kwargs: Any) -> dict[str, Any]:

        '''
        It prepares keyword arguments for XGBClassifier.fit, including the evaluation sets.

        Parameters
        ----------
        verbose: bool
            Whether to print evaluation messages during boosting [Default = False].

        **kwargs: Any
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        dict[str, Any]
            Prepared keyword arguments dictionary for fitting the model.
        '''
        eval_set = []

        if not self.X_train.empty:
            eval_set.append((self.X_train, self.y_train))

        if not self.X_val.empty:
            eval_set.append((self.X_val, self.y_val))

        fit_kwargs: dict[str, Any] = {"verbose": verbose}

        if eval_set:
            fit_kwargs["eval_set"] = eval_set

        fit_kwargs.update(kwargs)

        return fit_kwargs


    def _extract_history(self, eval_metric: str = "logloss") -> None:

        '''
        It extracts training and validation evaluation curves from evals_result into self.history.

        Parameters
        ----------
        eval_metric: str
            Evaluation metric name to retrieve from evals_result [Default = "logloss"].
        '''

        evals_result = getattr(self.model, "evals_result", lambda: {})()

        if not evals_result:
            return

        v0 = evals_result.get("validation_0", {})
        metric = eval_metric if eval_metric in v0 else ("logloss" if "logloss" in v0 else (next(iter(v0)) if v0 else None))

        if metric is None:
            return

        train_loss = evals_result.get("validation_0", {}).get(metric, [])
        val_loss = evals_result.get("validation_1", {}).get(metric, [])

        if train_loss:
            self.history["train_loss"] = [float(x) for x in train_loss]

        if val_loss:
            self.history["val_loss"] = [float(x) for x in val_loss]

        n_iters = len(train_loss) or len(val_loss)

        if n_iters > 0:
            self.history["iterations"] = list(range(1, n_iters + 1))