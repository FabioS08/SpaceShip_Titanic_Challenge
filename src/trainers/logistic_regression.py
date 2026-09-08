from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from .base import ModelTrainer
from typing import Any
import numpy as np


class LogisticRegressionTrainer(ModelTrainer):

    '''
    Specialized trainer for LogisticRegression.
    '''

    model_preset_key: str = "logistic_regression"
    default_name: str = "LogisticRegression"


    def _init_model(self, params: dict[str, Any]) -> LogisticRegression:

        p = params.copy()

        if "random_state" not in p:
            p["random_state"] = self.random_state

        if "max_iter" not in p:
            p["max_iter"] = 1000

        return LogisticRegression(**p)


    def fit(self, compute_sample_curve: bool = True, **kwargs: Any) -> 'LogisticRegressionTrainer':
      
        '''
        It fits LogisticRegression and optionally computes sample-size learning curves.
        '''

        self.model.fit(self.X_train, self.y_train, **kwargs)
        self.is_fitted = True

        if compute_sample_curve:

            train_sizes, train_scores, val_scores, _, _ = learning_curve(
                                                                            self.model,
                                                                            self.X_train,
                                                                            self.y_train,
                                                                            train_sizes = np.linspace(0.2, 1.0, 5),
                                                                            cv = 3,
                                                                            scoring = "neg_log_loss",
                                                                            random_state = self.random_state,
                                                                            return_times = True,
                                                                        )
            self.history["train_loss"] = [float(-x) for x in np.mean(train_scores, axis = 1)]
            self.history["val_loss"] = [float(-x) for x in np.mean(val_scores, axis = 1)]
            self.history["iterations"] = [int(s) for s in train_sizes]

        return self