from dataclasses import dataclass
import pandas as pd


@dataclass
class DataSplit:

    '''
    It is the standardized container for partitioned datasets, ensuring that all the candidate
    models are evaluated on the exact same test holdout split.
    '''

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_competition_test: pd.DataFrame | None = None

    def summary(self) -> str:

        '''
        It returns a formatted overview of sample sizes and positive target ratios across splits.
        '''

        s_tr = f"Train Set: {len(self.X_train):,} samples (% Positive Class: {self.y_train.mean() * 100:.1f}%)"
        s_val = f"Validation Set: {len(self.X_val):,} samples (% Positive Class: {self.y_val.mean() * 100:.1f}%)"
        s_te = f"Test Set (Shared): {len(self.X_test):,} samples (% Positive Class: {self.y_test.mean() * 100:.1f}%)"
        comp = f"Competition Test: {len(self.X_competition_test):,} samples" if self.X_competition_test is not None else "Competition Test: None"

        return f"[DataSplit] {s_tr} | {s_val} | {s_te} | {comp}"