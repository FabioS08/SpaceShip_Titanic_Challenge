from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from .constants import DEFAULT_EXCLUDE_COLS
from typing import Any, Literal, cast
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np


class MissingValuesAnalyzer:

    '''
    A diagnostic utility to analyze missing data mechanisms: MCAR (Missing Completely at Random), 
    MAR (Missing at Random) and MNAR (Missing Not at Random).

    Parameters
    ----------
     df:pd.DataFrame
      The dataset to analyze.

     exclude_cols:list[str] | None
      Columns to exclude from statistical and predictive tests (e.g. high-cardinality IDs, text).
    '''

    DEFAULT_EXCLUDE_COLS = DEFAULT_EXCLUDE_COLS

    def __init__(self, df: pd.DataFrame, exclude_cols: list[str] | None = None):
        
        self.df = df.copy()
        self.exclude_cols = exclude_cols if exclude_cols is not None else self.DEFAULT_EXCLUDE_COLS.copy()


    def test_missingness_correlation(self, method: Literal["pearson", "kendall", "spearman"] = "pearson") -> pd.DataFrame:

        '''
        Computes pairwise correlations between missingness indicators (R_i and R_j).
        If data is MCAR, correlations across all feature pairs should be near 0.00.
        Strong correlations (> 0.3 - 0.5) indicate co-dependent missingness (MAR or structural).

        Parameters
        ----------
         method:str
          Correlation method: 'pearson', 'kendall' or 'spearman' [Default = "pearson"].

        Returns
        -------
         corr_matrix:pd.DataFrame
          Pairwise correlation matrix of missingness indicators for columns with missing values.
        '''

        missing_cols = [col for col in self.df.columns if self.df[col].isnull().sum() > 0]

        if not missing_cols:
            print("No missing values found in the dataset.")
            return pd.DataFrame()

        null_indicator_df = self.df[missing_cols].isnull().astype(int)
        corr_matrix = null_indicator_df.corr(method = method)

        return corr_matrix


    def plot_missingness_correlation(self, figsize: tuple[float, float] = (10, 8), cmap: str = "coolwarm", 
                                     method: Literal["pearson", "kendall", "spearman"] = "pearson") -> None:

        '''
        Plots a heatmap of the missingness indicator correlation matrix.

        Parameters
        ----------
         figsize:tuple[float, float]
          Size of the heatmap figure [Default = (10, 8)].

         cmap:str
          Colormap to use [Default = "coolwarm"].

         method:str
          Correlation method: 'pearson', 'kendall' or 'spearman' [Default = "pearson"].

        Returns
        -------
         None
        '''

        corr_matrix = self.test_missingness_correlation(method = method)

        if corr_matrix.empty:
            return

        plt.figure(figsize = figsize)
        sns.heatmap(corr_matrix, annot = True, fmt = ".2f", cmap = cmap, center = 0, vmin = -1, vmax = 1, linewidths = 0.5)
        plt.title("Missingness Co-occurrence Correlation Matrix")
        plt.tight_layout()
        plt.show()


    def _validate_target_column(self, target_col: str) -> pd.Series | None:

        '''
        It validates whether the target column exists and contains a valid mix of missing and observed values.

        Parameters
        ----------
         target_col:str
          The name of the target column to validate.

        Returns
        -------
         missing_mask:pd.Series | None
          Boolean series indicating missing values if valid or None if 0% or 100% missing.
        '''

        if target_col not in self.df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame.")

        missing_mask = self.df[target_col].isnull()
        missing_count = missing_mask.sum()

        if missing_count == 0:

            print(f"Column '{target_col}' has 0 missing values.")
            return None

        if missing_count == len(self.df):

            print(f"Column '{target_col}' is 100% missing.")
            return None

        return missing_mask


    def _test_numeric_feature_difference(self, col: str, missing_mask: pd.Series, alpha: float) -> dict[str, Any] | None:

        '''
        It runs Welch's two-sample t-test comparing a numeric feature when the target is missing vs observed.

        Parameters
        ----------
         col:str
          The name of the numeric feature.

         missing_mask:pd.Series
          Boolean series where True indicates the target feature is missing.

         alpha:float
          Significance threshold.

        Returns
        -------
         dict[str, Any] | None
          Dictionary with test results if valid sample sizes exist, otherwise None.
        '''

        vals_missing = self.df.loc[missing_mask, col].dropna()
        vals_observed = self.df.loc[~missing_mask, col].dropna()

        if len(vals_missing) < 2 or len(vals_observed) < 2:
            return None

        stat_val, p_val = stats.ttest_ind(vals_missing, vals_observed, equal_var = False)
        stat_val_f = float(cast(Any, stat_val))
        p_val_f = float(cast(Any, p_val))
        mean_diff = float(vals_missing.mean() - vals_observed.mean())

        return {
                    "Feature": col,
                    "Type": "Numeric",
                    "Test": "Welch t-test",
                    "Statistic": stat_val_f,
                    "p-value": p_val_f,
                    "Significant (p < alpha)": bool(p_val_f < alpha),
                    "Notes": f"Mean Missing: {vals_missing.mean():.2f} vs Obs: {vals_observed.mean():.2f} (Diff: {mean_diff:+.2f})"
                }


    def _test_categorical_feature_difference(self, col: str, missing_mask: pd.Series, alpha: float) -> dict[str, Any] | None:

        '''
        It runs a Chi-Square test of independence comparing a categorical feature when the target is missing vs observed.

        Parameters
        ----------
         col:str
          The name of the categorical feature.

         missing_mask:pd.Series
          Boolean series where True indicates the target feature is missing.

         alpha:float
          Significance threshold.

        Returns
        -------
         dict[str, Any] | None
          Dictionary with test results if contingency table is valid, otherwise None.
        '''

        contingency = pd.crosstab(missing_mask, self.df[col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None

        stat_val, p_val, _, _ = stats.chi2_contingency(contingency)
        stat_val_f = float(cast(Any, stat_val))
        p_val_f = float(cast(Any, p_val))

        return {
                    "Feature": col,
                    "Type": "Categorical",
                    "Test": "Chi-Square",
                    "Statistic": stat_val_f,
                    "p-value": p_val_f,
                    "Significant (p < alpha)": bool(p_val_f < alpha),
                    "Notes": f"{contingency.shape[1]} categories evaluated"
                }


    def test_statistical_differences(self, target_col: str, alpha: float = 0.05) -> pd.DataFrame:

        '''
        Tests whether the distribution of other observed features differs between rows where target_col is missing vs observed:

            - Welch's Two-Sample t-test for numeric features.
            - Chi-Square test of independence for categorical/boolean features.

        Under MCAR, p-values should be uniformly distributed and > alpha across features.
        Under MAR, several features will show statistically significant differences (p < alpha).

        Parameters
        ----------
         target_col:str
          The column whose missingness mechanism is being tested.

         alpha:float
          Significance threshold [Default = 0.05].

        Returns
        -------
         results_df:pd.DataFrame
          Summary table containing each tested feature, test type, statistic, p-value and significance.
        '''

        missing_mask = self._validate_target_column(target_col = target_col)

        if missing_mask is None:
            return pd.DataFrame()

        results: list[dict[str, Any]] = []

        candidate_cols = [c for c in self.df.columns 
                          if c != target_col 
                          and c not in self.exclude_cols 
                          and self.df[c].nunique() > 1]

        for col in candidate_cols:

            if pd.api.types.is_numeric_dtype(self.df[col]):
                res = self._test_numeric_feature_difference(col = col, missing_mask = missing_mask, alpha = alpha)

            else:
                res = self._test_categorical_feature_difference(col = col, missing_mask = missing_mask, alpha = alpha)

            if res is not None:
                results.append(res)

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            results_df = results_df.sort_values(by = "p-value", ascending = True).reset_index(drop = True)

        return results_df


    def _drop_uninformative_columns(self, X: pd.DataFrame, max_cardinality: int = 100) -> pd.DataFrame:

        '''
        It drops columns that cannot be effectively used as predictors:

            - Columns that are 100% missing.
            - High-cardinality categorical features (unique values > max_cardinality).

        Parameters
        ----------
         X:pd.DataFrame
          The predictor feature matrix.

         max_cardinality:int
          Maximum allowed unique categories for object features [Default = 100].

        Returns
        -------
         X:pd.DataFrame
          The feature matrix with uninformative columns dropped.
        '''

        cols_to_drop = [c for c in X.columns if X[c].isnull().sum() == len(X) or (X[c].dtype == object and X[c].nunique() > max_cardinality)]

        if cols_to_drop:
            return X.drop(columns = cols_to_drop)

        return X


    def _impute_predictors(self, X: pd.DataFrame) -> pd.DataFrame:

        '''
        It imputes missing values in the predictor matrix X so classifiers can train cleanly:
            
            - Median for numeric columns (or 0 if median is NaN).
            - Mode for categorical columns (or 'Missing' if no mode exists).

        Parameters
        ----------
         X:pd.DataFrame
          The feature matrix containing missing values.

        Returns
        -------
         X:pd.DataFrame
          The feature matrix with missing values imputed.
        '''

        X = X.copy()

        for col in X.columns:

            if pd.api.types.is_numeric_dtype(X[col]):

                median_val = X[col].median()
                X[col] = X[col].fillna(0 if pd.isna(median_val) else median_val)

            else:
                
                mode_series = X[col].mode()
                mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "Missing"
                X[col] = X[col].fillna(mode_val)

        return X


    def _select_model(self, model_type: str, random_state: int = 42) -> LogisticRegression | RandomForestClassifier:

        '''
        It initializes the classification model used for predicting missingness:
            
            - 'lr': LogisticRegression
            - 'rf': RandomForestClassifier

        Parameters
        ----------
         model_type:str
          Classifier type: 'rf' for RandomForestClassifier or 'lr' for LogisticRegression.

         random_state:int
          Random seed for reproducibility [Default = 42].

        Returns
        -------
         model:LogisticRegression | RandomForestClassifier
          The initialized scikit-learn classifier.
        '''

        if model_type.lower() == "lr":
            return LogisticRegression(max_iter = 1500, random_state = random_state)

        elif model_type.lower() == "rf":
            return RandomForestClassifier(n_estimators = 200, max_depth = 6, random_state = random_state, n_jobs = -1)

        else:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'rf' or 'lr'.")


    def _evaluate_predictability_cv(self, model: LogisticRegression | RandomForestClassifier, X: pd.DataFrame, y: pd.Series,
                                    cv: int, random_state: int) -> tuple[np.ndarray, float, float]:

        '''
        It evaluates the model performance via stratified k-fold cross-validation with ROC-AUC scoring.

        Parameters
        ----------
         model:LogisticRegression | RandomForestClassifier
          The estimator to evaluate.

         X:pd.DataFrame
          The predictor matrix.

         y:pd.Series
          The binary missingness target indicator.

         cv:int
          Number of cross-validation folds.

         random_state:int
          Random seed for reproducibility.

        Returns
        -------
         scores:np.ndarray
          Cross-validation ROC-AUC scores for each fold.

         mean_score:float
          Mean ROC-AUC across all folds.

         std_score:float
          Standard deviation of ROC-AUC across all folds.
        '''

        cv_strategy = StratifiedKFold(n_splits = cv, shuffle = True, random_state = random_state)
        scores = cross_val_score(model, X, y, cv = cv_strategy, scoring = "roc_auc")
        mean_score = float(scores.mean())
        std_score = float(scores.std())

        return scores, mean_score, std_score


    def _extract_feature_importances(self, model: LogisticRegression | RandomForestClassifier, X: pd.DataFrame, y: pd.Series,
                                     top_n: int = 10) -> pd.Series:

        '''
        It fits the model on the full data and extracts the top feature importances or coefficients.

        Parameters
        ----------
         model:LogisticRegression | RandomForestClassifier
          The classifier to fit.

         X:pd.DataFrame
          The predictor matrix.

         y:pd.Series
          The binary target indicator.

         top_n:int
          Number of top predictive features to retain [Default = 10].

        Returns
        -------
         feat_imp:pd.Series
          Top feature importances or absolute coefficients sorted in descending order.
        '''

        model.fit(X, y)

        if isinstance(model, RandomForestClassifier):
            return pd.Series(model.feature_importances_, index = X.columns).sort_values(ascending = False).head(top_n)

        elif isinstance(model, LogisticRegression):
            return pd.Series(np.abs(model.coef_[0]), index = X.columns).sort_values(ascending = False).head(top_n)

        return pd.Series(dtype = float)


    def _interpret_predictability(self, mean_score: float, threshold: float = 0.58) -> tuple[str, str]:

        '''
        It interprets the predictability ROC-AUC score to provide a missingness mechanism conclusion.

        Parameters
        ----------
         mean_score:float
          The mean cross-validated ROC-AUC score.

         threshold:float
          The decision boundary threshold between MCAR and MAR [Default = 0.58].

        Returns
        -------
         conclusion:str
          High-level conclusion (MCAR or MAR).

         reasoning:str
          Textual rationale for the conclusion.
        '''

        if mean_score < threshold:

            conclusion = "Consistent with MCAR (Missing Completely at Random)"
            reasoning = "Other observed features cannot predict missingness (ROC-AUC ≈ 0.50)."

        else:

            conclusion = "Evidence of MAR (Missing at Random)"
            reasoning = f"Observed features predict missingness with ROC-AUC = {mean_score:.3f} (> {threshold:.2f})."

        return conclusion, reasoning


    def test_predictability(self, target_col: str, model_type: str = "rf", cv: int = 5, random_state: int = 42) -> dict[str, Any]:

        '''
        Trains a classifier to predict whether target_col is missing (R = 1) or observed (R = 0)
        from all other features in the dataset.

        Interpretation:
            - ROC-AUC ≈ 0.50 (e.g. < 0.55): Missingness cannot be predicted from observed data -> Consistent with MCAR.
            - ROC-AUC >> 0.50 (e.g. >= 0.65): Missingness is correlated with other observed variables -> Evidence of MAR.

        Parameters
        ----------
         target_col:str
          The column whose missingness indicator is the target.

        model_type:str
          Classifier type: 'rf' for RandomForestClassifier or 'lr' for LogisticRegression [Default = 'rf'].

        cv:int
          Number of stratified cross-validation folds [Default = 5].

        random_state:int
          Random seed for reproducibility [Default = 42].

        Returns
        -------
         summary:dict[str, Any]
          Dictionary containing mean ROC-AUC, standard deviation, top predictive features, and interpretation.
        '''

        missing_mask = self._validate_target_column(target_col = target_col)

        if missing_mask is None:
            raise ValueError(f"Target column '{target_col}' has either 0 missing values or is 100% missing.")

        y = missing_mask.astype(int)

        # Prepare predictor matrix X
        X = self.df.drop(columns = [target_col] + self.exclude_cols, errors = "ignore").copy()

        # Drop columns with high cardinality or all missing
        X = self._drop_uninformative_columns(X = X, max_cardinality = len(X) // 2)

        # Impute missing values in X so classifier can train cleanly
        X = self._impute_predictors(X = X)

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first = True)

        # Select model
        model = self._select_model(model_type = model_type, random_state = random_state)

        # Cross-validation evaluation
        scores, mean_score, std_score = self._evaluate_predictability_cv(model = model, X = X, y = y, cv = cv, random_state = random_state)

        # Fit once on full data to inspect feature importances / coefficients
        feat_imp = self._extract_feature_importances(model = model, X = X, y = y, top_n = 10)

        # Interpret predictability outcome
        conclusion, reasoning = self._interpret_predictability(mean_score = mean_score)

        return {
                    "target_col": target_col,
                    "mean_roc_auc": round(mean_score, 4),
                    "std_roc_auc": round(std_score, 4),
                    "cv_scores": [round(float(s), 4) for s in scores],
                    "conclusion": conclusion,
                    "reasoning": reasoning,
                    "top_features": feat_imp
                }


    def _diagnose_missingness_correlation(self, target_col: str) -> pd.Series:

        '''
        It evaluates and prints the strongest missingness co-occurrence correlation for target_col.

        Parameters
        ----------
         target_col:str
          The column whose missingness correlations are evaluated.

        Returns
        -------
         col_corrs:pd.Series
          Pairwise correlation values between target_col missingness and all other features.
        '''

        corr_matrix = self.test_missingness_correlation()

        if not corr_matrix.empty and target_col in corr_matrix.columns:

            col_corrs = corr_matrix[target_col].drop(index = target_col)
            max_corr_feat = col_corrs.abs().idxmax()
            max_corr_val = col_corrs[max_corr_feat]

            print(f"\n[Test 1] Missingness Co-occurrence Correlation:")
            print(f" - Strongest correlation with: '{max_corr_feat}' (r = {max_corr_val:+.3f})")

            if abs(max_corr_val) < 0.10:
                print(" - Result: Co-occurrence correlations are near 0.00 -> Consistent with MCAR.")
            
            else:
                print(f" - Result: Noticeable correlation with '{max_corr_feat}' -> Potential joint missingness.")

        else:

            col_corrs = pd.Series(dtype = float)

        return col_corrs


    def _diagnose_statistical_differences(self, target_col: str, alpha: float) -> tuple[pd.DataFrame, int]:

        '''
        It runs statistical distribution comparisons (t-test / chi-square) and prints a diagnostic report.

        Parameters
        ----------
         target_col:str
          The column whose missingness is evaluated.

         alpha:float
          Significance threshold for statistical tests.

        Returns
        -------
         stat_df:pd.DataFrame
          Summary table containing tested features, statistics, p-values, and significance.

         sig_count:int
          Number of features with statistically significant distribution differences.
        '''

        stat_df = self.test_statistical_differences(target_col = target_col, alpha = alpha)
        sig_count = int(stat_df["Significant (p < alpha)"].sum()) if not stat_df.empty else 0
        total_tested = len(stat_df)

        print(f"\n[Test 2] Statistical Distribution Comparison (alpha = {alpha}):")
        print(f" - Significant features found: {sig_count} out of {total_tested} tested.")

        if sig_count == 0:
            print(" - Result: No feature distributions differ significantly when missing vs. observed -> Consistent with MCAR.")

        else:

            print(f" - Result: {sig_count} feature(s) showed statistically significant differences -> Evidence of MAR.")

            if not stat_df.empty:
                
                print("\n   Top Significant Features:")
                print(stat_df[stat_df["Significant (p < alpha)"]][["Feature", "Test", "p-value", "Notes"]].to_string(index = False))

        return stat_df, sig_count


    def _diagnose_predictability(self, target_col: str, cv: int = 5) -> dict[str, Any]:

        '''
        It runs the machine learning predictability test and prints a diagnostic report.

        Parameters
        ----------
         target_col:str
          The column whose missingness is predicted.

         cv:int
          Number of stratified cross-validation folds [Default = 5].

        Returns
        -------
         pred_res:dict[str, Any]
          Dictionary containing mean ROC-AUC, standard deviation, top features, and interpretation.
        '''

        pred_res = self.test_predictability(target_col = target_col, cv = cv)

        print(f"\n[Test 3] Machine Learning Predictability Test:")
        print(f" - {cv}-Fold Cross-Validated ROC-AUC: {pred_res['mean_roc_auc']:.3f} (+/- {pred_res['std_roc_auc']:.3f})")
        print(f" - Interpretation: {pred_res['reasoning']}")

        return pred_res


    def _determine_overall_verdict(self, sig_count: int, mean_roc_auc: float, threshold: float = 0.58) -> tuple[str, str]:

        '''
        It synthesizes statistical distribution differences and ML predictability into an overall verdict.

        Parameters
        ----------
         sig_count:int
          Number of features with statistically significant distribution differences.

         mean_roc_auc:float
          Mean cross-validated ROC-AUC score from the predictability test.

         threshold:float
          Decision threshold for the predictability test [Default = 0.58].

        Returns
        -------
         verdict:str
          High-level verdict: 'MCAR (Missing Completely at Random)' or 'MAR (Missing at Random)'.

         explanation:str
          Actionable recommendation on appropriate imputation strategies.
        '''

        if sig_count == 0 and mean_roc_auc < threshold:

            verdict = "MCAR (Missing Completely at Random)"
            explanation = "Missingness is independent of all other observed features. \nStandard imputation (median/mode) or dropping rows will not introduce \nsystematic bias."

        else:

            verdict = "MAR (Missing at Random)"
            explanation = "Missingness is statistically correlated with other observed features. \nConditional imputation (e.g. KNN, MICE, or GroupBy) is recommended \nover simple mean imputation."

        return verdict, explanation


    def diagnose_missing_mechanism(self, target_col: str, alpha: float = 0.05, cv: int = 5) -> dict[str, Any]:

        '''
        Executes all three diagnostic methods for a given feature and prints a formatted summary report.

        Parameters
        ----------
         target_col:str
          The feature to analyze.

         alpha:float
          Significance threshold for statistical tests [Default = 0.05].

         cv:int
          Number of cross-validation folds for the predictability test [Default = 5].

        Returns
        -------
         diagnosis:dict[str, Any]
          Dictionary containing detailed outputs from all 3 tests.
        '''

        print("=" * 80)
        print(f" 🧪 MISSING DATA MECHANISM DIAGNOSIS FOR: '{target_col.upper()}' ".center(80))
        print("=" * 80)

        # 1. Missingness Correlation Check
        col_corrs = self._diagnose_missingness_correlation(target_col = target_col)

        # 2. Statistical Differences Test
        stat_df, sig_count = self._diagnose_statistical_differences(target_col = target_col, alpha = alpha)

        # 3. Machine Learning Predictability Test
        pred_res = self._diagnose_predictability(target_col = target_col, cv = cv)

        # Overall Verdict
        verdict, explanation = self._determine_overall_verdict(sig_count = sig_count, mean_roc_auc = pred_res["mean_roc_auc"])

        print("\n" + "-" * 80)
        print(f" 🎯 OVERALL VERDICT: {verdict}".center(80))
        print("-" * 80)
        print(f"Summary: {explanation}\n" + "=" * 80 + "\n")

        return {
                    "target_col": target_col,
                    "verdict": verdict,
                    "correlations": col_corrs,
                    "statistical_differences": stat_df,
                    "predictability": pred_res
                }