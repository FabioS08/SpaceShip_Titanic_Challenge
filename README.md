# Spaceship Titanic Challenge

> **A production-grade tabular Machine Learning framework and heterogeneous multi-model ensemble for the Kaggle Spaceship Titanic competition.**

![Status](https://img.shields.io/badge/Status-Completed-success)
![Type](https://img.shields.io/badge/Type-Kaggle_Competition-blue)
![Public LB Score](https://img.shields.io/badge/Kaggle_Public_LB-0.80664-brightgreen)
![Benchmark Matrix](https://img.shields.io/badge/Benchmark-7_Models_×_5_Presets-orange)
![Tech](https://img.shields.io/badge/Tech-PyTorch_|_CatBoost_|_LightGBM_|_XGBoost_|_Scikit--Learn-informational)
![License](https://img.shields.io/badge/License-GPL_v3.0-lightgrey)

---
<br></br>
## ⛓️ Table of Contents

1. [About The Project](#-about-the-project)
2. [Repository Structure](#-repository-structure)
3. [Getting Started](#-getting-started)
4. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
5. [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
6. [Model Benchmarking & Experimental Matrix](#-model-benchmarking--experimental-matrix)
7. [Heterogeneous Multi-Model Ensemble](#-heterogeneous-multi-model-ensemble-models-assemble)
8. [Code Usage Examples](#-code-usage-examples)
9. [A Note from the Cockpit](#-a-note-from-the-cockpit-and-a-challenge)
10. [References and Citation](#-references-and-citation)

<br></br>
## 📖 About The Project

The **Spaceship Titanic** is an interstellar passenger liner that collided with a spacetime anomaly while en route to three newly habitable exoplanets (*TRAPPIST-1e*, *55 Cancri e* and *PSO J318.5-22*). Nearly half of the 13,000+ passengers on board were transported into an alternate dimension. The objective of this competition is to predict which passengers were transported (`Transported = True / False`) using demographic, spatial and amenity expenditure records.

This repository implements a **production-grade, modular and leak-free Machine Learning framework**:

- **Modular Architecture (`src/`)**: Cleanly decoupled modules for data ingestion, missingness diagnostics, dataset preset building, model training, cross-validation and consensus ensembling.

- **Statistical Missingness Diagnostics (`MissingValuesAnalyzer`)**: Formal mechanism analysis (MCAR, MAR, MNAR) using hypothesis testing (Mann-Whitney U, Chi-square) and predictive ML classifiers.

- **Domain-Driven Hybrid Imputation**: Implemented deterministic domain rules (CryoSleep non-spending invariants, planetary cabin deck bounds, passenger group consistency) combined with KNN and simple imputation, retaining 100% of data samples without target leakage.

- **Configurable Dataset Presets**: 5 preconfigured feature representations (`baseline`, `drop_missing`, `aggregated_spending_only`, `raw_spending_only`, `linear_optimized`) tailored to distinct algorithm classes.

- **Systematic 7-Model × 5-Preset Cross-Benchmark**: Evaluates 35 distinct experimental pairs on a strictly isolated, identical test holdout with side-by-side performance heatmaps and sensitivity analyses.

- **Heterogeneous Multi-Model Ensemble**: Strategic majority voting across 5 models trained on full datasets with diverse preprocessing representations, achieving a **Kaggle Public Leaderboard score of 0.80664** 🏆.

<br></br>
## 🗼 Repository Structure

<details>
<summary><b>Visualize Repository Structure</b></summary>
<br>

```text
SpaceShip_Titanic_Challenge/
├── 1 - SpaceShip_EDA.ipynb                     # Interactive Exploratory Data Analysis & visual diagnostics
├── 2 - SpaceShip_Data_Processing.ipynb         # Statistical missingness diagnosis, feature engineering & presets
├── 3 - SpaceShip_Training_&_Evaluation.ipynb   # 35-experiment benchmark matrix, heatmaps & full ensemble
├── dataset/                                    # Competition datasets
│   ├── train.csv                               # Training set (8,693 passenger records)
│   ├── test.csv                                # Competition test set (4,277 passenger records)
│   └── sample_submission.csv                   # Kaggle submission format template
├── submissions/                                # Versioned prediction outputs
│   └── submission.csv                          # Submission file (Come on, are you really looking for a solution already?)
│
├── src/                                        # Core source package
│   ├── __init__.py                             # Package exports
│   ├── dataset.py                              # SpaceShipDataset pipeline & preset factory
│   ├── visualizer.py                           # DatasetVisualizer (distributions, spatial heatmaps, 3D PCA/t-SNE/UMAP)
│   ├── trainers/                               # Specialized model trainer classes
│   │   ├── __init__.py
│   │   ├── base.py                             # ModelTrainer abstract base class (training, holdout, CV, tuning)
│   │   ├── catboost.py                         # CatBoostTrainer (symmetric oblivious decision trees)
│   │   ├── lightgbm.py                         # LightGBMTrainer (leaf-wise gradient boosting)
│   │   ├── xgboost.py                          # XGBoostTrainer (depth-wise gradient boosting)
│   │   ├── random_forest.py                    # RandomForestTrainer (bagging ensemble)
│   │   ├── gradient_boosting.py                # GradientBoostingTrainer (staged deviance boosting)
│   │   ├── logistic_regression.py              # LogisticRegressionTrainer (regularized linear baseline)
│   │   ├── neural_net.py                       # NeuralNetTrainer (PyTorch MLP: Linear + BatchNorm + SiLU + Dropout)
│   │   └── model_comparison.py                 # compare_trainers() shared test holdout leaderboard
│   └── utils/                                  # Utilities, schema definitions & configuration
│       ├── __init__.py
│       ├── constants.py                        # Schema constants, feature groupings & deck hierarchy
│       ├── data_split.py                       # DataSplit dataclass ensuring identical test holdouts
│       ├── dataset_presets.py                  # DATASET_PRESETS configuration dictionary (5 presets)
│       ├── missing_values.py                   # MissingValuesAnalyzer (statistical & predictive mechanism tests)
│       ├── model_presets.py                    # MODEL_PRESETS tuning grids & search distributions
│       └── utils.py                            # Benchmark heatmaps, sensitivity plots, consensus & ensemble runner
├── environment.yaml                            # Conda environment specification
├── requirements.txt                            # Pip dependencies file
├── LICENSE                                     # GNU General Public License v3.0
└── README.md                                   # Project documentation
```

</details>

<br></br>
## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/FabioS08/SpaceShip_Titanic_Challenge.git
cd SpaceShip_Titanic_Challenge
```

### 2. Environment Setup

Create and activate the conda environment using the provided [`environment.yaml`](environment.yaml):

```bash
conda env create -f environment.yaml
conda activate SpaceShip
```

Alternatively, using `pip` with an existing Python 3.10+ environment:
```bash
pip install -r requirements.txt
```

> **Note for Apple Silicon (macOS ARM):**  
> If running LightGBM alongside PyTorch, ensure `libomp` is installed via Homebrew (`brew install libomp`) and export:
> ```bash
> export KMP_DUPLICATE_LIB_OK=True
> export OMP_NUM_THREADS=1
> ```

<br></br>
## 🔍 Exploratory Data Analysis (EDA)

Exploratory analysis is conducted interactively in [`1 - SpaceShip_EDA.ipynb`](1%20-%20SpaceShip_EDA.ipynb) using [`DatasetVisualizer`](src/visualizer.py). Key domain insights include:

1. **Balanced Target Distribution**: The training set contains 8,693 passengers with an almost even class balance (50.36% `Transported = True` vs. 49.64% `False`).

2. **CryoSleep Invariant (Primary Signal)**: CryoSleep is the single most predictive individual feature. Approximately **81.8%** of passengers in CryoSleep were transported. Furthermore, passengers in CryoSleep have strictly **$0 expenditure** across all amenities.

3. **Planetary Socioeconomic Stratification**:

   - **Europa**: Wealthiest cohort occupying luxury upper decks (`A`, `B`, `C`, `T`), exhibiting heavy amenity spending and high transport rates (~65.9%).

   - **Earth**: Economy cohort concentrated on deck `G` with low spending and lower transport rates (~42.4%).

   - **Mars**: Middle cohort predominantly on decks `D`, `E`, and `F` with moderate spending and average transport rates (~52.3%).

4. **Spatial Partitioning**: Passenger cabins (`Deck/Num/Side`) show clear spatial survival boundaries across decks and between Port (`P`) and Starboard (`S`).

5. **3D Interactive Projections**: Unsupervised manifold and dimensionality reduction via **PCA**, **t-SNE** and **UMAP** reveals structured passenger clustering along the CryoSleep and spending axes.

<br></br>
## ⚙️ Data Preprocessing & Feature Engineering

Implemented in [`src/dataset.py`](src/dataset.py) and validated in [`2 - SpaceShip_Data_Processing.ipynb`](2%20-%20SpaceShip_Data_Processing.ipynb).

### 1. Statistical Missingness Mechanism Diagnosis

Using [`MissingValuesAnalyzer`](src/utils/missing_values.py), each feature's missingness is classified through three rigorous tests:

- **Correlation of Missingness**: Pairwise correlation across missing indicators.

- **Statistical Difference Testing**: Mann-Whitney U test for continuous features and Chi-square test of independence for categorical features comparing missing vs. non-missing cohorts.

- **Predictability Modeling**: Cross-validated Random Forest and Logistic Regression classifiers trained to predict missingness flags. Features predictable with ROC-AUC $> 0.58$ are diagnosed as **MAR** (Missing At Random) or **MNAR** (Missing Not At Random).


### 2. Multi-Tiered Imputation Strategy

- **Deterministic Domain Inferences**:

  - `CryoSleep == True` $\implies$ Amenity spending is strictly $0.0$.

  - Amenity spending $> 0$ $\implies$ `CryoSleep` is strictly `False`.

  - Planetary deck bounds: Decks `A, B, C, T` are exclusively Europa; deck `G` is exclusively Earth.

  - Passenger Group dynamics: Passengers sharing a `Group_Id` (`gggg` from `gggg_pp`) inherit matching `Cabin` attributes (`Cabin_Deck`, `Cabin_Num`, `Cabin_Side`) and `HomePlanet`.

- **KNN Imputation**: Scikit-Learn `KNNImputer` ($k=5$) applied to remaining continuous and spatial variables (`Age`, `ShoppingMall`, `VRDeck`, `Cabin_Num`, etc...).

- **Univariate Imputation**: Mode and median fallbacks applied to remaining low-cardinality nominal attributes.

- **Row Dropping Alternative (`drop_missing`)**: Drops training rows containing missing values (6,606 clean training samples) for empirical comparison against imputation.


### 3. Feature Engineering

1. **Spatial Decomposition**: Splits raw `Cabin` string into `Cabin_Deck` (ordinal hierarchy $A \dots T$), `Cabin_Num` (continuous integer), and `Cabin_Side` binary indicator (`P` = Port, `S` = Starboard).

2. **Group Dynamics**: Extracts `Group_Id` and passenger sequence from `PassengerId`. Computes `Group_Size` and the binary `Is_Solo` indicator.

3. **Amenity Spending Aggregations**:

   - `Total_Spending`: Sum of expenditure across all 5 amenities (`RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`).

   - `Has_Spent`: Binary indicator reflecting whether a passenger incurred any amenity cost (`Total_Spending > 0`).

4. **Encoding & Scaling**:

   - `RobustScaler` applied to skewed numerical expenditures and continuous attributes.

   - `OneHotEncoder` applied to nominal categories (`HomePlanet`, `Destination`).

   - `OrdinalEncoder` applied to ordered spatial attributes (`Cabin_Deck`, `Cabin_Side`).


### 4. Configurable Dataset Presets

The pipeline provides 5 preconfigured representations defined in [`src/utils/dataset_presets.py`](src/utils/dataset_presets.py):

| Preset | Imputation Strategy | Spending Features | Group Features | Encoding | Scaling | Train Samples | Features |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | Hybrid (Domain + KNN + Simple) | All 5 Raw + `Total_Spending` + `Has_Spent` | `Group_Size` + `Is_Solo` | Auto (OneHot / Ordinal) | RobustScaler | 8,693 | 21 |
| **`drop_missing`** | Row Dropping on Train | All 5 Raw + `Total_Spending` + `Has_Spent` | `Group_Size` + `Is_Solo` | Auto (OneHot / Ordinal) | RobustScaler | 6,606 | 21 |
| **`aggregated_spending_only`** | Hybrid (Domain + KNN + Simple) | `Total_Spending` + `Has_Spent` (Raw dropped) | `Group_Size` + `Is_Solo` | Auto (OneHot / Ordinal) | RobustScaler | 8,693 | 16 |
| **`raw_spending_only`** | Hybrid (Domain + KNN + Simple) | Raw 5 Amenities Only (Aggregations omitted) | `Group_Size` + `Is_Solo` | Auto (OneHot / Ordinal) | RobustScaler | 8,693 | 19 |
| **`linear_optimized`** | Hybrid (Domain + KNN + Simple) | `Total_Spending` only (Raw & `Has_Spent` dropped) | `Group_Size` only (`Is_Solo` dropped) | Auto (OneHot / Ordinal) | RobustScaler | 8,693 | 18 |

<br></br>
## 📊 Model Benchmarking & Experimental Matrix

In [`3 - SpaceShip_Training_&_Evaluation.ipynb`](3%20-%20SpaceShip_Training_&_Evaluation.ipynb), all 7 model architectures are cross-evaluated against all 5 dataset presets (**35 total experiments**) using a shared, strictly isolated test holdout managed by [`DataSplit`](src/utils/data_split.py).

### 1. Top 15 Experimental Configurations (Ranked by Test Accuracy)

| Rank | Preset | Model Architecture | Accuracy | ROC-AUC | F1-Score | Precision | Recall | Log-Loss | Train Time |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | `drop_missing` | **CatBoost** | **0.8154** | **0.9092** | **0.8187** | 0.8103 | 0.8273 | **0.3744** | 0.57s |
| 🥈 **2** | `drop_missing` | **XGBoost** | **0.8154** | 0.9052 | 0.8182 | 0.8121 | 0.8243 | 0.3882 | 0.38s |
| 🥉 **3** | `baseline` | **XGBoost** | 0.8114 | 0.9044 | 0.8119 | **0.8157** | 0.8082 | 0.3864 | 0.42s |
| **4** | `baseline` | **LightGBM** | 0.8108 | 0.9064 | 0.8138 | 0.8070 | 0.8208 | 0.3802 | 0.14s |
| **5** | `baseline` | **GradientBoosting** | 0.8108 | 0.9060 | 0.8127 | 0.8104 | 0.8151 | 0.3813 | 1.52s |
| **6** | `drop_missing` | **LightGBM** | 0.8094 | 0.9053 | 0.8133 | 0.8026 | 0.8243 | 0.3830 | 0.10s |
| **7** | `linear_optimized` | **XGBoost** | 0.8091 | 0.9028 | 0.8081 | 0.8185 | 0.7979 | 0.3893 | 0.38s |
| **8** | `linear_optimized` | **CatBoost** | 0.8085 | 0.9059 | 0.8113 | 0.8054 | 0.8174 | 0.3794 | 0.73s |
| **9** | `linear_optimized` | **LightGBM** | 0.8079 | 0.9054 | 0.8096 | 0.8087 | 0.8105 | 0.3805 | 0.13s |
| **10** | `drop_missing` | **GradientBoosting** | 0.8079 | 0.9053 | 0.8116 | 0.8021 | 0.8213 | 0.3838 | 1.12s |
| **11** | `linear_optimized` | **GradientBoosting** | 0.8074 | 0.9048 | 0.8087 | 0.8091 | 0.8082 | 0.3817 | 1.26s |
| **12** | `raw_spending_only` | **LightGBM** | 0.8068 | 0.9079 | 0.8095 | 0.8041 | 0.8151 | 0.3764 | 0.12s |
| **13** | `raw_spending_only` | **XGBoost** | 0.8056 | 0.9026 | 0.8055 | 0.8121 | 0.7991 | 0.3893 | 0.39s |
| **14** | `raw_spending_only` | **GradientBoosting** | 0.8045 | 0.9045 | 0.8070 | 0.8025 | 0.8116 | 0.3823 | 1.29s |
| **15** | `raw_spending_only` | **CatBoost** | 0.8045 | 0.9045 | 0.8083 | 0.7984 | 0.8185 | 0.3823 | 0.62s |

### 2. Optimal Dataset Preset per Model Architecture

| Model | Best Preset | Best Accuracy | Best ROC-AUC | Worst Preset | Worst Accuracy | Performance Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | `drop_missing` | **0.8154** | 0.9052 | `aggregated_spending_only` | 0.7550 | **+6.04%** |
| **CatBoost** | `drop_missing` | **0.8154** | **0.9092** | `aggregated_spending_only` | 0.7550 | **+6.04%** |
| **GradientBoosting** | `baseline` | 0.8108 | 0.9060 | `aggregated_spending_only` | 0.7660 | **+4.49%** |
| **LightGBM** | `baseline` | 0.8108 | 0.9064 | `aggregated_spending_only` | 0.7654 | **+4.54%** |
| **RandomForest** | `linear_optimized` | 0.8039 | 0.8936 | `aggregated_spending_only` | 0.7487 | **+5.52%** |
| **LogisticRegression** | `drop_missing` | 0.7973 | 0.8785 | `aggregated_spending_only` | 0.7240 | **+7.33%** |
| **NeuralNet (PyTorch)** | `raw_spending_only` | 0.7918 | 0.8771 | `aggregated_spending_only` | 0.7332 | **+5.87%** |

### 3. Overall Dataset Preset Ranking

Averaged across all 7 model families:

| Rank | Dataset Preset | Mean Accuracy | Max Accuracy | Mean ROC-AUC | Mean F1-Score | Mean Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`drop_missing`** | **0.8042** | **0.8154** | 0.8957 | **0.8098** | 0.62s |
| 🥈 **2** | **`baseline`** | 0.8022 | 0.8114 | **0.8964** | 0.8041 | 1.21s |
| 🥉 **3** | **`linear_optimized`** | 0.8017 | 0.8091 | 0.8953 | 0.8036 | 0.74s |
| **4** | **`raw_spending_only`** | 0.8008 | 0.8068 | 0.8951 | 0.8042 | 0.70s |
| **5** | **`aggregated_spending_only`** | 0.7496 | 0.7660 | 0.8158 | 0.7282 | 0.73s |


> **Key Empirical Finding on Feature Collinearity**: Collapsing individual amenity expenditures into a single aggregate (`aggregated_spending_only`) causes a massive **4.5% to 7.3% accuracy drop** across every algorithm. Retaining the individual amenity spending channels is critical, as different planets and age cohorts spend disproportionately on specific amenities (e.g. Europa passengers on Spa & VRDeck vs. Earth passengers on FoodCourt).

<br></br>
## 🤝 Heterogeneous Multi-Model Ensemble (Models, assemble!🔨)

To achieve peak predictive performance and maximize generalization on the Kaggle competition test set, we construct a **5-model heterogeneous ensemble** with majority voting:

### 1. Ensemble Architecture & Dataset Rationale

| Model | Trainer Class | Dataset Preset | Architectural Rationale | Decision Threshold |
| :--- | :--- | :---: | :--- | :---: |
| **NeuralNet** | `NeuralNetTrainer` | `linear_optimized` | Scaled, collinearity-free tabular representation optimized for deep learning gradients (PyTorch MLP: Linear + BatchNorm1d + SiLU + Dropout). | 0.50 |
| **LightGBM** | `LightGBMTrainer` | `aggregated_spending_only` | Leaf-wise tree boosting providing complementary structural diversity. | 0.50 |
| **GradientBoosting** | `GradientBoostingTrainer` | `baseline` | Classic staged boosting providing robust, well-calibrated probability estimates. | 0.50 |
| **XGBoost** | `XGBoostTrainer` | `aggregated_spending_only` | Depth-wise tree growth capturing interactions on aggregated features. | 0.50 |
| **CatBoost** | `CatBoostTrainer` | `raw_spending_only` | Symmetric oblivious trees leveraging raw amenity spending patterns with ordered statistics. | 0.50 |

### 2. Training Strategy & Majority Voting Rule

- **Full-Dataset Retraining**: Each model is trained on **100% of labeled training data** (8,693 samples) without reserving validation holdouts, maximizing sample efficiency.
- **Majority Voting**: Each model casts a binary prediction (`True` / `False`) on the 4,277 test passengers. The ensemble predicts `Transported = True` if **3 or more models** vote `True`.

```text
NeuralNet (linear_opt)      ──┐
LightGBM (agg_spending)     ──┤
GradientBoosting (baseline) ──┼──> [ Majority Voting: Σ Votes ≥ 3 ] ──> Final Prediction
XGBoost (agg_spending)      ──┤                                         (Transported: True / False)
CatBoost (raw_spending)     ──┘
```

### 3. Consensus Diagnostics & Competition Score

Evaluated via [`plot_ensemble_consensus()`](src/utils/utils.py):
- **Total Test Passengers**: 4,277
- **Unanimous Agreement (0 or 5 votes)**: **3,092 passengers (72.3%)**
- **Predicted Class Distribution**:
  - `Transported = True`: 2,103 (49.17%)
  - `Transported = False`: 2,174 (50.83%)
- **Kaggle Public Leaderboard Score**: **`0.80664`** 🥇

<br></br>
## 📟 Code Usage Examples

#### Load a Preprocessed Dataset Preset

```python
from src import SpaceShipDataset

# Load clean, preprocessed matrices for the baseline preset
X_train, y_train, X_competition_test = SpaceShipDataset.build(preset = "baseline", dir = "./dataset")
print(f"X_train: {X_train.shape} | y_train: {y_train.shape} | X_test: {X_competition_test.shape}")
```

#### Train an Individual Model

```python
from src import CatBoostTrainer, SpaceShipDataset

X_train, y_train, X_test = SpaceShipDataset.build(preset = "baseline", dir = "./dataset")

# Initialize trainer with train data and competition test holdout
trainer = CatBoostTrainer(X_train = X_train, y_train = y_train, X_competition_test = X_test)
trainer.fit(verbose = False)

# Generate predictions
test_predictions = trainer.predict(X_test, threshold = 0.5)
```

<br></br>
## 🪐 A Note from the Cockpit (and a Challenge!)

Look, I know the Spaceship Titanic isn't training a 7000B-parameter frontier model or landing a booster on a drone ship. But as someone who has always looked at the aerospace world with starry-eyed admiration, how could I resist jumping into hyperspace to save thousands of interstellar tourists from getting swallowed by a spacetime anomaly? 🛸

In the precious little free time I could scrape together, I wanted to turn what is usually a run-of-the-mill tutorial challenge into a proper engineering playground: a clean, modular, leak-free framework designed to easily plug in new model architectures, preprocessing pipelines and ensembling strategies.

If you are new to Machine Learning, consider this repository your flight simulator. Tear it apart, inspect the statistical missingness diagnostics, experiment with the dataset presets, and study how different algorithm families behave on the exact same holdout split.

> 🤫 **A quick confession on hyperparameters:**  
> You might notice that the default parameters in [`src/utils/model_presets.py`](src/utils/model_presets.py) are intentionally left void. Where's the fun in handing you a pre-packaged cheat code? The secret sauce is out there for you to uncover: play with the architectures, fine-tune the search spaces, reelaborate the dataset as you see fit, and see if you can top my leaderboard score!

May the gradients be ever in your favor. Good luck, young Padawan! 🌌

<br></br>
## 📖 References and Citation

- **Kaggle Challenge**: [Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic)
- **Gradient Boosting Libraries**: [CatBoost](https://catboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/), [XGBoost](https://xgboost.readthedocs.io/)
- **Deep Learning**: [PyTorch](https://pytorch.org/)

If you use this repository or framework in your project, please cite:

```bibtex
@misc{spaceship_titanic_challenge,
  author  = {Schilirò, Biagio Fabio},
  title   = {Spaceship Titanic Challenge: End-to-End Tabular Machine Learning Pipeline and Deep Ensemble Framework},
  url     = {https://github.com/FabioS08/SpaceShip_Titanic_Challenge},
  year    = {2026}
}
```