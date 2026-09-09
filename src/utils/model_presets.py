from typing import Any


MODEL_PRESETS: dict[str, dict[str, Any]] = {

    "catboost": {

        "description": "CatBoostClassifier with symmetric (oblivious) decision trees.",
        "default_params": {},
        "tuning_grids": {

            "fast": {
                        "depth": [4, 6, 8],
                        "learning_rate": [0.03, 0.05],
                        "l2_leaf_reg": [1.0, 3.0, 5.0],
                    },

            "thorough": {
                            "depth": [4, 6, 7, 8],
                            "learning_rate": [0.01, 0.03, 0.05, 0.08],
                            "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
                            "subsample": [0.7, 0.8, 1.0],
                        },
        },

        "search_distributions": {
                                    "depth": [4, 5, 6, 7, 8],
                                    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                                    "l2_leaf_reg": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
                                },
    },


    "lightgbm": {

        "description": "LGBMClassifier with leaf-wise tree growth.",
        "default_params": {},
        "tuning_grids": {

            "fast": {
                        "num_leaves": [15, 31, 63],
                        "learning_rate": [0.03, 0.05],
                        "max_depth": [4, 6, 8],
                    },

            "thorough": {
                            "num_leaves": [15, 31, 63, 127],
                            "learning_rate": [0.01, 0.03, 0.05],
                            "max_depth": [4, 6, 8, -1],
                            "subsample": [0.6, 0.8, 1.0],
                            "colsample_bytree": [0.6, 0.8, 1.0],
                            "reg_alpha": [0.0, 0.1, 1.0],
                            "reg_lambda": [0.0, 0.1, 1.0],
                        },
        },

        "search_distributions": {
                                    "num_leaves": [15, 20, 31, 45, 63],
                                    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
                                    "max_depth": [4, 5, 6, 7, 8],
                                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                                    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                                },
    },


    "xgboost": {

        "description": "XGBClassifier with depth-wise tree expansion.",
        "default_params": {},
        "tuning_grids": {

            "fast": {
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.03, 0.05],
                        "n_estimators": [200, 300],
                    },

            "thorough": {
                            "max_depth": [3, 4, 5, 6, 7],
                            "learning_rate": [0.01, 0.03, 0.05],
                            "subsample": [0.7, 0.8, 1.0],
                            "colsample_bytree": [0.7, 0.8, 1.0],
                            "gamma": [0.0, 0.1, 0.2],
                        },
        },

        "search_distributions": {
                                    "max_depth": [3, 4, 5, 6, 7],
                                    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
                                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                                },
    },


    "random_forest": {

        "description": "RandomForestClassifier with bootstrap ensemble trees.",
        "default_params": {},
        "tuning_grids": {

            "fast": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [8, 12, 16],
                        "min_samples_leaf": [1, 2, 4],
                    },

            "thorough": {
                            "n_estimators": [100, 200, 300, 500],
                            "max_depth": [6, 8, 10, 14, None],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "max_features": ["sqrt", "log2", 0.8],
                        },
        },

        "search_distributions": {
                                    "n_estimators": [100, 200, 300, 400],
                                    "max_depth": [6, 8, 10, 12, 15, None],
                                    "min_samples_leaf": [1, 2, 3, 4, 5],
                                },
    },


    "gradient_boosting": {

        "description": "Scikit-learn GradientBoostingClassifier / HistGradientBoostingClassifier.",
        "default_params": {},
        "tuning_grids": {

            "fast": {
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.03, 0.05, 0.1],
                        "n_estimators": [150, 250],
                    },

            "thorough": {
                            "max_depth": [3, 4, 5, 6],
                            "learning_rate": [0.02, 0.04, 0.06, 0.1],
                            "subsample": [0.7, 0.8, 0.9, 1.0],
                            "min_samples_leaf": [1, 2, 4],
                        },
        },

        "search_distributions": {
                                    "max_depth": [3, 4, 5, 6],
                                    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
                                },
    },


    "logistic_regression": {

        "description": "LogisticRegression for linear boundaries and baselines.",
        "default_params": {},
        "tuning_grids": {

            "fast": {
                        "C": [0.01, 0.1, 1.0, 10.0],
                    },

            "thorough": {
                            "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                            "penalty": ["l2"],
                        },
        },

        "search_distributions": {
                                    "C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                                },
    },


    "neural_net": {

        "description": "Multi-layer Perceptron / PyTorch TabularNN.",
        "default_params": {},
        "tuning_grids": {

            "fast": {
                        "hidden_layer_sizes": [(128, 64), (128, 64, 32), (64, 32)],
                        "alpha": [1e-4, 1e-3],
                        "learning_rate_init": [1e-3, 5e-4],
                    },

            "thorough": {
                            "hidden_layer_sizes": [(256, 128, 64), (128, 64, 32), (128, 64), (64, 32)],
                            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                            "learning_rate_init": [2e-3, 1e-3, 5e-4],
                            "activation": ["relu", "tanh"],
                        },
        },

        "search_distributions": {
                                    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                                    "learning_rate_init": [2e-3, 1e-3, 5e-4, 1e-4],
                                },
    }
}


def get_model_preset(model_name: str) -> dict[str, Any]:

    '''
    It retrieves the parameter dictionary for a given model.
    
    Parameters
    ----------
     model_name:str
      Model name.
    '''

    norm = model_name.lower().strip().replace(" ", "_")
    alias_map = {
                    "rf": "random_forest",
                    "cat": "catboost",
                    "lgb": "lightgbm",
                    "lgbm": "lightgbm",
                    "xgb": "xgboost",
                    "gb": "gradient_boosting",
                    "lr": "logistic_regression",
                    "mlp": "neural_net",
                    "nn": "neural_net",
                }

    resolved = alias_map.get(norm, norm)

    if resolved not in MODEL_PRESETS:
        
        avail = ", ".join(f"'{k}'" for k in MODEL_PRESETS.keys())
        raise KeyError(f"Unknown model preset '{model_name}'. Available: {avail}")
    
    return MODEL_PRESETS[resolved]


def list_model_presets() -> list[str]:

    '''
    It returns a list of all available model preset keys.

    Returns
    -------
     list[str]
      List of model preset identifier strings.
    '''

    return list(MODEL_PRESETS.keys())