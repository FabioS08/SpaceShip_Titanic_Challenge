from typing import Any


# Dataset experimental presets for model evaluation and preprocessing pipelines
DATASET_PRESETS: dict[str, dict[str, Any]] = {

    "baseline": {

        "description": "Full dataset with hybrid imputation (Domain + KNN + Simple), individual amenities + Total_Spending + Has_Spent, both Group_Size and Is_Solo, auto encoding (OneHot for nominal, Ordinal for deck/side), and RobustScaler.",
        "missing_strategy": "impute",
        "introduce_features": True,
        "drop_original_amenities": False,
        "include_has_spent": True,
        "include_is_solo": True,
        "encoding_strategy": "auto",
        "scale_numeric": True,
    },
    
    "drop_missing": {
        "description": "Row-dropping on train for missing values instead of imputation, all amenities + Total_Spending + Has_Spent, both Group_Size and Is_Solo, auto encoding, and RobustScaler.",

        "missing_strategy": "drop",
        "introduce_features": True,
        "drop_original_amenities": False,
        "include_has_spent": True,
        "include_is_solo": True,
        "encoding_strategy": "auto",
        "scale_numeric": True,
    },
    
    "aggregated_spending_only": {

        "description": "Hybrid imputation with individual amenity expenses dropped (retaining only Total_Spending and Has_Spent to reduce collinearity), both Group_Size and Is_Solo, auto encoding, and RobustScaler.",
        "missing_strategy": "impute",
        "introduce_features": True,
        "drop_original_amenities": True,
        "include_has_spent": True,
        "include_is_solo": True,
        "encoding_strategy": "auto",
        "scale_numeric": True,
    },
    
    "raw_spending_only": {

        "description": "Hybrid imputation keeping only original individual amenity features without aggregated Total_Spending or Has_Spent indicators, both Group_Size and Is_Solo, auto encoding, and RobustScaler.",
        "missing_strategy": "impute",
        "introduce_features": False,
        "drop_original_amenities": False,
        "include_has_spent": False,
        "include_is_solo": True,
        "encoding_strategy": "auto",
        "scale_numeric": True,
    },
    
    "linear_optimized": {
        "description": "Tailored for linear/distance models (LogisticRegression, SVM, Neural Nets): hybrid imputation, all spending features, Is_Solo dropped to eliminate multicollinearity with Group_Size, and RobustScaler.",
        "missing_strategy": "impute",
        "introduce_features": False,
        "drop_original_amenities": True,
        "include_has_spent": False, 
        "include_is_solo": False,
        "encoding_strategy": "auto",
        "scale_numeric": True,
    }
}
