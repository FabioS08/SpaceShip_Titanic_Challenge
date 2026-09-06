# Target column
TARGET_COL: str = "Transported"

# PassengerId and Cabin decomposition features
CABIN_SPLIT_COLS: list[str] = ["Cabin_Deck", "Cabin_Num", "Cabin_Side"]
GROUP_SPLIT_COLS: list[str] = ["Group_Id", "Group_Size", "Is_Solo"]

# High-cardinality identifiers / raw metadata to drop after feature engineering
HIGH_CARDINALITY_COLS: list[str] = ["PassengerId", "Cabin", "Name", "Group_Id"]

# Spending / Amenity features
SPENDING_COLS: list[str] = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Deterministic domain rule mappings
EUROPA_DECKS: list[str] = ["A", "B", "C", "T"]
EARTH_DECKS: list[str] = ["G"]

# KNN baseline predictor features
KNN_BASE_NUMERIC_COLS: list[str] = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Group_Size", "Cabin_Num"]

# Default imputation feature routing (simple univariate vs KNN)
DEFAULT_SIMPLE_IMPUTE_COLS: list[str] = ["HomePlanet", "CryoSleep", "Destination", "VIP", "RoomService", "FoodCourt", "Spa"]
DEFAULT_KNN_IMPUTE_COLS: list[str] = ["Age", "ShoppingMall", "VRDeck", "Cabin_Num", "Cabin_Deck", "Cabin_Side"]

# Feature type enforcement groups
BOOL_COLS: list[str] = ["CryoSleep", "VIP", "Transported", "Is_Solo", "Has_Spent"]
INT_COLS: list[str] = ["Age", "Cabin_Num", "Group_Size", "Group_Id"]
FLOAT_COLS: list[str] = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Total_Spending"]
CAT_COLS: list[str] = ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_Side"]

# Categorical routing & hierarchy
NOMINAL_CAT_COLS: list[str] = ["HomePlanet", "Destination"]
ORDINAL_CAT_COLS: list[str] = ["Cabin_Deck", "Cabin_Side"]

CABIN_DECK_ORDER: list[str] = ["A", "B", "C", "D", "E", "F", "G", "T"]
CABIN_SIDE_ORDER: list[str] = ["P", "S"]

DEFAULT_ORDINAL_CATEGORIES: dict[str, list[str]] = {"Cabin_Deck": CABIN_DECK_ORDER, "Cabin_Side": CABIN_SIDE_ORDER}

# Exclusion sets for diagnostics & visualizer
DEFAULT_EXCLUDE_COLS: list[str] = ["PassengerId", "Name", "Cabin"]
EXCLUDED_CATEGORICAL_COLS: list[str] = ["PassengerId", "Name", "Cabin", "Transported"]