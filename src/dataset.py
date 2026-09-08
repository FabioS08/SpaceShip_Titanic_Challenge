from .utils.constants import (TARGET_COL, CABIN_SPLIT_COLS, GROUP_SPLIT_COLS, HIGH_CARDINALITY_COLS,
                              SPENDING_COLS, EUROPA_DECKS, EARTH_DECKS, KNN_BASE_NUMERIC_COLS,
                              DEFAULT_SIMPLE_IMPUTE_COLS, DEFAULT_KNN_IMPUTE_COLS,
                              BOOL_COLS, INT_COLS, FLOAT_COLS, CAT_COLS, NOMINAL_CAT_COLS, ORDINAL_CAT_COLS,
                              CABIN_DECK_ORDER, CABIN_SIDE_ORDER, DEFAULT_ORDINAL_CATEGORIES)
from .utils.dataset_presets import DATASET_PRESETS
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from typing import Any, Literal
from textwrap import dedent
from pathlib import Path
import pandas as pd
import numpy as np
import contextlib
import kagglehub
import io


class SpaceShipDataset:

    # Class constants for feature definitions and dataset schema
    TARGET_COL: str = TARGET_COL
    CABIN_SPLIT_COLS: list[str] = CABIN_SPLIT_COLS
    GROUP_SPLIT_COLS: list[str] = GROUP_SPLIT_COLS
    HIGH_CARDINALITY_COLS: list[str] = HIGH_CARDINALITY_COLS
    SPENDING_COLS: list[str] = SPENDING_COLS
    EUROPA_DECKS: list[str] = EUROPA_DECKS
    EARTH_DECKS: list[str] = EARTH_DECKS
    KNN_BASE_NUMERIC_COLS: list[str] = KNN_BASE_NUMERIC_COLS
    DEFAULT_SIMPLE_IMPUTE_COLS: list[str] = DEFAULT_SIMPLE_IMPUTE_COLS
    DEFAULT_KNN_IMPUTE_COLS: list[str] = DEFAULT_KNN_IMPUTE_COLS
    BOOL_COLS: list[str] = BOOL_COLS
    INT_COLS: list[str] = INT_COLS
    FLOAT_COLS: list[str] = FLOAT_COLS
    CAT_COLS: list[str] = CAT_COLS
    NOMINAL_CAT_COLS: list[str] = NOMINAL_CAT_COLS
    ORDINAL_CAT_COLS: list[str] = ORDINAL_CAT_COLS
    CABIN_DECK_ORDER: list[str] = CABIN_DECK_ORDER
    CABIN_SIDE_ORDER: list[str] = CABIN_SIDE_ORDER
    DEFAULT_ORDINAL_CATEGORIES: dict[str, list[str]] = DEFAULT_ORDINAL_CATEGORIES
    DATASET_PRESETS: dict[str, dict[str, Any]] = DATASET_PRESETS

    '''
    Class to load and preprocess the SpaceShip Titanic Dataset.

    Parameters
    ----------
     dir:str
      The directory containing the dataset

     extended_features:bool
      Whether to create an extended version of the features (e.g. cabin splitted in multiple features)

    Attributes
    ----------
     dir:str
      The directory containing the dataset

     train:pd.DataFrame
      The training dataset
     
     test:pd.DataFrame
      The test dataset

    '''

    def __init__(self, dir: str = './', extended_features: bool = True):

        self.dir = Path(dir)
        download_data_required = not self._check_dataset_existence()
        self.load_data(download_data_required)

        if extended_features:
            self.preprocess_dataset()


    @classmethod
    def list_presets(cls) -> pd.DataFrame:

        '''
        Returns a formatted DataFrame of all available dataset presets and their preprocessing configurations.

        Returns
        -------
         pd.DataFrame
          Summary table containing preset names, descriptions, and feature/imputation settings.
        '''

        records = []
        for name, cfg in cls.DATASET_PRESETS.items():

            records.append({
                                "Preset": name,
                                "Description": cfg["description"],
                                "Missing Strategy": cfg["missing_strategy"],
                                "New Features": cfg.get("introduce_features", True),
                                "Drop Amenities": cfg["drop_original_amenities"],
                                "Has_Spent": cfg["include_has_spent"],
                                "Is_Solo": cfg.get("include_is_solo", True),
                                "Encoding Strategy": cfg["encoding_strategy"],
                                "Scale Numeric": cfg["scale_numeric"],
                            })

        return pd.DataFrame(records)


    @classmethod
    def _resolve_preset_config(cls, preset: str) -> dict[str, Any]:

        '''
        Resolves pipeline configuration directly from a predefined preset.

        Parameters
        ----------
         preset:str
          Preset identifier key in DATASET_PRESETS.

        Returns
        -------
         dict[str, Any]
          Dictionary containing the preset configuration parameters.
        '''

        if preset not in cls.DATASET_PRESETS:
            available = ", ".join([f"'{p}'" for p in cls.DATASET_PRESETS.keys()])
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")

        return cls.DATASET_PRESETS[preset]


    @classmethod
    def _resolve_custom_config(cls, missing_strategy: Literal["impute", "drop"] | None = None,
                               introduce_features: bool | None = None, drop_original_amenities: bool | None = None,
                               include_has_spent: bool | None = None, include_is_solo: bool | None = None,
                               encoding_strategy: Literal["auto", "onehot", "ordinal"] | None = None,
                               scale_numeric: bool | None = None) -> dict[str, Any]:

        '''
        Resolves custom pipeline configuration applying baseline defaults.

        Parameters
        ----------
         missing_strategy:Literal['impute', 'drop'] | None
          Missing value resolution strategy [Default = 'impute'].

         introduce_features:bool | None
          Whether to introduce aggregated spending features [Default = True].

         drop_original_amenities:bool | None
          Whether to drop individual amenity features [Default = False].

         include_has_spent:bool | None
          Whether to introduce the Has_Spent indicator [Default = True].

         include_is_solo:bool | None
          Whether to retain the Is_Solo indicator (when False, only Group_Size is retained to avoid multicollinearity) [Default = True].

         encoding_strategy:Literal['auto', 'onehot', 'ordinal'] | None
          Categorical encoding strategy [Default = 'auto'].

         scale_numeric:bool | None
          Whether to scale numeric features [Default = True].

        Returns
        -------
         dict[str, Any]
          Dictionary containing the resolved configuration parameters.
        '''

        return {
                    "missing_strategy": missing_strategy if missing_strategy is not None else "impute",
                    "introduce_features": introduce_features if introduce_features is not None else True,
                    "drop_original_amenities": drop_original_amenities if drop_original_amenities is not None else False,
                    "include_has_spent": include_has_spent if include_has_spent is not None else True,
                    "include_is_solo": include_is_solo if include_is_solo is not None else True,
                    "encoding_strategy": encoding_strategy if encoding_strategy is not None else "auto",
                    "scale_numeric": scale_numeric if scale_numeric is not None else True,
                    "description": "Custom configuration"
                }


    @classmethod
    def _print_pipeline_configuration(cls, config: dict[str, Any], preset: str | None = None) -> None:

        '''
        Prints the pipeline configuration banner and selected options.

        Parameters
        ----------
         config:dict[str, Any]
          Resolved pipeline configuration dictionary.

         preset:str | None
          Preset identifier name, or None for custom configuration. [Default = None]

        Returns
        -------
         None
        '''

        banner_title = f"🚀 SpaceShipDataset Pipeline: Preset '{preset}'" if preset else "🚀 SpaceShipDataset Pipeline: Custom Configuration"
        print("\n" + "=" * 85)
        print(banner_title)
        if preset:
            print(f"📋 Rationale: {config['description']}")
        print("⚙️ Configuration:")
        print(f"  • Missing Strategy:         {config['missing_strategy'].upper()}")
        feat_items = [
            f"[{'✓' if config['introduce_features'] else ' '}] Total_Spending",
            f"[{'✓' if config['introduce_features'] and config['include_has_spent'] else ' '}] Has_Spent",
            f"[{'✓' if config.get('include_is_solo', True) else ' '}] Is_Solo",
        ]
        print(f"  • Feature Engineering:      {', '.join(feat_items)}")
        print(f"  • Original Amenities:       {'Dropped (Aggregated spending only)' if config['drop_original_amenities'] else 'Retained'}")
        print(f"  • Categorical Encoding:     {config['encoding_strategy'].upper()}")
        print(f"  • Numeric Scaling:          {'RobustScaler' if config['scale_numeric'] else 'Skipped (Unscaled)'}")
        print("=" * 85 + "\n")


    @classmethod
    def _print_pipeline_summary(cls, ds: "SpaceShipDataset") -> None:

        '''
        Prints the dataset summary banner including shapes, null counts, and feature list.

        Parameters
        ----------
         ds:SpaceShipDataset
          Processed SpaceShipDataset instance.

        Returns
        -------
         None
        '''

        features = [c for c in ds.train.columns if c != cls.TARGET_COL]
        print("\n" + "=" * 85)
        print("✅ Dataset Variant Ready:")
        print(f"  • X_train: {ds.train.shape[0]:,} rows × {len(features)} features")
        print(f"  • y_train: {ds.train.shape[0]:,} rows (Target: '{cls.TARGET_COL}')")
        print(f"  • X_test:  {ds.test.shape[0]:,} rows × {len(ds.test.columns)} features")
        print(f"  • Remaining Nulls: Train = {ds.train.isnull().sum().sum()} | Test = {ds.test.isnull().sum().sum()}")
        print(f"  • Features ({len(features)}): {', '.join(features)}")
        print("=" * 85 + "\n")


    @classmethod
    def _extract_xy(cls, ds: "SpaceShipDataset") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

        '''
        Extracts training feature matrix, target vector, and test feature matrix from a SpaceShipDataset.

        Parameters
        ----------
         ds:SpaceShipDataset
          Processed SpaceShipDataset instance.

        Returns
        -------
         tuple[pd.DataFrame, pd.Series, pd.DataFrame]
          A tuple containing (X_train, y_train, X_test).
        '''

        X_train = ds.train.drop(columns = [cls.TARGET_COL])
        y_train = ds.train[cls.TARGET_COL]
        X_test = ds.test.copy()

        return X_train, y_train, X_test


    @classmethod
    def _execute_pipeline(cls, dir: str, config: dict[str, Any]) -> "SpaceShipDataset":

        '''
        Instantiates a fresh SpaceShipDataset and executes the step-by-step preprocessing pipeline.

        Parameters
        ----------
         dir:str
          Directory where train.csv and test.csv are located.

         config:dict[str, Any]
          Resolved pipeline configuration dictionary.

        Returns
        -------
         SpaceShipDataset
          Processed SpaceShipDataset instance.
        '''

        ds = cls(dir = dir)

        # Step 1: Missing values
        ds.handle_missing_values(strategy = config["missing_strategy"])

        # Step 2: High cardinality identifiers
        ds.drop_high_cardinality_features()

        # Step 3: New feature introduction
        ds.new_feature_introduction(introduce_features = config["introduce_features"], drop_original_amenities = config["drop_original_amenities"],
                                    include_has_spent = config["include_has_spent"])

        # Step 4: Group features resolution (Group_Size vs Is_Solo)
        ds.handle_group_features(include_is_solo = config.get("include_is_solo", True))

        # Step 5: Enforce types
        ds.handle_types()

        # Step 6: Scaling & categorical encoding
        numeric_cols: list[str] | None = None if config["scale_numeric"] else []
        ds.scale_and_encode_features(numeric_cols = numeric_cols, encoding_strategy = config["encoding_strategy"])

        return ds


    @classmethod
    def build(cls, preset: str | None = None, dir: str = "./", missing_strategy: Literal["impute", "drop"] | None = None, introduce_features: bool | None = None,
              drop_original_amenities: bool | None = None, include_has_spent: bool | None = None, include_is_solo: bool | None = None,
              encoding_strategy: Literal["auto", "onehot", "ordinal"] | None = None,
              scale_numeric: bool | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

        '''
        Factory method that instantiates a fresh SpaceShipDataset, executes the end-to-end preprocessing pipeline according 
        to a designated preset (or custom configuration) and returns ready-to-train feature matrices and target vector.

        Parameters
        ----------
         preset:str | None
          Name of a predefined experimental preset. [Default = None]

         dir:str
          Directory where train.csv and test.csv are located [Default = './'].

         missing_strategy:Literal['impute', 'drop'] | None
          Strategy for resolving missing values when preset is None.

         introduce_features:bool | None
          Whether to introduce aggregated spending features (Total_Spending, Has_Spent) when preset is None.

         drop_original_amenities:bool | None
          Whether to drop individual amenity spending features after aggregation when preset is None.

         include_has_spent:bool | None
          Whether to introduce the binary Has_Spent indicator when preset is None.

         include_is_solo:bool | None
          Whether to retain the Is_Solo indicator (when False, only Group_Size is retained to avoid multicollinearity) when preset is None.

         encoding_strategy:Literal['auto', 'onehot', 'ordinal'] | None
          Categorical encoding strategy ('auto', 'onehot', or 'ordinal') when preset is None.

         scale_numeric:bool | None
          Whether to scale numerical features using RobustScaler when preset is None.

         verbose:bool
          Whether to print progress messages during pipeline execution [Default = True].

        Returns
        -------
         tuple[pd.DataFrame, pd.Series, pd.DataFrame]
          A tuple containing (X_train, y_train, X_test).
        '''

        # Resolve configuration from preset or custom parameters
        if preset is not None:
            config = cls._resolve_preset_config(preset = preset)

        else:
            config = cls._resolve_custom_config(missing_strategy = missing_strategy, introduce_features = introduce_features,
                                                drop_original_amenities = drop_original_amenities, include_has_spent = include_has_spent,
                                                include_is_solo = include_is_solo,
                                                encoding_strategy = encoding_strategy, scale_numeric = scale_numeric)

        # Run with or without stdout output
        if verbose:

            cls._print_pipeline_configuration(config = config, preset = preset)
            ds = cls._execute_pipeline(dir = dir, config = config)
            cls._print_pipeline_summary(ds = ds)

        else:

            with contextlib.redirect_stdout(io.StringIO()):
                ds = cls._execute_pipeline(dir = dir, config = config)            

        return cls._extract_xy(ds = ds)


    def _check_dataset_existence(self) -> bool:

        '''
        It checks whether the dataset has been already downloaded in the target directory.

        Parameters
        ----------
         None

        Returns
        -------
         download_data_required:bool
          True if the dataset has been found, False otherwise
        '''

        is_train = (self.dir / 'train.csv').exists()
        is_test = (self.dir / 'test.csv').exists()

        if is_train and is_test:
            print(f"The dataset has been found in '{self.dir}'!")
            return True
        
        print(f"The dataset has not been found in '{self.dir}'!")
        return False


    def load_data(self, download_data_required: bool) -> None:

        '''
        It loads the dataset in memory.

        Parameters
        ----------
         download_data_required:bool
          True if the dataset has to be downloaded, False otherwise
        
        Returns
        -------
         None
        '''

        if download_data_required:
            kagglehub.competition_download('spaceship-titanic', output_dir = str(self.dir))

        self.train = pd.read_csv(self.dir / 'train.csv')
        self.test = pd.read_csv(self.dir / 'test.csv')  


    @staticmethod
    def _preprocess_split_data(df: pd.DataFrame) -> pd.DataFrame:

        # Decompose Cabin
        df[CABIN_SPLIT_COLS] = df["Cabin"].str.split("/", expand = True)
        df["Cabin_Num"] = pd.to_numeric(df["Cabin_Num"], errors = "coerce").astype("Int64")

        # Extract Group Size from PassengerId
        df["Group_Id"] = df["PassengerId"].apply(lambda x: int(x.split("_")[0]))
        group_sizes = df["Group_Id"].value_counts()
        df["Group_Size"] = df["Group_Id"].map(group_sizes)
        df["Is_Solo"] = df["Group_Size"] == 1

        return df


    def preprocess_dataset(self) -> None:

        '''
        It preprocesses the dataset to split the features that contains multiple information, such as 
        cabin and passengerId.
        '''

        self.train = self._preprocess_split_data(self.train)
        self.test = self._preprocess_split_data(self.test)


    def _build_group_mappings(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

        '''
        It builds lookup mappings for HomePlanet and Cabin components from observed group members across train and test.

        Returns
        -------
         tuple[pd.Series, pd.Series, pd.Series, pd.Series]
          Lookups mapping Group_Id to HomePlanet, Cabin_Deck, Cabin_Side, and Cabin_Num.
        '''

        combined = pd.concat([self.train, self.test], ignore_index = True)

        planet_map = combined.dropna(subset = ["HomePlanet"]).groupby("Group_Id")["HomePlanet"].first()
        deck_map = combined.dropna(subset = ["Cabin_Deck"]).groupby("Group_Id")["Cabin_Deck"].first()
        side_map = combined.dropna(subset = ["Cabin_Side"]).groupby("Group_Id")["Cabin_Side"].first()
        num_map = combined.dropna(subset = ["Cabin_Num"]).groupby("Group_Id")["Cabin_Num"].first()

        return planet_map, deck_map, side_map, num_map


    @staticmethod
    def _impute_cryosleep_from_spending(df: pd.DataFrame, amenities: list[str]) -> None:

        '''
        Rule 1: If a passenger spent money on amenities, they cannot be in CryoSleep (CryoSleep = False).

        Parameters
        ----------
         df:pd.DataFrame
          The dataframe partition to update in-place.

         amenities:list[str]
          List of amenity expenditure feature names.

        Returns
        -------
         None
        '''

        if "CryoSleep" in df.columns:

            has_spending = df[amenities].sum(axis = 1) > 0
            df.loc[df["CryoSleep"].isnull() & has_spending, "CryoSleep"] = False


    @staticmethod
    def _impute_amenities_from_cryosleep(df: pd.DataFrame, amenities: list[str]) -> None:

        '''
        Rule 2: If a passenger is in CryoSleep, all amenity expenditures must be 0.0.

        Parameters
        ----------
         df:pd.DataFrame
          The dataframe partition to update in-place.

         amenities:list[str]
          List of amenity expenditure feature names.

        Returns
        -------
         None
        '''

        if "CryoSleep" in df.columns:

            is_cryosleep = df["CryoSleep"] == True
            
            for col in amenities:
                if col in df.columns:
                    df.loc[is_cryosleep & df[col].isnull(), col] = 0.0


    @staticmethod
    def _impute_homeplanet_from_deck(df: pd.DataFrame) -> None:

        '''
        Rule 3: Cabin Deck determines HomePlanet (Decks A, B, C, T are Europa; Deck G is Earth).

        Parameters
        ----------
         df:pd.DataFrame
          The dataframe partition to update in-place.

        Returns
        -------
         None
        '''

        if "HomePlanet" in df.columns and "Cabin_Deck" in df.columns:

            df.loc[df["HomePlanet"].isnull() & df["Cabin_Deck"].isin(EUROPA_DECKS), "HomePlanet"] = "Europa"
            df.loc[df["HomePlanet"].isnull() & df["Cabin_Deck"].isin(EARTH_DECKS), "HomePlanet"] = "Earth"


    @staticmethod
    def _impute_from_group_mappings(df: pd.DataFrame, planet_map: pd.Series, deck_map: pd.Series, side_map: pd.Series, num_map: pd.Series) -> None:

        '''
        Rules 4 & 5: Imputes HomePlanet and Cabin features from other travel group members with the same Group_Id.

        Parameters
        ----------
         df:pd.DataFrame
          The dataframe partition to update in-place.

         planet_map:pd.Series
          Lookup mapping Group_Id to HomePlanet.

         deck_map:pd.Series
          Lookup mapping Group_Id to Cabin_Deck.

         side_map:pd.Series
          Lookup mapping Group_Id to Cabin_Side.

         num_map:pd.Series
          Lookup mapping Group_Id to Cabin_Num.

        Returns
        -------
         None
        '''

        if "HomePlanet" in df.columns and "Group_Id" in df.columns:
            df["HomePlanet"] = df["HomePlanet"].fillna(df["Group_Id"].map(planet_map))

        if "Group_Id" in df.columns:

            if "Cabin_Deck" in df.columns:
                df["Cabin_Deck"] = df["Cabin_Deck"].fillna(df["Group_Id"].map(deck_map))
            
            if "Cabin_Side" in df.columns:
                df["Cabin_Side"] = df["Cabin_Side"].fillna(df["Group_Id"].map(side_map))
            
            if "Cabin_Num" in df.columns:
                df["Cabin_Num"] = df["Cabin_Num"].fillna(df["Group_Id"].map(num_map))


    def _impute_domain_rules(self) -> None:

        '''
        It applies deterministic domain-specific constraints across both train and test sets:
        
            - CryoSleep = False if any amenity expenditure > 0.
            - Amenities = 0.0 if CryoSleep is True.
            - HomePlanet = Europa if Cabin_Deck in ['A', 'B', 'C', 'T'].
            - HomePlanet = Earth if Cabin_Deck == 'G'.
            - HomePlanet and Cabin features shared among passengers with the same Group_Id.

        Returns
        -------
         None
        '''

        amenities = self.SPENDING_COLS
        planet_map, deck_map, side_map, num_map = self._build_group_mappings()

        for df in [self.train, self.test]:

            # Rule 1: CryoSleep from spending
            self._impute_cryosleep_from_spending(df = df, amenities = amenities)

            # Rule 2: Amenities from CryoSleep
            self._impute_amenities_from_cryosleep(df = df, amenities = amenities)

            # Rule 3: HomePlanet from Cabin Deck
            self._impute_homeplanet_from_deck(df = df)

            # Rules 4 & 5: HomePlanet and Cabin components from Group_Id
            self._impute_from_group_mappings(df = df, planet_map = planet_map, deck_map = deck_map, side_map = side_map, num_map = num_map)


    def _impute_numeric_feature(self, col: str) -> None:

        '''
        It imputes missing values in a numeric feature using the median calculated on the train set.

        Parameters
        ----------
         col:str
          Feature column name to impute.

        Returns
        -------
         None
        '''

        median_val = self.train[col].median()
        self.train[col] = self.train[col].fillna(median_val)

        if col in self.test.columns:
            self.test[col] = self.test[col].fillna(median_val)


    def _impute_categorical_feature(self, col: str) -> None:

        '''
        It imputes missing values in a categorical or boolean feature using the mode calculated on the train set.

        Parameters
        ----------
         col:str
          Feature column name to impute.

        Returns
        -------
         None
        '''

        mode_series = self.train[col].mode()

        if len(mode_series) > 0:

            mode_val = mode_series.iloc[0]
            self.train[col] = self.train[col].fillna(mode_val)

            if col in self.test.columns:
                self.test[col] = self.test[col].fillna(mode_val)


    def _impute_simple(self, cols: list[str]) -> None:

        '''
        It imputes missing values using univariate summary statistics fitted strictly on the train set:
           
            - Numeric features: Median
            - Categorical / Boolean features: Mode

        Parameters
        ----------
         cols:list[str]
          List of column names to impute.

        Returns
        -------
         None
        '''

        for col in cols:

            if col not in self.train.columns or col == self.TARGET_COL:
                continue

            if pd.api.types.is_numeric_dtype(self.train[col]):
                self._impute_numeric_feature(col = col)

            else:
                self._impute_categorical_feature(col = col)


    def _prepare_knn_matrices(self, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[dict[str, int], dict[int, str], int]]]:

        '''
        It prepares and encodes feature matrices from train and test sets for KNN imputation.
        Numerical features are coerced to numeric, while categorical features are ordinally encoded
        using mappings learned strictly from the train set.

        Parameters
        ----------
         cols:list[str]
          List of column names targeted for KNN imputation.

        Returns
        -------
         tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[dict[str, int], dict[int, str], int]]]
          A tuple containing (X_train, X_test, encoders).
        '''

        base_numeric = self.KNN_BASE_NUMERIC_COLS
        predictor_cols = [c for c in base_numeric if c in self.train.columns]
        feature_set = list(dict.fromkeys(predictor_cols + [c for c in cols if c in self.train.columns and c != self.TARGET_COL]))

        X_train = pd.DataFrame(index = self.train.index)
        X_test = pd.DataFrame(index = self.test.index)

        encoders: dict[str, tuple[dict[str, int], dict[int, str], int]] = {}

        for col in feature_set:

            if pd.api.types.is_numeric_dtype(self.train[col]):

                X_train[col] = pd.to_numeric(self.train[col], errors = "coerce")
                X_test[col] = pd.to_numeric(self.test[col], errors = "coerce")

            else:

                train_non_null = self.train[col].dropna().astype(str)
                categories = sorted(train_non_null.unique())

                if len(categories) > 0:

                    cat_to_int = {cat: idx for idx, cat in enumerate(categories)}
                    int_to_cat = {idx: cat for idx, cat in enumerate(categories)}
                    encoders[col] = (cat_to_int, int_to_cat, len(categories))

                    X_train[col] = self.train[col].astype(str).map(cat_to_int)
                    X_test[col] = self.test[col].astype(str).map(cat_to_int)

        return X_train, X_test, encoders


    def _assign_knn_categorical_feature(self, col: str, train_imp_col: pd.Series, test_imp_col: pd.Series,
                                        int_to_cat: dict[int, str], n_cats: int) -> None:

        '''
        It decodes and assigns KNN-imputed categorical features back to train and test sets.

        Parameters
        ----------
         col:str
          Categorical feature column name.

         train_imp_col:pd.Series
          Imputed continuous values for the train set.

         test_imp_col:pd.Series
          Imputed continuous values for the test set.

         int_to_cat:dict[int, str]
          Mapping from integer index back to categorical category string.

         n_cats:int
          Number of unique categories.

        Returns
        -------
         None
        '''

        train_ints = train_imp_col.round().clip(0, n_cats - 1).astype(int)
        test_ints = test_imp_col.round().clip(0, n_cats - 1).astype(int)

        self.train[col] = train_ints.map(int_to_cat)
        if col in self.test.columns:
            self.test[col] = test_ints.map(int_to_cat)


    def _assign_knn_numeric_feature(self, col: str, train_imp_col: pd.Series, test_imp_col: pd.Series) -> None:

        '''
        It assigns KNN-imputed numeric features back to train and test sets.

        Parameters
        ----------
         col:str
          Numeric feature column name.

         train_imp_col:pd.Series
          Imputed values for the train set.

         test_imp_col:pd.Series
          Imputed values for the test set.

        Returns
        -------
         None
        '''

        self.train[col] = train_imp_col
        if col in self.test.columns:
            self.test[col] = test_imp_col


    def _impute_knn(self, cols: list[str], n_neighbors: int = 5) -> None:

        '''
        It imputes missing values using a k-Nearest Neighbors imputer fit exclusively on the train set.
        Supports both numerical features and temporarily encoded categorical features.

        Parameters
        ----------
         cols:list[str]
          List of column names to impute with KNN.

         n_neighbors:int
          Number of nearest neighbors to use [Default = 5].

        Returns
        -------
         None
        '''

        X_train, X_test, encoders = self._prepare_knn_matrices(cols = cols)

        imputer = KNNImputer(n_neighbors = n_neighbors)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        train_imp_df = pd.DataFrame(X_train_imp, columns = X_train.columns, index = self.train.index)
        test_imp_df = pd.DataFrame(X_test_imp, columns = X_test.columns, index = self.test.index)

        for col in cols:

            if col not in self.train.columns or col == self.TARGET_COL:
                continue

            if col in encoders:

                _, int_to_cat, n_cats = encoders[col]
                self._assign_knn_categorical_feature(col = col, train_imp_col = train_imp_df[col], test_imp_col = test_imp_df[col], 
                                                     int_to_cat = int_to_cat, n_cats = n_cats)

            else:

                self._assign_knn_numeric_feature(col = col, train_imp_col = train_imp_df[col], test_imp_col = test_imp_df[col])


    def impute_missing_values(self, simple_cols: list[str] | None = None, knn_cols: list[str] | None = None,
                              apply_domain_rules: bool = True, n_neighbors: int = 10) -> None:

        '''
        Orchestrates missing value imputation across train and test sets without data leakage:

            1. (Optional) Applies deterministic domain constraints (CryoSleep, Deck, GroupId).
            2. (Optional) Imputes designated MAR features using KNNImputer.
            3. (Optional) Imputes designated MCAR (or remaining) features using train medians/modes.

        Parameters
        ----------
         simple_cols:list[str] | None
          Features to impute with univariate median/mode. If both simple_cols and knn_cols are None,
          all the remaining missing features are automatically imputed with median/mode.

         knn_cols:list[str] | None
          Features to impute with KNNImputer (fit on train only).

         apply_domain_rules:bool
          Whether to apply deterministic domain constraints first [Default = True].

         n_neighbors:int
          Number of nearest neighbors for KNNImputer [Default = 10].

        Returns
        -------
         None
        '''

        # Resolve default imputation strategy if neither is explicitly passed
        if simple_cols is None and knn_cols is None:
            simple_cols = self.DEFAULT_SIMPLE_IMPUTE_COLS.copy()
            knn_cols = self.DEFAULT_KNN_IMPUTE_COLS.copy()

        # Deterministic domain constraints
        if apply_domain_rules:
            self._impute_domain_rules()

        # KNN Imputation for MAR features
        if knn_cols:
            self._impute_knn(cols = knn_cols, n_neighbors = n_neighbors)

        # 4. Simple Imputation for MCAR / specified features
        if simple_cols:
            self._impute_simple(cols = simple_cols)

        elif knn_cols is None:
            
            remaining_missing = [c for c in self.train.columns if c != self.TARGET_COL and self.train[c].isnull().sum() > 0]
            self._impute_simple(cols = remaining_missing)


    def handle_missing_values(self, strategy: str = "impute", subset: list[str] | None = None, apply_domain_rules: bool | None = None, 
                              simple_cols: list[str] | None = None, knn_cols: list[str] | None = None, n_neighbors: int = 10) -> None:

        '''
        Handles missing values across the dataset using the specified strategy ('impute' or 'drop').

        Parameters
        ----------
         strategy:str
          Strategy to resolve missing values: 'impute' or 'drop'. [Default = 'impute']
          - 'impute': Imputes missing values across train and test sets using domain rules, KNN and simple imputation.
          - 'drop': Drops rows with missing values from the train set only (preserving the test set for submission integrity).

         subset:list[str] | None
          Specific columns to consider when strategy='drop'. If None, considers all columns in train. [Default = None]

         apply_domain_rules:bool | None
          Whether to apply deterministic domain rules before imputation/dropping. [Default = True when strategy='impute', and False when strategy='drop']

         simple_cols:list[str] | None
          Features to impute with univariate median/mode when strategy = 'impute'.

         knn_cols:list[str] | None
          Features to impute with KNNImputer when strategy='impute'.

         n_neighbors:int
          Number of nearest neighbors for KNNImputer when strategy='impute' [Default = 10].

        Returns
        -------
         None
        '''

        if strategy not in ["impute", "drop"]:
            raise ValueError(f"Invalid strategy '{strategy}'. Please choose either 'impute' or 'drop'.")

        print(f"\nBefore {strategy}: Train Set Rows: {len(self.train):,} | Missing Values: {self.train.isnull().sum().sum():,}")
        print(f"            : Test Set Rows:  {len(self.test):,} | Missing Values: {self.test.isnull().sum().sum():,}")

        do_domain_rules = (strategy == "impute") if apply_domain_rules is None else apply_domain_rules

        if strategy == "impute":

            self.impute_missing_values(simple_cols = simple_cols, knn_cols = knn_cols,
                                       apply_domain_rules = do_domain_rules, n_neighbors = n_neighbors)

        elif strategy == "drop":

            if do_domain_rules:
                self._impute_domain_rules()

            train_before = len(self.train)
            self.train = self.train.dropna(subset = subset).reset_index(drop = True)
            train_after = len(self.train)
            dropped = train_before - train_after
            print(f"Dropped {dropped:,} rows with missing values from train ({train_before:,} -> {train_after:,}).")

            # Impute remaining missing values in test set using train statistics (test rows cannot be dropped)
            test_missing = [c for c in self.test.columns if self.test[c].isnull().sum() > 0]
            if test_missing:
                self._impute_simple(cols = test_missing)

        print(f"\nAfter {strategy}: Train Set Rows: {len(self.train):,} | Missing Values: {self.train.isnull().sum().sum():,}")
        print(f"            : Test Set Rows:  {len(self.test):,} | Missing Values: {self.test.isnull().sum().sum():,}")


    def drop_high_cardinality_features(self, cols: list[str] | None = None) -> None:

        '''
        Removes high-cardinality identifier and raw metadata features that provide no direct predictive
        power after feature extraction and imputation:
        
            - PassengerId: Already used to derive Group_Id, Group_Size, and Is_Solo.
            - Cabin: Already decomposed into Cabin_Deck, Cabin_Num, and Cabin_Side.
            - Name: Passenger full name with high cardinality and no predictive signal.
            - Group_Id: Used for group-level imputation and Group_Size, no longer needed for training.

        Parameters
        ----------
         cols:list[str] | None
          List of column names to drop. If None, drops ['PassengerId', 'Cabin', 'Name', 'Group_Id']. [Default = None]

        Returns
        -------
         None
        '''

        if cols is None:
            cols = self.HIGH_CARDINALITY_COLS.copy()

        train_cols_to_drop = [c for c in cols if c in self.train.columns]
        test_cols_to_drop = [c for c in cols if c in self.test.columns]

        self.train.drop(columns = train_cols_to_drop, inplace = True)
        self.test.drop(columns = test_cols_to_drop, inplace = True)

        print(f"Removed high-cardinality features from train and test sets: {', '.join(train_cols_to_drop)}")


    def new_feature_introduction(self, introduce_features: bool = True, drop_original_amenities: bool = False, include_has_spent: bool = True) -> None:

        '''
        Introduces aggregated spending features across both train and test sets:

            - Total_Spending: Sum of all amenity expenditures (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck).

            - Has_Spent: Boolean indicator of whether total spending > 0 (primarily intended for tree-based models).

        Parameters
        ----------
         introduce_features:bool
          Whether to introduce aggregated spending features. When False, skips feature engineering [Default = True].

         drop_original_amenities:bool
          Whether to drop the individual spending features ('RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck') after aggregation [Default = False].

         include_has_spent:bool
          Whether to create the binary 'Has_Spent' indicator feature [Default = True].

        Returns
        -------
         None
        '''

        if not introduce_features:
            print("Skipped new feature introduction (retained original individual amenities only).")
            return

        spending_cols = self.SPENDING_COLS

        self.train["Total_Spending"] = self.train[spending_cols].sum(axis = 1)
        if include_has_spent:
            self.train["Has_Spent"] = self.train["Total_Spending"] > 0

        self.test["Total_Spending"] = self.test[spending_cols].sum(axis = 1)
        if include_has_spent:
            self.test["Has_Spent"] = self.test["Total_Spending"] > 0

        new_features = ["Total_Spending"] + (["Has_Spent"] if include_has_spent else [])
        print(f"Introduced new features: {', '.join(new_features)}")

        if drop_original_amenities:
            self.train.drop(columns = spending_cols, inplace = True)
            self.test.drop(columns = spending_cols, inplace = True)
            print(f"Dropped original amenity features: {', '.join(spending_cols)}")


    def handle_group_features(self, include_is_solo: bool = True) -> None:

        '''
        Resolves group-related features across train and test sets:
        retains Group_Size and optionally drops Is_Solo to avoid multicollinearity.

        Parameters
        ----------
         include_is_solo:bool
          Whether to retain the binary Is_Solo feature. When False, drops Is_Solo from train and test sets [Default = True].

        Returns
        -------
         None
        '''

        if not include_is_solo:
            for df in (self.train, self.test):
                if "Is_Solo" in df.columns:
                    df.drop(columns = ["Is_Solo"], inplace = True)

            print("Dropped redundant feature 'Is_Solo' (retained Group_Size only).")


    def handle_types(self, categorical_as_category: bool = False, custom_types: dict[str, Any] | None = None) -> None:

        '''
        Enforces proper data types across train and test sets:
        
            - Booleans (bool): CryoSleep, VIP, Transported, Is_Solo, Has_Spent.
            - Integers (int64): Age (rounded), Cabin_Num (rounded), Group_Size, Group_Id.
            - Floats (float64): RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Total_Spending.
            - Categoricals (str or category): HomePlanet, Destination, Cabin_Deck, Cabin_Side.

        Parameters
        ----------
         categorical_as_category:bool
          Whether to cast categorical features to pandas 'category' dtype instead of 'str'. [Default = False]

         custom_types:dict[str, Any] | None
          Optional dictionary mapping column names to custom dtypes to override defaults. [Default = None]

        Returns
        -------
         None
        '''

        bool_cols = self.BOOL_COLS
        int_cols = self.INT_COLS
        float_cols = self.FLOAT_COLS
        cat_cols = self.CAT_COLS

        for df in [self.train, self.test]:

            for col in bool_cols:
                if col in df.columns:
                    df[col] = df[col].astype("boolean") if df[col].isnull().any() else df[col].astype(bool)

            for col in int_cols:
                if col in df.columns:
                    rounded = df[col].round()
                    df[col] = rounded.astype("Int64") if rounded.isnull().any() else rounded.astype(int)

            for col in float_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype("category") if categorical_as_category else df[col].astype(str)

            if custom_types:
                for col, dtype in custom_types.items():
                    if col in df.columns:
                        df[col] = df[col].astype(dtype)

        print("Enforced proper data types across train and test sets:")
        print("  - Booleans (bool): " + ", ".join([c for c in bool_cols if c in self.train.columns]))
        print("  - Integers (int64): " + ", ".join([c for c in int_cols if c in self.train.columns]))
        print("  - Floats (float64): " + ", ".join([c for c in float_cols if c in self.train.columns]))
        cat_target = "category" if categorical_as_category else "str"
        print(f"  - Categoricals ({cat_target}): " + ", ".join([c for c in cat_cols if c in self.train.columns]))


    def _scale_numeric_features(self, cols: list[str]) -> None:

        '''
        Scales numerical features using RobustScaler fit exclusively on the train set.

        Parameters
        ----------
         cols:list[str]
          List of numerical column names to scale.

        Returns
        -------
         None
        '''

        if not cols:
            return

        self.scaler = RobustScaler()
        self.train[cols] = self.scaler.fit_transform(self.train[cols])
        self.test[cols] = self.scaler.transform(self.test[cols])
        self.scaled_numeric_cols = cols


    def _onehot_encode_features(self, cols: list[str]) -> None:

        '''
        One-hot encodes nominal categorical features across train and test sets without data leakage.

        Parameters
        ----------
         cols:list[str]
          List of categorical column names to one-hot encode.

        Returns
        -------
         None
        '''

        if not cols:
            return

        self.onehot_encoder = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)
        train_encoded = pd.DataFrame(
                                        np.asarray(self.onehot_encoder.fit_transform(self.train[cols])),
                                        columns = self.onehot_encoder.get_feature_names_out(cols),
                                        index = self.train.index
                                    )
        test_encoded = pd.DataFrame(
                                        np.asarray(self.onehot_encoder.transform(self.test[cols])),
                                        columns = self.onehot_encoder.get_feature_names_out(cols),
                                        index = self.test.index
                                    )

        self.train = pd.concat([self.train.drop(columns = cols), train_encoded], axis = 1)
        self.test = pd.concat([self.test.drop(columns = cols), test_encoded], axis = 1)
        self.encoded_onehot_cols = cols


    def _ordinal_encode_features(self, cols: list[str], categories: dict[str, list[str]] | None = None) -> None:

        '''
        Ordinally encodes hierarchical or binary categorical features across train and test sets.

        Parameters
        ----------
         cols:list[str]
          List of categorical column names to ordinally encode.

         categories:dict[str, list[str]] | None
          Optional dictionary mapping column names to ordered category lists.

        Returns
        -------
         None
        '''

        if not cols:
            return

        user_categories = categories or {}
        categories_list = []

        for col in cols:
        
            if col in user_categories:
                categories_list.append(user_categories[col])
        
            else:
                train_unique = sorted(self.train[col].dropna().unique().tolist())
                categories_list.append(train_unique)

        self.ordinal_encoder = OrdinalEncoder(categories = categories_list, handle_unknown = "use_encoded_value", unknown_value = -1)
        self.train[cols] = self.ordinal_encoder.fit_transform(self.train[cols])
        self.test[cols] = self.ordinal_encoder.transform(self.test[cols])
        self.encoded_ordinal_cols = cols


    def _encode_categorical_features(self, onehot_cols: list[str] | None = None, ordinal_cols: list[str] | None = None,
                                     ordinal_categories: dict[str, list[str]] | None = None) -> None:

        '''
        Encodes categorical features routing nominal features to OneHotEncoder and hierarchical/binary
        features to OrdinalEncoder.

        Parameters
        ----------
         onehot_cols:list[str] | None
          Categorical columns to one-hot encode.

         ordinal_cols:list[str] | None
          Categorical columns to ordinally encode.

         ordinal_categories:dict[str, list[str]] | None
          Ordered category mappings for ordinal features.

        Returns
        -------
         None
        '''

        if onehot_cols:
            self._onehot_encode_features(cols = onehot_cols)

        if ordinal_cols:
            self._ordinal_encode_features(cols = ordinal_cols, categories = ordinal_categories)


    def _encode_boolean_features(self, encode_target: bool = True) -> None:

        '''
        Converts boolean features (and optionally the target 'Transported') to 0/1 integers.

        Parameters
        ----------
         encode_target:bool
          Whether to cast the target 'Transported' in train to integer [Default = True].

        Returns
        -------
         None
        '''

        bool_cols = [
                        c for c in self.train.columns
                        if c != self.TARGET_COL and (self.train[c].dtype == bool or self.train[c].dtype == "boolean")
                    ]

        for col in bool_cols:
           
            self.train[col] = self.train[col].astype(int)
           
            if col in self.test.columns:
                self.test[col] = self.test[col].astype(int)

        if encode_target and self.TARGET_COL in self.train.columns:
            self.train[self.TARGET_COL] = self.train[self.TARGET_COL].astype(int)


    def _route_categorical_columns(self, encoding_strategy: str, onehot_cols: list[str] | None = None,
                                   ordinal_cols: list[str] | None = None) -> tuple[list[str], list[str]]:

        '''
        Resolves which categorical features should be one-hot encoded and which should be ordinally encoded.

        Parameters
        ----------
         encoding_strategy:str
          Encoding strategy ('auto', 'onehot', or 'ordinal').

         onehot_cols:list[str] | None
          Explicit list of one-hot columns, if any.

         ordinal_cols:list[str] | None
          Explicit list of ordinal columns, if any.

        Returns
        -------
         tuple[list[str], list[str]]
          A tuple containing (resolved_onehot_cols, resolved_ordinal_cols).
        '''

        all_cat_cols = [
                            c for c in self.train.columns
                            if c != self.TARGET_COL and (pd.api.types.is_string_dtype(self.train[c]) or isinstance(self.train[c].dtype, pd.CategoricalDtype))
                        ]

        resolved_onehot: list[str] = []
        resolved_ordinal: list[str] = []

        if encoding_strategy == "onehot":
            resolved_onehot = onehot_cols if onehot_cols is not None else all_cat_cols

        elif encoding_strategy == "ordinal":
            resolved_ordinal = ordinal_cols if ordinal_cols is not None else all_cat_cols

        elif encoding_strategy == "auto":
            resolved_onehot = [c for c in self.NOMINAL_CAT_COLS if c in self.train.columns] if onehot_cols is None else onehot_cols
            resolved_ordinal = [c for c in self.ORDINAL_CAT_COLS if c in self.train.columns] if ordinal_cols is None else ordinal_cols

        else:
            raise ValueError(f"Invalid encoding_strategy '{encoding_strategy}'. Please choose 'auto', 'onehot', or 'ordinal'.")

        return resolved_onehot, resolved_ordinal


    def scale_and_encode_features(self, numeric_cols: list[str] | None = None, onehot_cols: list[str] | None = None,
                                  ordinal_cols: list[str] | None = None, ordinal_categories: dict[str, list[str]] | None = None,
                                  encoding_strategy: Literal["auto", "onehot", "ordinal"] = "auto",
                                  cast_booleans: bool = True, encode_target: bool = True) -> None:

        '''
        Scales numerical features using RobustScaler and encodes categorical features across train and test sets,
        fitting strictly on the train set to prevent data leakage:

            1. Scales numerical features using RobustScaler (robust to extreme spending outliers).
            2. Encodes nominal categorical features (HomePlanet, Destination) using OneHotEncoder.
            3. Encodes hierarchical/binary features (Cabin_Deck, Cabin_Side) using OrdinalEncoder.
            4. Converts boolean features to 0/1 integers.

        Parameters
        ----------
         numeric_cols:list[str] | None
          Numerical features to scale. If None, automatically selects all numeric non-boolean features
          excluding the target 'Transported'. [Default = None]

         onehot_cols:list[str] | None
          Categorical features to one-hot encode. If None and encoding_strategy='auto', defaults to
          nominal features ['HomePlanet', 'Destination']. [Default = None]

         ordinal_cols:list[str] | None
          Categorical features to ordinally encode. If None and encoding_strategy='auto', defaults to
          hierarchical/binary features ['Cabin_Deck', 'Cabin_Side']. [Default = None]

         ordinal_categories:dict[str, list[str]] | None
          Custom ordered category mappings for ordinal features. Defaults to natural ship deck hierarchy
          and port/starboard: {'Cabin_Deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], 'Cabin_Side': ['P', 'S']}.

         encoding_strategy:Literal['auto', 'onehot', 'ordinal']
          Encoding routing strategy:
          - 'auto': OneHot for nominal (HomePlanet, Destination), Ordinal for Deck/Side.
          - 'onehot': OneHot encodes all categorical features.
          - 'ordinal': Ordinally encodes all categorical features. [Default = 'auto']

         cast_booleans:bool
          Whether to cast boolean features to binary 0/1 integers. [Default = True]

         encode_target:bool
          Whether to encode the target 'Transported' in the train set as 0/1 integers. [Default = True]

        Returns
        -------
         None
        '''

        # Determine numeric columns to scale
        if numeric_cols is None:
            numeric_cols = [
                                c for c in self.train.columns
                                if c != self.TARGET_COL and pd.api.types.is_numeric_dtype(self.train[c]) and self.train[c].dtype != bool
                            ]

        # Determine categorical columns routing
        onehot_cols, ordinal_cols = self._route_categorical_columns(encoding_strategy = encoding_strategy, onehot_cols = onehot_cols, ordinal_cols = ordinal_cols)

        default_ordinal_categories = self.DEFAULT_ORDINAL_CATEGORIES.copy()
        if ordinal_categories is not None:
            default_ordinal_categories.update(ordinal_categories)

        # Scale numerical features
        self._scale_numeric_features(cols = numeric_cols)

        # Encode categorical features
        self._encode_categorical_features(onehot_cols = onehot_cols, ordinal_cols = ordinal_cols, ordinal_categories = default_ordinal_categories)

        # Cast boolean and target features
        if cast_booleans:
            self._encode_boolean_features(encode_target = encode_target)

        print("Feature scaling and encoding complete:")
        if numeric_cols:
            print(f"  - Scaled with RobustScaler ({len(numeric_cols)} features): {', '.join(numeric_cols)}")
        else:
            print("  - Scaled with RobustScaler (0 features): Skipped (features left unscaled)")
        if onehot_cols:
            print(f"  - One-Hot Encoded ({len(onehot_cols)} features): {', '.join(onehot_cols)}")
        if ordinal_cols:
            print(f"  - Ordinally Encoded ({len(ordinal_cols)} features): {', '.join(ordinal_cols)}")
        print(f"  - Final Train shape: {self.train.shape} | Final Test shape: {self.test.shape}")


    def _format_features(self) -> tuple[str, str]:
        
        train_cols = ", ".join(self.train.columns)
        test_cols = ", ".join(self.test.columns)

        return train_cols, test_cols
    
    
    def print_stats(self):

        width = 230
        train_cols, test_cols = self._format_features()

        text = dedent(f"""
        +{"=" * (width - 2)}+
        |{" SpaceShip Titanic Dataset ".center(width - 2)}|
        +{"=" * (width - 2)}+
        | {f"Train Data Size: {len(self.train):,} rows":<{width - 4}} |
        | {f"Train Features ({len(self.train.columns)}): {train_cols}":<{width - 4}} |
        | {"":<{width - 4}} |
        | {f"Test Data Size:  {len(self.test):,} rows":<{width - 4}} |
        | {f"Test Features  ({len(self.test.columns)}): {test_cols}":<{width - 4}} |
        +{"=" * (width - 2)}+
        """).strip()

        print(text)


    @staticmethod
    def _process_split_target(split: str) -> str:

        split_lower = split.strip().lower()
        if split_lower not in ['train', 'test']:
            raise ValueError(f"Invalid split '{split}'. Please choose either 'train' or 'test'.")

        return split_lower


    @staticmethod
    def _process_feature(col: str, df: pd.DataFrame) -> dict:

        '''
        It collects information about a feature (i.e. its type, number of unique values, 
        missing values, etc...).

        Parameters
        ----------
         col:str
          The feature to process

         df:pd.DataFrame
          The dataframe to process
        
        Returns
        -------
         dict
          A dictionary containing information about the feature
        '''


        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        n_unique = df[col].nunique()
        non_null = df[col].dropna()
        sample_val = str(non_null.iloc[0]) if len(non_null) > 0 else 'N/A'
        if len(sample_val) > 25:
            sample_val = sample_val[:22] + '...'

        return {
                    'Feature': col,
                    'Dtype': str(df[col].dtype),
                    'Non-Null Count': f"{len(df) - null_count:,}",
                    'Missing Count': f"{null_count:,}",
                    'Missing (%)': f"{null_pct:.2f}%",
                    'Unique Values': f"{n_unique:,}",
                    'Sample Value': sample_val
                }


    def _collect_info_dataset(self, df: pd.DataFrame) -> pd.DataFrame:

        '''
        It collects information about each feature in the dataframe.

        Parameters
        ----------
         df:pd.DataFrame
          The dataframe to process
        
        Returns
        -------
         pd.DataFrame
          A dataframe containing information about each feature
        '''

        info_records = []

        for col in df.columns:
            info_records.append(self._process_feature(col, df))

        return pd.DataFrame(info_records)


    def _describe_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:

        '''
        It describes the numerical features in the dataframe.

        Parameters
        ----------
         df:pd.DataFrame
          The dataframe to process
        
        Returns
        -------
         pd.DataFrame
          A dataframe containing the description of the numerical features
        '''

        num_cols = df.select_dtypes(include = 'number').columns
        desc_df = df[num_cols].describe().T if len(num_cols) > 0 else pd.DataFrame()

        return desc_df


    @staticmethod
    def _print_data_info(name: str, df: pd.DataFrame, info_df: pd.DataFrame, desc_df: pd.DataFrame) -> None:

        '''
        It shows the data.

        Parameters
        ----------
         info_df:pd.DataFrame
          A dataframe containing information about each feature
         
         desc_df:pd.DataFrame
          A dataframe containing the description of the numerical features
        
        Returns
        -------
         None
        '''

        banner_width = 95
        print("\n" + "=" * banner_width)
        print(f" 🚀 {name.upper()} DATASET — INFO & STATISTICAL SUMMARY ".center(banner_width))
        print(f" Total Rows: {df.shape[0]:,}  |  Total Columns: {df.shape[1]} ".center(banner_width))
        print("=" * banner_width)
        print("\n\n")

        print("📋 Feature Overview & Missing Values".center(banner_width))
        print("\n")
        print(info_df.to_string(index = False))
        print("\n")

        if not desc_df.empty:

            print("\n")
            print("📊 Numerical Descriptive Statistics".center(banner_width))
            print("\n")
            print(desc_df.to_string())

        print("=" * banner_width + "\n")


    def show_info(self, split: str = 'train') -> None:

        '''
        Displays a combined summary of dataset info (data types, missing values, unique counts, sample values) 
        and numerical descriptive statistics.

        Parameters
        ----------
         split : str
          The dataset partition to display ('train' or 'test') [Default: 'train'].
        
        Returns
        -------
         None
        '''

        split_choice = self._process_split_target(split)
        df = self.train if split_choice == 'train' else self.test
        name = split_choice.capitalize()

        # Build Info Overview Table
        info_df = self._collect_info_dataset(df)
        
        # Build Numerical Description Table
        desc_df = self._describe_numerical_features(df)
        
        self._print_data_info(name, df, info_df, desc_df)