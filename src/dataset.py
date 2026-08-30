from textwrap import dedent
from pathlib import Path
import pandas as pd
import kagglehub


class SpaceShipDataset:

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
        df[["Cabin_Deck", "Cabin_Num", "Cabin_Side"]] = df["Cabin"].str.split("/", expand = True)
        df["Cabin_Num"] = pd.to_numeric(df["Cabin_Num"], errors="coerce")

        # Extract Group Size from PassengerId
        df["Group_Id"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
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