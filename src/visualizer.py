from sklearn.preprocessing import StandardScaler
from matplotlib.container import BarContainer
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import umap


# Define some visualization parameters
sns.set_theme(style = "whitegrid", palette = "muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlepad"] = 20
plt.rcParams["axes.labelsize"] = 12


class DatasetVisualizer:

    # Non-numeric columns excluded from categorical feature analysis (identifiers, high-cardinality text or target)
    EXCLUDED_CATEGORICAL_COLS = ["PassengerId", "Name", "Cabin", "Transported"]

    # Spending features
    SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    # Cabin deck order from top to bottom of ship
    DECK_ORDER = ["A", "B", "C", "D", "E", "F", "G", "T"]

    def __init__(self, df: pd.DataFrame):
        self.df = df 

    
    def _create_missing_values_df(self) -> pd.DataFrame:

        '''

        It creates a dataframe containing the percentage of missing values in the dataset.

        Returns
        -------
         missing_df:pd.DataFrame
          A dataframe containing the percentage of missing values in the dataset.
        '''

        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100

        missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pct})
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by = "Missing Count", ascending = False)

        return missing_df


    @staticmethod
    def _add_bar_labels(ax: plt.Axes, fmt: str | None = None, padding: int = 3, fontsize: int = 9) -> None:

        '''
        It adds text labels to all BarContainers in a given matplotlib Axes.

        Parameters
        ----------
         ax:plt.Axes
          The matplotlib Axes object containing bar containers.

         fmt:str | None
          String formatting code (e.g. "%.2f", "%.2f%%", "$%.0f") or None to display raw values [Default = None].

         padding:int
          Distance between the label and the bar edge in points [Default = 3].

         fontsize:int
          Font size of the label text [Default = 9].

        Returns
        -------
         None
        '''

        for container in ax.containers:

            if isinstance(container, BarContainer):

                if fmt is not None:
                    ax.bar_label(container, fmt = fmt, padding = padding, fontsize = fontsize)
                else:
                    ax.bar_label(container, padding = padding, fontsize = fontsize)


    def _plot_missing_values(self, missing_df: pd.DataFrame):

        fig, ax = plt.subplots()
        sns.barplot(x = missing_df.index, y = missing_df["Percentage (%)"], hue = missing_df.index, palette = "viridis", legend = False, ax = ax) 
        plt.title("Missing Values per Feature (% of Total)")
        plt.xticks(rotation = 45, ha = "right")
        plt.ylabel("Missing (%)")
        plt.xlabel("")

        # Add percentage values above each bar
        self._add_bar_labels(ax, fmt = "%.2f%%", padding = 3, fontsize = 9)
        ax.margins(y = 0.15)

        plt.tight_layout()
        plt.show()


    def visualize_missing_values(self):
        
        '''
        It generates a plot containing the percentage of missing values in the dataset.
        '''

        missing_df = self._create_missing_values_df()
        self._plot_missing_values(missing_df)


    def visualize_target_balance(self):

        '''
        It generates a plot illustrating the balance of the target values.
        '''

        # Define the colours for the two labels
        color_map = {False: "#e74c3c", True: "#2ecc71"}
        fig, axes = plt.subplots(1, 2)

        # Bar count
        sns.countplot(data = self.df, x = "Transported", palette = color_map, hue = "Transported", legend = False, ax = axes[0])
        axes[0].set_title("Target Distribution (Counts)")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Number of People")
        axes[0].set_xticks([])

        for p in axes[0].patches:

            height = p.get_height()
            axes[0].annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2.0, height / 2.0), ha = "center", va = "center",
                             color = "white", weight = "bold", fontsize = 11)

        # Pie chart with consistent colors matching value keys
        counts = self.df["Transported"].value_counts()
        pie_colors = [color_map[val] for val in counts.index]

        axes[1].pie(counts.values, autopct = "%1.1f%%", colors = pie_colors, explode = [0.03, 0.03], startangle = 90,
                    textprops = {"color": "white", "fontsize": 11})
        axes[1].set_ylabel("")
        axes[1].set_title("Target Proportion (%)")

        # Single shared legend
        legend_elements = [Patch(facecolor = color_map[False], label = "False"),
                           Patch(facecolor = color_map[True], label = "True")]
        fig.legend(handles = legend_elements, title = "Transported", loc = "lower center", bbox_to_anchor = (0.5, 1.05), ncol = 2)

        plt.tight_layout()
        plt.show()


    def _get_categorical_columns(self, max_cardinality: int = 10) -> list[str]:
        
        '''
        It returns categorical features suitable for plotting (excluding identifiers, text, and target).

        Parameters
        ----------
         max_cardinality:int
          The maximum number of unique values a feature can have to be considered categorical

        Returns
        -------
         list[str]
          A list of categorical features
        '''
        
        return [
                    col for col in self.df.select_dtypes(exclude = "number").columns
                    if col not in self.EXCLUDED_CATEGORICAL_COLS and self.df[col].nunique() <= max_cardinality
                ]


    def _create_dynamic_subplots(self, n_items: int, n_cols: int = 2, col_width: float = 8.0, row_height: float = 4.5,
                                 figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, np.ndarray]:
            
            '''
            It creates a dynamic grid of subplots based on the number of items.
            
            Parameters
            ----------
             n_items:int
              The number of items to plot
             
             n_cols:int
              The number of columns
             
             col_width:float
              The width of a column
             
             row_height:float
              The height of a row
             
             figsize:tuple[float, float] | None
              The size of the figure
            
            Returns
            -------
             tuple[plt.Figure, np.ndarray]
              A tuple containing the figure and the axes
            '''
            
            n_cols = min(n_cols, n_items)
            n_rows = (n_items + n_cols - 1) // n_cols

            if figsize is None:
                figsize = (col_width * n_cols, row_height * n_rows)

            fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize)
            axes = np.atleast_1d(axes).flatten()

            # Clean up any extra/unused subplot slots from the figure
            for j in range(n_items, len(axes)):
                fig.delaxes(axes[j])

            return fig, axes[:n_items]


    def visualize_categorical_vs_target(self):

        '''
        It generates a plot illustrating the distribution of categorical features vs the target variable.
        '''

        categorical_cols = self._get_categorical_columns()

        if not categorical_cols:

            print("No categorical columns found.")
            return

        fig, axes = self._create_dynamic_subplots(n_items = len(categorical_cols), n_cols = 2, col_width = 8.0, row_height = 4.5)

        # Plot each categorical feature
        for i, col in enumerate(categorical_cols):

            sns.countplot(data = self.df, x = col, hue = "Transported", palette = ["#e74c3c", "#2ecc71"], ax = axes[i])
            axes[i].set_title(f"{col} vs Transported")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Passenger Count")
            axes[i].legend(title = "Transported", loc = "upper right")

        plt.tight_layout()
        plt.show()


    def visualize_categorical_transport_rates(self):

        '''
        It generates bar plots showing the proportion (rate) of transported passengers for each category across categorical features.
        '''

        categorical_cols = self._get_categorical_columns()
        
        if not categorical_cols:

            print("No categorical columns found.")
            return

        fig, axes = self._create_dynamic_subplots(n_items = len(categorical_cols), n_cols = 2, col_width = 8, row_height = 4.5)

        for i, col in enumerate(categorical_cols):

            rate_df = self.df.groupby(col, observed = False)["Transported"].mean().reset_index()
            sns.barplot(data = rate_df, x = col, y = "Transported", hue = "Transported", palette = "Blues", hue_norm = (0, 1), legend = False, ax = axes[i])
            axes[i].set_title(f"{col} Transport Rate", weight = "bold")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Transported Rate")
            axes[i].set_ylim(0, 1.05)
            axes[i].axhline(0.5, color = "red", linestyle = "--", alpha = 0.6)
            self._add_bar_labels(axes[i], fmt = "%.2f", padding = 3, fontsize = 10)

        plt.tight_layout()
        plt.show()


    def visualize_age_distribution(self):

        '''
        It generates a plot showing the Age distribution (KDE / Histogram) by Transported status,
        along with the transported rate broken down by age groups.
        '''

        fig, axes = plt.subplots(1, 2, figsize = (16, 5))

        # KDE / Histogram
        sns.histplot(data = self.df, x = "Age", hue = "Transported", kde = True, bins = 30, palette = ["#e74c3c", "#2ecc71"], ax = axes[0])
        axes[0].set_title("Age Distribution by Transported Status")
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Passenger Count")

        # Transport rate by Age Groups
        age_groups = pd.cut(
            self.df["Age"],
            bins = [-1, 12, 18, 25, 35, 50, 65, 100],
            labels = ["0-12 (Child)", "13-18 (Teen)", "19-25 (Young Adult)", "26-35 (Adult)", "36-50 (Middle-Age)", "51-65 (Mature)", "65+ (Senior)"]
        )
        age_rate = self.df.groupby(age_groups, observed = False)["Transported"].mean().reset_index()
        age_rate.columns = ["Age_Group", "Transported"]

        sns.barplot(data = age_rate, x = "Age_Group", y = "Transported", hue = "Age_Group", palette = "coolwarm", dodge = False, legend = False, ax = axes[1])
        axes[1].set_title("Transported Rate by Age Group")
        axes[1].set_xlabel("Age Group")
        axes[1].set_ylabel("Transported Rate")
        axes[1].set_ylim(0, 1.05)
        axes[1].axhline(0.5, color = "red", linestyle = "--", alpha = 0.6)
        plt.setp(axes[1].get_xticklabels(), rotation = 35, ha = "right")
        self._add_bar_labels(axes[1], fmt = "%.2f", padding = 3, fontsize = 9)

        plt.tight_layout()
        plt.show()


    def _prepare_spending_data(self) -> tuple[pd.DataFrame, list[str]]:

        '''
        It prepares a copy of the dataframe with Total_Spending and Zero_Spending status computed,
        along with the ordered list of all spending-related features to visualize.

        Parameters
        ----------
         None

        Returns
        -------
         spending_df:pd.DataFrame
          A copy of the dataframe containing computed Total_Spending and Zero_Spending features.

         all_features:list[str]
          Ordered list of features: Zero_Spending, Total_Spending, and individual amenities.
        '''

        spending_df = self.df.copy()

        # Compute Total_Spending if not present
        if "Total_Spending" not in spending_df.columns:
            spending_df["Total_Spending"] = spending_df[self.SPENDING_COLS].sum(axis = 1)

        # Compute Zero_Spending status
        spending_df["Zero_Spending"] = spending_df["Total_Spending"] == 0

        all_features = ["Zero_Spending", "Total_Spending"] + self.SPENDING_COLS

        return spending_df, all_features


    def visualize_spending_vs_target(self, plot_type: str = "bar", showfliers: bool = False):

        '''
        It generates plots comparing spending features against the target (Transported), including:

            - Zero Spending status (Spent Money vs Zero Spending)
            - Total Spending
            - Individual amenity expenses (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)

        Parameters
        ----------
         plot_type:str
          Type of plot for continuous spending amounts: 'bar' for mean expenditure or 'box' for boxplots [Default = "bar"].

         showfliers:bool
          Whether to show outlier points when plot_type = 'box' [Default = False].
        '''

        temp_df, all_features = self._prepare_spending_data()
        fig, axes = self._create_dynamic_subplots(n_items = len(all_features), n_cols = 2, col_width = 8, row_height = 4.5)

        # Zero Spending countplot
        sns.countplot(data = temp_df, x = "Zero_Spending", hue = "Transported", palette = ["#e74c3c", "#2ecc71"], ax = axes[0])
        axes[0].set_title("Zero Spending vs Transported")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Passenger Count")
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(["Spent Money (> 0)", "Zero Spending (= 0)"])
        self._add_bar_labels(axes[0], padding = 3, fontsize = 9)

        # Spending features (Total_Spending and individual amenities)
        for i, col in enumerate(all_features[1:]):
            ax = axes[i + 1]

            if plot_type == "box":

                sns.boxplot(data = temp_df, x = "Transported", y = col, hue = "Transported", palette = ["#e74c3c", "#2ecc71"], showfliers = showfliers, 
                            legend = False, ax = ax)
                ax.set_title(f"{col} vs Transported")
                ax.set_ylabel("Amount ($)")
            
            else:

                sns.barplot(data = temp_df, x = "Transported", y = col, hue = "Transported", palette = ["#e74c3c", "#2ecc71"], legend = False, ax = ax)
                ax.set_title(f"Mean {col}")
                ax.set_ylabel("Average Amount ($)")
                self._add_bar_labels(ax, fmt = "$%.0f", padding = 3, fontsize = 9)

            ax.set_xlabel("Transported")

        plt.tight_layout()
        plt.show()


    def visualize_correlation_matrix(self, cols: list[str] | None = None, figsize: tuple[float, float] = (10, 8)):

        '''
        It generates a heatmap showing the Pearson correlation matrix between numerical features and the target.

        Parameters
        ----------
         cols:list[str] | None
          List of numerical feature names to include in the correlation matrix. If None, defaults to primary numerical and spending features [Default = None].

         figsize:tuple[float, float]
          Size of the heatmap figure [Default = (10, 8)].

        Returns
        -------
         None
        '''

        temp_df, _ = self._prepare_spending_data()

        if cols is None:
            cols = ["Age"] + self.SPENDING_COLS + ["Total_Spending", "Zero_Spending", "Group_Size", "Transported"]

        cols = [col for col in cols if col in temp_df.columns]
        corr_matrix = temp_df[cols].astype(float).corr()

        plt.figure(figsize = figsize)
        sns.heatmap(corr_matrix, annot = True, fmt = ".2f", cmap = "coolwarm", center = 0, square = True, linewidths = 0.5)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()


    def visualize_cabin_spatial_heatmap(self, bins: list[int] | None = None, labels: list[str] | None = None, figsize: tuple[float, float] = (18, 5)):

        '''
        It generates side-by-side heatmaps showing the Transported rate across the ship's 3D spatial layout:
        Decks (A through T) vs longitudinal Ship Sections (Bow to Stern), split by Port (P) and Starboard (S) sides.

        Parameters
        ----------
         bins:list[int] | None
          Thresholds to bin cabin numbers into longitudinal sections [Default = [0, 300, 600, 900, 1200, 1500, 2000]].

         labels:list[str] | None
          Labels corresponding to the cabin number bins [Default = ["0-300 (Bow)", "300-600", "600-900 (Mid)", "900-1200", "1200-1500", "1500+ (Stern)"]].

         figsize:tuple[float, float]
          Size of the heatmap figure [Default = (18, 5)].

        Returns
        -------
         None
        '''

        temp_df = self.df.copy()

        if bins is None:
            bins = [0, 300, 600, 900, 1200, 1500, 2000]

        if labels is None:
            labels = ["0-300 (Bow)", "300-600", "600-900 (Mid)", "900-1200", "1200-1500", "1500+ (Stern)"]

        temp_df["Ship_Section"] = pd.cut(temp_df["Cabin_Num"], bins = bins, labels = labels, include_lowest = True)

        fig, axes = self._create_dynamic_subplots(n_items = 2, n_cols = 2, col_width = 9, row_height = 5, figsize = figsize)
        sides = [("P", "Port Side (P)"), ("S", "Starboard Side (S)")]

        for i, (side, side_label) in enumerate(sides):
            
            pivot = temp_df[temp_df["Cabin_Side"] == side].pivot_table(
                index = "Cabin_Deck", columns = "Ship_Section", values = "Transported", aggfunc = "mean", observed = False
            ).reindex(self.DECK_ORDER)

            sns.heatmap(pivot, annot = True, fmt = ".2f", cmap = "RdYlGn", center = 0.5, vmin = 0.2, vmax = 0.9, ax = axes[i])
            axes[i].set_title(f"{side_label}: Transported Rate Heatmap")
            axes[i].set_xlabel("Ship Section (Cabin Num)")
            axes[i].set_ylabel("Deck" if i == 0 else "")

        plt.tight_layout()
        plt.show()


    def _prepare_reduction_matrix(self) -> tuple[np.ndarray, pd.Series]:

        '''
        It prepares and standardizes the feature matrix for dimensionality reduction algorithms.

        Parameters
        ----------
         None

        Returns
        -------
         X_scaled:np.ndarray
          Standardized feature matrix combining imputed numerical and one-hot encoded categorical features.

         y:pd.Series
          The binary target variable (Transported).
        '''

        df, _ = self._prepare_spending_data()

        # Numerical features
        num_cols = ["Age", "CryoSleep", "VIP", "Cabin_Num", "Group_Size", "Is_Solo", "Zero_Spending"] + self.SPENDING_COLS + ["Total_Spending"]
        num_cols = [c for c in num_cols if c in df.columns]
        num_df = df[num_cols].copy()

        for col in num_cols:
            num_df[col] = pd.to_numeric(num_df[col], errors = "coerce")
            num_df[col] = num_df[col].fillna(num_df[col].median())

        # Categorical features
        cat_cols = [c for c in ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_Side"] if c in df.columns]
        cat_df = df[cat_cols].fillna("Unknown")
        cat_encoded = pd.get_dummies(cat_df, drop_first = True, dtype = float)

        X = pd.concat([num_df, cat_encoded], axis = 1)
        X_scaled = StandardScaler().fit_transform(X)
        y = df["Transported"]

        return X_scaled, y


    @staticmethod
    def _sample_reduction_data(X: np.ndarray, y: pd.Series, sample_size: int | None = None, random_state: int = 42) -> tuple[np.ndarray, pd.Series]:

        '''
        It optionally draws a random subset of feature samples and target labels for faster computation.

        Parameters
        ----------
         X:np.ndarray
          The feature matrix.

         y:pd.Series
          The corresponding target labels.

         sample_size:int | None
          Number of samples to draw. If None or greater than the number of samples, returns the original data [Default = None].

         random_state:int
          Random seed for reproducibility [Default = 42].

        Returns
        -------
         X_sampled:np.ndarray
          Sampled feature matrix.

         y_sampled:pd.Series
          Sampled target labels.
        '''

        if sample_size is not None and sample_size < len(X):

            np.random.seed(random_state)
            indices = np.random.choice(len(X), size = sample_size, replace = False)
            return X[indices], y.iloc[indices]

        return X, y


    @staticmethod
    def _compute_3d_pca(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, list[str], str]:

        '''
        It computes a 3D Principal Component Analysis (PCA) projection of the feature matrix.

        Parameters
        ----------
         X:np.ndarray
          Standardized feature matrix.

         random_state:int
          Random seed for reproducibility [Default = 42].

        Returns
        -------
         coords:np.ndarray
          3D PCA coordinates (shape: [N, 3]).

         dim_labels:list[str]
          Labels for the 3 dimensions including individual explained variance percentages.

         title:str
          Plot title with total explained variance and sample count.
        '''

        reducer = PCA(n_components = 3, random_state = random_state)
        coords = reducer.fit_transform(X)
        ev = reducer.explained_variance_ratio_ * 100
        dim_labels = [f"PC1 ({ev[0]:.1f}%)", f"PC2 ({ev[1]:.1f}%)", f"PC3 ({ev[2]:.1f}%)"]
        title = f"3D PCA Projection by Transported Status\n(Total Explained Variance: {ev.sum():.1f}%, N={len(X):,})"

        return coords, dim_labels, title


    @staticmethod
    def _compute_3d_tsne(X: np.ndarray, perplexity: float = 30.0, max_iter: int = 500, random_state: int = 42) -> tuple[np.ndarray, list[str], str]:

        '''
        It computes a 3D t-Distributed Stochastic Neighbor Embedding (t-SNE) manifold projection.

        Parameters
        ----------
         X:np.ndarray
          Standardized feature matrix.

         perplexity:float
          Perplexity parameter for t-SNE [Default = 30.0].

         max_iter:int
          Maximum iterations for optimization [Default = 500].

         random_state:int
          Random seed for reproducibility [Default = 42].

        Returns
        -------
         coords:np.ndarray
          3D t-SNE coordinates (shape: [N, 3]).

         dim_labels:list[str]
          Dimension labels: ["t-SNE 1", "t-SNE 2", "t-SNE 3"].

         title:str
          Plot title including sample count.
        '''

        reducer = TSNE(n_components = 3, perplexity = perplexity, random_state = random_state, init = "pca", learning_rate = "auto", max_iter = max_iter)
        coords = reducer.fit_transform(X)
        dim_labels = ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
        title = f"3D t-SNE Manifold Projection by Transported Status (N={len(X):,})"

        return coords, dim_labels, title


    @staticmethod
    def _compute_3d_umap(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, list[str], str]:

        '''
        It computes a 3D Uniform Manifold Approximation and Projection (UMAP).

        Parameters
        ----------
         X:np.ndarray
          Standardized feature matrix.

         random_state:int
          Random seed for reproducibility [Default = 42].

        Returns
        -------
         coords:np.ndarray
          3D UMAP coordinates (shape: [N, 3]).

         dim_labels:list[str]
          Dimension labels: ["UMAP 1", "UMAP 2", "UMAP 3"].

         title:str
          Plot title including sample count.
        '''

        reducer = umap.UMAP(n_components = 3, random_state = random_state, n_jobs = 1)
        coords = reducer.fit_transform(X)
        dim_labels = ["UMAP 1", "UMAP 2", "UMAP 3"]
        title = f"3D UMAP Projection by Transported Status (N={len(X):,})"

        return coords, dim_labels, title


    @staticmethod
    def _plot_3d_scatter(coords: np.ndarray, y: pd.Series, dim_labels: list[str], title: str, width: int = 950, height: int = 700, alpha: float = 0.7, 
                         s: float = 3.5) -> None:

        '''
        It renders an interactive Plotly 3D scatter plot of the projected coordinates colored by target status.

        Parameters
        ----------
         coords:np.ndarray
          Projected 3D coordinates (shape: [N, 3]).

         y:pd.Series
          The target labels (Transported status).

         dim_labels:list[str]
          Labels for the 3 axes.

         title:str
          Title of the plot.

         width:int
          Width of the plot in pixels [Default = 950].

         height:int
          Height of the plot in pixels [Default = 700].

         alpha:float
          Point transparency [Default = 0.7].

         s:float
          Point marker size [Default = 3.5].

        Returns
        -------
         None
        '''

        fig = go.Figure()

        colors = {False: "#e74c3c", True: "#2ecc71"}
        labels_map = {False: "Not Transported (False)", True: "Transported (True)"}

        for target_val in [False, True]:

            mask = (y.values == target_val)
            fig.add_trace(
                go.Scatter3d(
                    x = coords[mask, 0], y = coords[mask, 1], z = coords[mask, 2],
                    mode = "markers",
                    marker = dict(size = s, color = colors[target_val], opacity = alpha, line = dict(width = 0)),
                    name = labels_map[target_val],
                    hovertemplate = f"<b>{labels_map[target_val]}</b><br>{dim_labels[0]}: %{{x:.2f}}<br>{dim_labels[1]}: %{{y:.2f}}<br>{dim_labels[2]}: %{{z:.2f}}<extra></extra>"
                )
            )

        fig.update_layout(
            title = dict(text = title.replace("\n", "<br>"), x = 0.5, font = dict(size = 14, family = "sans-serif")),
            scene = dict(
                xaxis = dict(title = dim_labels[0], backgroundcolor = "rgb(245, 245, 245)", gridcolor = "white"),
                yaxis = dict(title = dim_labels[1], backgroundcolor = "rgb(245, 245, 245)", gridcolor = "white"),
                zaxis = dict(title = dim_labels[2], backgroundcolor = "rgb(245, 245, 245)", gridcolor = "white"),
            ),
            legend = dict(
                title = dict(text = "Transported"), yanchor = "top", y = 0.95, xanchor = "right", x = 0.95, bgcolor = "rgba(255, 255, 255, 0.8)",
                bordercolor = "rgba(0, 0, 0, 0.1)", borderwidth = 1
            ),
            width = width,
            height = height,
            margin = dict(l = 0, r = 0, b = 0, t = 50),
            template = "plotly_white"
        )

        fig.show()


    def visualize_3d_projection(self, method: str = "pca", sample_size: int | None = None, random_state: int = 42, perplexity: float = 30.0, 
                                max_iter: int = 500, width: int = 950, height: int = 700, alpha: float = 0.7, s: float = 3.5):

        '''
        It projects the passenger feature space into 3D using a dimensionality reduction algorithm
        (PCA, t-SNE, or UMAP) and displays an interactive Plotly 3D scatter plot colored by Transported status.

        Parameters
        ----------
         method:str
          Dimensionality reduction algorithm: 'pca' (or 'pcs'), 'tsne', or 'umap' [Default = "pca"].

         sample_size:int | None
          Optional number of rows to randomly sample for faster calculation and clearer 3D display. If None, uses all rows [Default = None].

         random_state:int
          Random seed for reproducibility [Default = 42].

         perplexity:float
          Perplexity parameter used when method='tsne' [Default = 30.0].

         max_iter:int
          Maximum iterations when method='tsne' [Default = 500].

         width:int
          Width of the interactive Plotly 3D plot in pixels [Default = 950].

         height:int
          Height of the interactive Plotly 3D plot in pixels [Default = 700].

         alpha:float
          Point transparency [Default = 0.7].

         s:float
          Point marker size [Default = 3.5].

        Returns
        -------
         None
        '''

        X_scaled, y = self._prepare_reduction_matrix()

        # Optional sampling
        X_scaled, y = self._sample_reduction_data(X_scaled, y, sample_size = sample_size, random_state = random_state)

        n_samples = len(X_scaled)
        method_norm = method.lower().strip()

        if method_norm in ["pca", "pcs"]:
            coords, dim_labels, title = self._compute_3d_pca(X_scaled, random_state = random_state)

        elif method_norm in ["tsne", "t-sne"]:
            coords, dim_labels, title = self._compute_3d_tsne(X_scaled, perplexity = perplexity, max_iter = max_iter, random_state = random_state)

        elif method_norm == "umap":
            coords, dim_labels, title = self._compute_3d_umap(X_scaled, random_state = random_state)

        else:
            raise ValueError(f"Unsupported dimensionality reduction method '{method}'. Choose from: 'pca' (or 'pcs'), 'tsne', or 'umap'.")

        # Render interactive Plotly 3D scatter plot
        self._plot_3d_scatter(coords = coords, y = y, dim_labels = dim_labels, title = title, width = width, height = height, alpha = alpha, s = s)