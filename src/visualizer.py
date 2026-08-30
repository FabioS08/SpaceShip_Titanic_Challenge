import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd


# Define some visualization parameters
sns.set_theme(style = "whitegrid", palette = "muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlepad"] = 20
plt.rcParams["axes.labelsize"] = 12


class DatasetVisualizer:

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
        
    
    def _plot_missing_values(self, missing_df: pd.DataFrame):

        fig, ax = plt.subplots()
        sns.barplot(
                        x = missing_df.index,
                        y = missing_df["Percentage (%)"],
                        hue = missing_df.index,
                        palette = "viridis",
                        legend = False,
                        ax = ax,  # type: ignore
                    )
        plt.title("Missing Values per Feature (% of Total)")
        plt.xticks(rotation = 45, ha = "right")
        plt.ylabel("Missing (%)")
        plt.xlabel("")

        # Add percentage values above each bar
        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt = "%.2f%%", padding = 3, fontsize = 9)
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