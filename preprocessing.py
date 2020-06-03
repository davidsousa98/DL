# Import libraries
import pandas as pd
import numpy as np
import zipfile as zp
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split

def get_files_zip(zip):
    """
        Function that lists files inside the .zip folders.

        :param zip: name of the zip file.
    """
    if zip == 'game_stats':
        with zp.ZipFile("./data/Game_stats.zip") as myzip:
            list = myzip.namelist()
    elif zip =='league_table':
        with zp.ZipFile("./data/league_table.zip") as myzip:
            list = myzip.namelist()
    return list

def append_files(files):
    """
        Function that reads and appends files inside the .zip folders.

        :param files: list of strings with paths to files.

        Returns:
            - DataFrame with all the files appended.
        """
    list_files = []
    for file in files:
        try:
            with zp.ZipFile("./data/Game_stats.zip") as myzip:
                with myzip.open(file) as myfile:
                    df = pd.read_csv(myfile)
                    list_files.append(df)

        except Exception:
            with zp.ZipFile("./data/league_table.zip") as myzip:
                with myzip.open(file) as myfile:
                    df = pd.read_excel(myfile)
                    list_files.append(df)

    df_appened = pd.concat(list_files, axis=0, sort=False)
    return df_appened.reset_index()


# Reading files from zip without extracting them
get_files_zip('game_stats')
get_files_zip('league_table')

# Append all files into one single DataFrame
df_stats_full = append_files(get_files_zip('game_stats'))
df_table_full = append_files(get_files_zip('league_table'))

# Slice DataFrames
df_stats = df_stats_full[['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HF',
                          'AF','HC','AC','HY','AY','HR','AR']]

df_table = df_table_full[['Squad', 'MP','GDiff', 'Pts', 'Season']]

###########################################  MISSING VALUES TREATMENT  #################################################

# Check for missing values
missings_stats = df_stats.isnull().sum()
missings_table = df_table.isnull().sum()

nulls_stats = df_stats.loc[df_stats.HF.isnull()]
nulls_table = df_table.loc[df_table.Squad.isnull()]

# Drop rows with exceeded number of nulls
df_stats = df_stats[df_stats['HomeTeam'].notna()]
df_stats = df_stats.loc[df_stats['HS'].notna()]

df_table = df_table[df_table['Squad'].notna()]

# Filling missing values based on the most similar matches
df_stats.reset_index(inplace=True, drop=True)
temp_df = df_stats.drop(columns=["Div", "Date", "HomeTeam", "AwayTeam"])
imputer = KNNImputer(n_neighbors=5)
filled_df = pd.DataFrame(imputer.fit_transform(temp_df))
filled_df.columns = temp_df.columns
filled_df = pd.concat([filled_df, df_stats["Div"], df_stats["Date"], df_stats["HomeTeam"], df_stats["AwayTeam"]], axis=1)

##################################  TRANSFORM AND CREATE VARIABLES  ####################################################

# Season variable
filled_df['year'] = pd.DatetimeIndex(filled_df['Date']).year

filled_df["month"] = filled_df["Date"].str.split("/", n=2, expand=True)[1]
filled_df['month'] = filled_df['month'].astype(int)

filled_df["season1"] = filled_df["season2"] = np.nan
filled_df.loc[(filled_df["month"] >= 7) & (filled_df["month"] <= 12), "season1"] = filled_df["year"]
filled_df.loc[(filled_df["month"] >= 7) & (filled_df["month"] <= 12), "season2"] = filled_df["year"]+1
filled_df.loc[(filled_df["month"] >= 1) & (filled_df["month"] <= 6), "season1"] = filled_df["year"]-1
filled_df.loc[(filled_df["month"] >= 1) & (filled_df["month"] <= 6), "season2"] = filled_df["year"]

filled_df['season1'] = filled_df['season1'].astype(str)
filled_df['season2'] = filled_df['season2'].astype(str)
filled_df['season1'] = filled_df['season1'].str[:4]
filled_df['season2'] = filled_df['season2'].str[2:4]

filled_df["Season"] = filled_df["season1"] + "/" + filled_df["season2"]
filled_df.drop(columns=["Date", "season1", "season2", "year", "month"], inplace=True)

# League variable
filled_df.loc[filled_df["Div"] == "D1", "League"] = "Bundesliga"
filled_df.loc[filled_df["Div"] == "B1", "League"] = "Jupiler"
filled_df.loc[filled_df["Div"] == "N1", "League"] = "Eredivisie"
filled_df.loc[filled_df["Div"] == "SP1", "League"] = "La Liga"
filled_df.loc[filled_df["Div"] == "D1", "League"] = "Bundesliga"
filled_df.loc[filled_df["Div"] == "E0", "League"] = "Premier League"
filled_df.loc[filled_df["Div"] == "F1", "League"] = "Ligue 1"
filled_df.loc[filled_df["Div"] == "G1", "League"] = "Super League"
filled_df.loc[filled_df["Div"] == "I1", "League"] = "Serie A"
filled_df.loc[filled_df["Div"] == "T1", "League"] = "Super Lig"
filled_df.loc[filled_df["Div"] == "P1", "League"] = "Liga NOS"
filled_df.drop(columns="Div", inplace=True)


# Rationing variables
df_table["points_per_game"] = df_table["Pts"] / df_table["MP"]
df_table["goaldiff_per_game"] = df_table["GDiff"] / df_table["MP"]
df_table.drop(columns=["Pts", "GDiff", "MP"], inplace=True)


# Groupby
df_grouped = df_stats.groupby('HomeTeam').count()


########################################## OUTLIERS RECOGNITION ########################################################

# Boxplot visualization
f, axes = plt.subplots(2, 5, figsize=(4, 2), squeeze=False)
sb.boxplot(df["First_Policy"], color="skyblue", ax=axes[0, 0])
sb.boxplot(df["Birthday"], color="olive", ax=axes[0, 1])
sb.boxplot(df["Salary"], color="gold", ax=axes[0, 2])
sb.boxplot(df["CMV"], whis=5, color="green", ax=axes[0, 3])
sb.boxplot(df["Claims"], color="yellow", ax=axes[0, 4])
sb.boxplot(df["Motor"], color="orange", ax=axes[1, 0])
sb.boxplot(df["Household"], whis=7, color="brown", ax=axes[1, 1])
sb.boxplot(df["Health"], whis=2.5, color="lavender", ax=axes[1, 2])
sb.boxplot(df["Life"], whis=7.5, color="greenyellow", ax=axes[1, 3])
sb.boxplot(df["Work_Compensations"], whis=7, color="m", ax=axes[1, 4])

# Histogram visualization
f, axes = plt.subplots(2, 5, figsize=(7, 7))
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
sb.distplot(df["First_Policy"], color="skyblue", ax=axes[0, 0], kde=False)
sb.distplot(df["Birthday"], color="olive", ax=axes[0, 1], kde=False)
sb.distplot(df["Salary"], color="gold", ax=axes[0, 2], kde=False)
sb.distplot(df["CMV"], color="green", ax=axes[0, 3], kde=False)
sb.distplot(df["Claims"], color="yellow", ax=axes[0, 4], kde=False)
sb.distplot(df["Motor"], color="orange", ax=axes[1, 0], kde=False)
sb.distplot(df["Household"], color="brown", ax=axes[1, 1], kde=False)
sb.distplot(df["Health"], color="lavender", ax=axes[1, 2], kde=False)
sb.distplot(df["Life"], color="greenyellow", ax=axes[1, 3], kde=False)
sb.distplot(df["Work_Compensations"], color="m", ax=axes[1, 4], kde=False)

df['Outlier'] = 0
df.loc[df['First_Policy'] > 2020, 'Outlier'] = 1

df['Outlier'].value_counts()

df = df.loc[df['Outlier'] == 0]

# Correlation analysis
plt.rcParams['figure.figsize'] = (12,12)

corr_matrix=df.corr(method='????')
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(data=corr_matrix, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
plt.tight_layout()



variables = ['Severity', 'Age', 'Male', 'Santa Fe', 'Family_cases', 'Parents_infected_bin']

# Data partition
test_df = df.loc[df["Season"] == "2019/20"]

# note: exclude season 19/20 from train/val
X = df[variables]
y = df['Deceased']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=15, shuffle=True)












