###########################################################  SAMPLE  ###################################################

# Import libraries
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import zipfile as zp
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_files_zip(zip):
    """
        Function that lists files inside the .zip folders.

        :param zip: name of the zip file.
    """
    if zip == 'game_stats':
        with zp.ZipFile(r"./data/Game_stats.zip") as myzip:
            list = myzip.namelist()
    elif zip =='league_table':
        with zp.ZipFile(r"./data/league_table.zip") as myzip:
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

####################################################### EXPLORE  #######################################################

# Slice DataFrames
df_stats = df_stats_full[['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HF',
                          'AF','HC','AC','HY','AY','HR','AR']]

df_table = df_table_full[['Squad', 'MP','GDiff', 'Pts', 'Season']]

# Missing values treatment
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

###########################################  DATA PREPARATION  #########################################################

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
# Groupby HomeTeam and AwayTeam
df_home = filled_df.groupby(['HomeTeam', 'Season']).mean()
df_away = filled_df.groupby(['AwayTeam', 'Season']).mean()

#Change the names for concatenation purposes
df_away_copy = df_away.copy()
df_away_copy.columns = ['FTAG','FTHG','HTAG','HTHG','AS','HS','AST','HST','AF','HF','AC','HC',
                        'AY','HY','AR','HR']


#Average Home and Away games to simplify data and have info by game
df_join = pd.concat([df_home,df_away_copy], axis = 1)
df_join = (df_join.groupby(lambda x:x, axis=1).sum())/2


# Add the column League to the new df
filled_df_sliced = filled_df[['HomeTeam','League']].drop_duplicates()
df_join = df_join.reset_index(level=[0,1])
df_join = pd.merge(df_join,filled_df_sliced,on = 'HomeTeam')


# Ensure that the clubs have the same name in both DataFrames
df_home = filled_df.groupby(['HomeTeam', 'Season']).mean().reset_index()
df_away = filled_df.groupby(['AwayTeam', 'Season']).mean().reset_index()

# Check Team Names
# df_table['Team_Name'] = df_table['Squad'] + "/" + df_table['Season']
# df_home['Team_Name'] = df_home['HomeTeam'] + "/" + df_home['Season']
# df_away['Team_Name'] = df_away['AwayTeam'] + "/" + df_away['Season']

table_team_names = df_table.Squad.unique()
home_team_names = df_home.HomeTeam.unique()
table_team_names_df = pd.DataFrame(sorted(table_team_names), columns=['table_names'])
home_team_names_df = pd.DataFrame(sorted(home_team_names), columns=['home_names'])

diff_names = []
for team in table_team_names:
    if team not in home_team_names:
        diff_names.append(team)
diff_names = pd.DataFrame(sorted(diff_names))


pair_names = pd.read_excel(r'./data/diff_names.xlsx', header=None)
pair_names.columns = ["Squad", "Team"]

df_table = df_table.merge(pair_names, on="Squad", how="left")

df_table.loc[df_table["Team"].isnull(), "Team"] = df_table["Squad"]
df_table.drop(columns='Squad', inplace=True)

df_join.rename(columns={"HomeTeam": "Team"}, inplace=True)
df = df_join.merge(df_table, how='inner', on=['Team', 'Season'])

df = df[["Team", "Season", "League", "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC",
         "HF", "AF", "HY", "AY", "HR", "AR", "goaldiff_per_game", "points_per_game"]]

df.columns = ["Team", "Season", "League", "Goals", "Goals_against", "Halftime_goals", "Halftime_goals_against", "Shots",
              "Shots_against", "Shots_target", "Shots_target_against", "Corners", "Corners_against",
              "Fouls", "Fouls_against", "Yellow", "Yellow_against", "Red", "Red_against", "Goal_diff", "Points"]

# Create train and test sets
df_test = df.loc[df["Season"] == "2019/20"].copy()
df_train = df.loc[df["Season"] != "2019/20"].copy()

# Obtain insights from the data
insights = df_train.describe()

# Outliers Recognition
# Boxplot visualization
f, axes = plt.subplots(6, 3, figsize=(12, 10))
sb.boxplot(df_train["Goals"], ax=axes[0, 0])
sb.boxplot(df_train["Goals_against"], ax=axes[0, 1])
sb.boxplot(df_train["Halftime_goals"], ax=axes[0, 2])
sb.boxplot(df_train["Halftime_goals_against"], ax=axes[1, 0])
sb.boxplot(df_train["Shots"], ax=axes[1, 1])
sb.boxplot(df_train["Shots_against"], ax=axes[1, 2])
sb.boxplot(df_train["Shots_target"], ax=axes[2, 0])
sb.boxplot(df_train["Shots_target_against"], ax=axes[2, 1])
sb.boxplot(df_train["Corners"], ax=axes[2, 2])
sb.boxplot(df_train["Corners_against"], ax=axes[3, 0])
sb.boxplot(df_train["Fouls"], ax=axes[3, 1])
sb.boxplot(df_train["Fouls_against"], ax=axes[3, 2])
sb.boxplot(df_train["Yellow"], ax=axes[4, 0])
sb.boxplot(df_train["Yellow_against"], ax=axes[4, 1])
sb.boxplot(df_train["Red"], ax=axes[4, 2])
sb.boxplot(df_train["Red_against"], ax=axes[5, 0])
sb.boxplot(df_train["Goal_diff"], ax=axes[5, 1])
sb.boxplot(df_train["Points"], ax=axes[5, 2])
plt.tight_layout()


# Histogram visualization
f, axes = plt.subplots(6, 3, figsize=(12, 10))
sb.distplot(df_train["Goals"], ax=axes[0, 0], kde=True)
sb.distplot(df_train["Goals_against"], ax=axes[0, 1], kde=True)
sb.distplot(df_train["Halftime_goals"], ax=axes[0, 2], kde=True)
sb.distplot(df_train["Halftime_goals_against"], ax=axes[1, 0], kde=True)
sb.distplot(df_train["Shots"], ax=axes[1, 1], kde=True)
sb.distplot(df_train["Shots_against"], ax=axes[1, 2], kde=True)
sb.distplot(df_train["Shots_target"], ax=axes[2, 0], kde=True)
sb.distplot(df_train["Shots_target_against"], ax=axes[2, 1], kde=True)
sb.distplot(df_train["Corners"], ax=axes[2, 2], kde=True)
sb.distplot(df_train["Corners_against"], ax=axes[3, 0], kde=True)
sb.distplot(df_train["Fouls"], ax=axes[3, 1], kde=True)
sb.distplot(df_train["Fouls_against"], ax=axes[3, 2], kde=True)
sb.distplot(df_train["Yellow"], ax=axes[4, 0], kde=True)
sb.distplot(df_train["Yellow_against"], ax=axes[4, 1], kde=True)
sb.distplot(df_train["Red"], ax=axes[4, 2], kde=True)
sb.distplot(df_train["Red_against"], ax=axes[5, 0], kde=True)
sb.distplot(df_train["Goal_diff"], ax=axes[5, 1], kde=True)
sb.distplot(df_train["Points"], ax=axes[5, 2], kde=True)
plt.tight_layout()

####################################################### MODIFY #########################################################

# Correlation analysis between original variables
def correlation_matrix(df):
    plt.rcParams['figure.figsize'] = (12,12)
    corr_matrix = df.corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sb.heatmap(data=corr_matrix, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
    plt.tight_layout()

correlation_matrix(df_train)

# Drop variable 'Goal diff' since it is very correlated w/ Points
df_train = df_train.drop(columns='Goal_diff')
df_test = df_test.drop(columns='Goal_diff')

# Transform and create variables
# Metrics per goals
df_train['Shots_p/goal'] = df_train['Shots']/df_train['Goals']
df_train['Shots_p/goal_against'] = df_train['Shots_against']/df_train['Goals_against']
df_test['Shots_p/goal'] = df_test['Shots']/df_test['Goals']
df_test['Shots_p/goal_against'] = df_test['Shots_against']/df_test['Goals_against']

df_train['Corners_p/goal'] = df_train['Corners']/df_train['Goals']
df_train['Corners_p/goal_against'] = df_train['Corners_against']/df_train['Goals_against']
df_test['Corners_p/goal'] = df_test['Corners']/df_test['Goals']
df_test['Corners_p/goal_against'] = df_test['Corners_against']/df_test['Goals_against']

df_train['Goals_1stHalf'] = df_train['Halftime_goals']/df_train['Goals']
df_test['Goals_1stHalf'] = df_test['Halftime_goals']/df_test['Goals']

# Metrics per fouls and cards
df_train['Total_cards'] = df_train['Yellow'] + df_train['Red']
df_train['Total_cards_against'] = df_train['Yellow_against'] + df_train['Red_against']
df_test['Total_cards'] = df_test['Yellow'] + df_test['Red']
df_test['Total_cards_against'] = df_test['Yellow_against'] + df_test['Red_against']

df_train['Danger_fouls'] = df_train['Total_cards']/df_train['Fouls']
df_test['Danger_fouls'] = df_test['Total_cards']/df_test['Fouls']

# Metrics per shots
df_train['Shots_precision'] = df_train['Shots_target']/df_train['Shots']
df_train['Shots_precision_against'] = df_train['Shots_target_against']/df_train['Shots_against']
df_test['Shots_precision'] = df_test['Shots_target']/df_test['Shots']
df_test['Shots_precision_against'] = df_test['Shots_target_against']/df_test['Shots_against']

# Correlation analysis between transformed variables
correlation_matrix(df_train)


# Data Standardization
X_train = df_train.drop(columns=['Points', 'Team', 'Season', 'League'])
y_train = df_train['Points']

X_test = df_test.drop(columns=['Points', 'Team', 'Season', 'League'])
y_test = df_test['Points']

scaler = StandardScaler().fit(X_train)
scaler_X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
scaler_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Feature Selection
variables = ['Goals_against','Corners_against','Corners_p/goal','Shots_p/goal','Shots_target_against','Fouls',
             'Shots_precision_against','Shots','Total_cards_against','Shots_target','Corners','Corners_p/goal_against', 'Corners_p/goal']

scaler_X_train = scaler_X_train[variables]
scaler_X_test = scaler_X_test[variables]

# Lasso Regression
def plot_importance(coef,name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(8,10))
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using " + name + " Model")
    plt.show()

reg = LassoCV()
reg.fit(scaler_X_train, y_train)
coef = pd.Series(reg.coef_, index=scaler_X_train.columns)
coef.sort_values()
plot_importance(coef, 'Lasso')

# Ridge Regression
ridge = RidgeCV()
ridge.fit(X=scaler_X_train, y=y_train)
coef_ridge = pd.Series(ridge.coef_, index=scaler_X_train.columns)
print(coef_ridge.sort_values())
plot_importance(coef_ridge,'Ridge')

# Correlation after feature selection
correlation_matrix(scaler_X_train)