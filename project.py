#############################################   SAMPLE  ################################################################
import matplotlib
# matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import zipfile as zp
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sb
from itertools import combinations_with_replacement
import keras
from keras import layers, models
from keras.layers import Dropout, Dense, TimeDistributed, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import xarray as xr


def get_files_zip(zip):
    """
    Function that lists files inside the .zip folders.

    :param zip: name of the zip file.

    """
    if zip == 'game_stats':
        with zp.ZipFile(r"./data/Game_stats.zip") as myzip:
            list = myzip.namelist()
    elif zip == 'league_table':
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
                    df_csv = pd.read_csv(myfile)
                    list_files.append(df_csv)

        except Exception:
            with zp.ZipFile("./data/league_table.zip") as myzip:
                with myzip.open(file) as myfile:
                    df_excel = pd.read_excel(myfile)
                    list_files.append(df_excel)

    df_append = pd.concat(list_files, axis=0, sort=False)
    return df_append.reset_index()


# Reading files from zip without extracting them
get_files_zip('game_stats')
get_files_zip('league_table')

# Append all files into one single DataFrame
df_stats_full = append_files(get_files_zip('game_stats'))
df_table_full = append_files(get_files_zip('league_table'))

###############################################   EXPLORE  #############################################################


# Slice DataFrames
df_stats = df_stats_full[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST',
                          'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]

df_table = df_table_full[['Squad', 'MP', 'GDiff', 'Pts', 'Season']]

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

## Data Preparation
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

# Change the names for concatenation purposes
df_away_copy = df_away.copy()
df_away_copy.columns = ['FTAG', 'FTHG', 'HTAG', 'HTHG', 'AS', 'HS', 'AST', 'HST', 'AF', 'HF', 'AC', 'HC',
                        'AY', 'HY', 'AR', 'HR']


# Average Home and Away games to simplify data and have info by game
df_join = pd.concat([df_home, df_away_copy], axis=1)
df_join = (df_join.groupby(lambda x: x, axis=1).sum())/2


# Add the column League to the new df
filled_df_sliced = filled_df[['HomeTeam', 'League']].drop_duplicates()
df_join = df_join.reset_index(level=[0, 1])
df_join = pd.merge(df_join, filled_df_sliced, on='HomeTeam')


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

# See which team names are spelled differently
diff_names = []
for team in table_team_names:
    if team not in home_team_names:
        diff_names.append(team)
diff_names = pd.DataFrame(sorted(diff_names))

# Save pair of names into excel file
pair_names = pd.read_excel(r'./data/diff_names.xlsx', header=None)
pair_names.columns = ["Squad", "Team"]

# Modify and ensure that all team names are equal
df_table = df_table.merge(pair_names, on="Squad", how="left")
df_table.loc[df_table["Team"].isnull(), "Team"] = df_table["Squad"]
df_table.drop(columns='Squad', inplace=True)
df_join.rename(columns={"HomeTeam": "Team"}, inplace=True)
df = df_join.merge(df_table, how='inner', on=['Team', 'Season'])

# Change variable names for more intuitive labels
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

###############################################  MODIFY  ###############################################################


def correlation_matrix(df):
    """
    Function to create and plot correlation matrix between variablles.

    :param df: dataframe to assess the correlations.

    """

    plt.rcParams['figure.figsize'] = (12, 12)
    corr_matrix = df.corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sb.heatmap(data=corr_matrix, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
    plt.tight_layout()


# Plot correlation matrix
correlation_matrix(df_train)

# Drop variable 'Goal diff' since it is very correlated w/ Points
df_train = df_train.drop(columns='Goal_diff')
df_test = df_test.drop(columns='Goal_diff')

## Transform and create variables
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

# League
df_train['League_quality'] = 3
df_train.loc[(df['League'] == 'Eredivisie') | (df['League'] == 'Liga NOS'), 'League_quality'] = 2
df_train.loc[(df['League'] == 'Super League') | (df['League'] == 'Jupiler') | (df['League'] == 'Super Lig'), 'League_quality'] = 1
df_test['League_quality'] = 3
df_test.loc[(df_test['League'] == 'Eredivisie') | (df_test['League'] == 'Liga NOS'), 'League_quality'] = 2
df_test.loc[(df_test['League'] == 'Super League') | (df_test['League'] == 'Jupiler') | (df_test['League'] == 'Super Lig'), 'League_quality'] = 1


# Correlation analysis between transformed variables
correlation_matrix(df_train)

# Data Standardization
X_train = df_train.drop(columns=['Points', 'Team', 'Season', 'League'])
y_train = df_train['Points']

df_league_games = df_test[['League', 'Team']]
df_league_games = df_league_games.set_index('Team')
X_test = df_test.drop(columns=['Points', 'Season', 'League']).set_index('Team')
y_test = df_test['Points']

## Feature Selection
# Select varibles to use in the model
variables = ['Goals', 'Corners_p/goal_against', 'Corners', 'Shots_target', 'Total_cards_against',
             'Shots_precision_against', 'Fouls', 'Shots_p/goal', 'Shots_target_against', 'Corners_p/goal',
             'Corners_against', 'Goals_against']

scaler = StandardScaler().fit(X_train[variables])
scaler_X_train = pd.DataFrame(scaler.transform(X_train[variables]), columns=X_train[variables].columns)
scaler_X_test = pd.DataFrame(scaler.transform(X_test[variables]), columns=X_test[variables].columns)
scaler_X_train = scaler_X_train[variables]
scaler_X_test = scaler_X_test[variables]


def plot_importance(coeff, name):
    """
    Function to plot variables importance based on regression coefficients.

    :param coeff: regression coefficients.
    :param name: regression name.

    """
    imp_coef = coeff.sort_values()
    plt.figure(figsize=(8, 10))
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using " + name + " regression")
    plt.show()


# Ridge Regression
ridge = RidgeCV()
ridge.fit(X=scaler_X_train, y=y_train)
coef_ridge = pd.Series(ridge.coef_, index=scaler_X_train.columns)
print(coef_ridge.sort_values())
plot_importance(coef_ridge, 'Ridge')

# Lasso Regression
reg = LassoCV()
reg.fit(scaler_X_train, y_train)
coef = pd.Series(reg.coef_, index=scaler_X_train.columns)
coef.sort_values()
plot_importance(coef, 'Lasso')

# Correlation after feature selection
correlation_matrix(scaler_X_train)


## Set up environment with fixed seeds to ensure reproducibility of results
# https://stackoverflow.com/questions/59075244/if-keras-results-are-not-reproducible-whats-the-best-practice-for-comparing-mo

# Seed value
seed_value = 0

# Set the `PYTHONHASHSEED` environment variable at a fixed value and deactivate the GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Function to set random seed as a constant
import tensorflow as tf
import random


def reset_seeds(reset_graph_with_backend=None):
    """
    Function to set fixed seed for python libraries.

    """
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()

    np.random.seed(1) #numpy seed
    random.seed(2) #random seed
    tf.compat.v1.set_random_seed(3) #tensorflow seed

#############################################   MODEL - PART 1  ########################################################


## DENSELY CONNECTED LAYERS
# Define model for GridSearch
def build_model_grid(dense_layer_sizes, regularizers, initializer, activation='relu', optimizer='RMSprop'):
    reset_seeds()
    model = models.Sequential()
    model.add(layers.Dense(dense_layer_sizes[0], activation=activation, kernel_regularizer=regularizers,
                           kernel_initializer=initializer, input_shape=(scaler_X_train.shape[1],)))

    for units in dense_layer_sizes[1:]:  #loop to create hidden layers
        model.add(layers.Dense(units, activation=activation, kernel_regularizer=regularizers,
                               kernel_initializer=initializer))

    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def combination_layers(min_neurons, max_neurons, n_layers):
    """
    Function that creates all possible combinations of units for a given range and layer size.

    :param min_neurons: minimum number of units of the network.
    :param max_neurons: maximum number of units of the network.
    :param n_layers: number of layers.

    """
    l = []
    for i in range(min_neurons,max_neurons):
        l.append(i)
    layersize = list(combinations_with_replacement(l,n_layers))
    return layersize

## Grid Search
k = 5
cv = KFold(n_splits=k, shuffle=True, random_state=15)
Keras_estimator = KerasRegressor(build_fn=build_model_grid)

# Define grid of parameters to be tested
param_grid = {
    'epochs': [25, 50],
    'activation': ['relu', 'elu', 'selu', 'tanh', 'sigmoid'],
    'dense_layer_sizes': combination_layers(10, 100, 2), #all combinations of 10-100 neurons for a hidden layer of size 2
    'regularizers': ['l1', 'l2', 'l1_l2'],
    'initializer': ['random_normal', 'identity', 'constant', 'random_uniform'],
    'optimizer': ['RMSprop', 'Adam', 'sgd', 'Adadelta']
}

grid = GridSearchCV(estimator=Keras_estimator, param_grid=param_grid, n_jobs=-1, cv=cv,
                    scoring='neg_mean_absolute_error', return_train_score=True, verbose=1)
grid_result = grid.fit(scaler_X_train, y_train)


# Summary of results
print('Mean test score: {}'.format(np.mean(grid.cv_results_['mean_test_score'])))
print('Mean train score: {}'.format(np.mean(grid.cv_results_['mean_train_score'])))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# Export results to excel for later comparation
def save_excel(dataframe, sheetname, filename="gridresults"):
    """
    Function that saves the evaluation metrics into a excel file.

    :param dataframe: dataframe that contains the metrics.
    :param sheetname: name of the sheet containing the parameterization.
    :param filename: specifies the name of the xlsx file.

    """
    if not os.path.isfile("./data/outputs/{}.xlsx".format(filename)):
        mode = 'w'
    else:
        mode = 'a'
    writer = pd.ExcelWriter('./data/outputs/{}.xlsx'.format(filename), engine='openpyxl', mode=mode)
    dataframe.to_excel(writer, sheet_name=sheetname)
    writer.save()
    writer.close()


gridresults = pd.DataFrame(grid_result.cv_results_)
save_excel(gridresults, "model_v0")


## Test final models
# Define model
def build_model():
    reset_seeds()
    model = models.Sequential()
    model.add(layers.Dense(28, activation='selu', kernel_regularizer='l2', kernel_initializer='random_normal',
                           input_shape=(scaler_X_train.shape[1],)))
    model.add(layers.Dense(42, activation='selu', kernel_regularizer='l2', kernel_initializer='random_normal'))
    model.add(layers.Dense(1))
    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    return model


# Define callbacks
callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_mae', save_best_only=True),
                  keras.callbacks.EarlyStopping(monitor='val_mae', mode='min', patience=7)]

# Define number of epochs
num_epochs = 100
y_train = y_train.to_numpy()

# Cross validation for final model
cvscores_val = []
rsquare_val = []
for train, val in cv.split(scaler_X_train, y_train):

    model = build_model()

    history = model.fit(scaler_X_train.loc[train], y_train[train], validation_data=(scaler_X_train.loc[val], y_train[val]),
                        epochs=num_epochs, verbose=0, callbacks=callbacks_list)

    # Evaluate the model
    scores_val = model.evaluate(scaler_X_train.loc[val], y_train[val], verbose=0)
    labels_val = model.predict(scaler_X_train.loc[val])
    rsquare_val.append(r2_score(y_train[val], labels_val) * 100)
    print("%s: %.2f%%" % (model.metrics_names[1], scores_val[1]*100))
    print("R-squared: %.2f%%" % (r2_score(y_train[val], labels_val) * 100))
    cvscores_val.append(scores_val[1] * 100)

# Summary of results
print("MAPE validation score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_val), np.std(cvscores_val)))
print("R-Squared validation score: %.2f%% (+/- %.2f%%)" % (np.mean(rsquare_val), np.std(rsquare_val)))

# Plot train and validation error
plt.clf()
history_dict = history.history
mae_values = history_dict['mae']
val_mae_values = history_dict['val_mae']
epochs = range(0, len(history_dict['mae']))

plt.plot(epochs[2:], mae_values[2:], 'bo', label='Training mae')
plt.plot(epochs[2:], val_mae_values[2:], 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

## Predict the output
scores_test = model.evaluate(scaler_X_test, y_test, verbose=0)
print("Test MAPE: %.2f%%" % (scores_test[1]*100))
labels_test = model.predict(scaler_X_test)

# Number of Matches Played per League
matches_dict = {'Liga NOS': 34,
                'Super League': 30,
                'Eredivisie': 34,
                'La Liga': 38,
                'Ligue 1': 38,
                'Premier League': 38,
                'Serie A': 38,
                'Bundesliga': 34,
                'Jupiler': 40,
                'Super Lig': 34}

# Create DataFrame with predictions and other relevant information
games = pd.DataFrame(matches_dict.values(), columns=['Matches_Played'])
games['League'] = matches_dict.keys()
final_classification = df_league_games.reset_index().merge(games, how='left', on=['League'])
final_classification['points_per_game'] = labels_test
final_classification['Points'] = round(final_classification['points_per_game'] * final_classification['Matches_Played'])
final_classification = final_classification.sort_values(by='points_per_game', ascending=False)
final_classification.reset_index(inplace=True, drop=True)
final_classification.index += 1
final_classification = final_classification[['Team', 'League', 'points_per_game', 'Matches_Played', 'Points']]
final_classification.columns = ['Team', 'League', 'Points per game', 'Matches Played', 'Points']

# Classification table for 'Liga NOS'
liganos = final_classification.loc[final_classification['League'] == 'Liga NOS']
liganos.reset_index(inplace=True, drop=True)
liganos.index += 1

# Classification table for 'Serie A'
seriea = final_classification.loc[final_classification['League'] == 'Serie A']
seriea.reset_index(inplace=True, drop=True)
seriea.index += 1

# Classification table for 'Premier League'
premierleague = final_classification.loc[final_classification['League'] == 'Premier League']
premierleague.reset_index(inplace=True, drop=True)
premierleague.index += 1

#############################################   MODEL - PART 2  ########################################################

## LONG SHORT-TERM MEMORY (LSTM)
# Data preprocessing
df_lstm = df_train.copy()
df_lstm_1920 = df_test.copy()

# Selecting Observations
df_lstm_2 = df_lstm[['Team', 'Season']].copy()
df_lstm_2['18_19'] = 0
df_lstm_2.loc[df_lstm_2['Season'] == '2018/19', '18_19'] = 1
df_lstm_2['17_18'] = 0
df_lstm_2.loc[df_lstm_2['Season'] == '2017/18', '17_18'] = 1
df_lstm_2 = df_lstm_2.groupby(['Team']).sum()[['18_19', '17_18']]
df_lstm_2 = df_lstm_2.loc[(df_lstm_2['18_19'] == 1) & (df_lstm_2['17_18'] == 1)].reset_index()

df_lstm_1920 = df_lstm_1920.loc[df_lstm_1920['Team'].isin(list(df_lstm_2.Team.unique()))]
df_lstm = df_lstm.loc[(df_lstm['Team'].isin(list(df_lstm_1920.Team.unique()))) &
                      (df_lstm['Season'].isin(['2018/19', '2017/18']))]

# Create MultiIndex and concatenate seasons
df_lstm.set_index(['Team', 'Season'], inplace=True)
df_lstm_1920.set_index(['Team', 'Season'], inplace=True)

df_lstm = pd.concat([df_lstm, df_lstm_1920]).sort_index()

# Setting dependent and independent variables & Train, Test dataset splits
X_lstm = df_lstm.drop(columns=['Points'])
y_lstm = df_lstm[['Points']]

X_lstm_train = X_lstm[114:]
X_lstm_val = X_lstm[:114]
y_lstm_train = y_lstm[114:]
y_lstm_val = y_lstm[:114]

# Feature Selection
variables_lstm = ['Goals', 'Corners_p/goal_against', 'Corners', 'Shots_target', 'Total_cards_against',
                  'Shots_precision_against', 'Fouls', 'Shots_p/goal', 'Shots_target_against', 'Corners_p/goal',
                  'Corners_against', 'Goals_against']

X_lstm = X_lstm[variables_lstm]
X_lstm_train = X_lstm_train[variables_lstm]
X_lstm_val = X_lstm_val[variables_lstm]

# Data Standardization
scaler = StandardScaler().fit(X_lstm[variables_lstm])

scaler_X_lstm = pd.DataFrame(scaler.transform(X_lstm[variables_lstm]),
                             columns=X_lstm[variables_lstm].columns, index=X_lstm.index)

scaler_X_lstm_train = pd.DataFrame(scaler.transform(X_lstm_train[variables_lstm]),
                                   columns=X_lstm_train[variables_lstm].columns, index=X_lstm_train.index)

scaler_X_lstm_val = pd.DataFrame(scaler.transform(X_lstm_val[variables_lstm]),
                                 columns=X_lstm_val[variables_lstm].columns, index=X_lstm_val.index)

scaler_X_lstm = scaler_X_lstm[variables_lstm]
scaler_X_lstm_train = scaler_X_lstm_train[variables_lstm]
scaler_X_lstm_val = scaler_X_lstm_val[variables_lstm]


scaler_X_lstm = np.array(scaler_X_lstm).reshape(128, 3, 12)
y_lstm = np.array(y_lstm).reshape(128, 3, 1)

scaler_X_lstm_train = np.array(scaler_X_lstm_train).reshape(90, 3, 12)
y_lstm_train = np.array(y_lstm_train).reshape(90, 3, 1)

scaler_X_lstm_val = np.array(scaler_X_lstm_val).reshape(38, 3, 12)
y_lstm_val = np.array(y_lstm_val).reshape(38, 3, 1)


## Create LSTM configurations
reset_seeds()  # guarantee reproducibility
model = models.Sequential()
model.add(LSTM(150, input_shape=(3, len(variables)), return_sequences= True, activation="selu"))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='Adam')
print(model.summary())

# Train LSTM
model.fit(scaler_X_lstm_train, y_lstm_train, epochs=100, verbose=2)

# Model Evaluation
scores_lstm_val = model.evaluate(scaler_X_lstm_val, y_lstm_val, verbose=0)
print(scores_lstm_val)

# Fit to all the model
model.fit(scaler_X_lstm, y_lstm)

## Predict Season 2019/20
lstm_pred = model.predict(scaler_X_lstm)

cols_names = ['Team', 'Season', 'e']
index = pd.MultiIndex.from_product([range(x) for x in lstm_pred.shape], names=cols_names)
lstm_pred = pd.DataFrame({'Points per Game': lstm_pred.flatten()}, index=index)['Points per Game'].reset_index().drop(['e'], axis = 1)
lstm_pred['Team'] = X_lstm.reset_index()['Team']
lstm_pred['Season'] = X_lstm.reset_index()['Season']

lstm_pred_1920 = lstm_pred.loc[lstm_pred['Season'] == '2019/20']
lstm_pred_1920 = lstm_pred_1920.sort_values(by=['Points per Game'], ascending=False).drop(['Season'], axis=1)
lstm_pred_1819 = lstm_pred.loc[lstm_pred['Season'] == '2018/19']
lstm_pred_1819 = lstm_pred_1819.sort_values(by=['Points per Game'], ascending=False).drop(['Season'], axis=1)
lstm_pred_1718 = lstm_pred.loc[lstm_pred['Season'] == '2017/18']
lstm_pred_1718 = lstm_pred_1718.sort_values(by=['Points per Game'], ascending=False).drop(['Season'], axis=1)

###############################################  OTHER VISUALIZATIONS  #################################################


# import plotly.offline as pyo
# from plotly import graph_objs as go
# from plotly.subplots import make_subplots

# Plot train and validation MAE
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(len(dict_history["epoch"])-2)]
#
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
#
# result_graph = pd.DataFrame([['1 layer', 'MAPE', np.mean(cvscores_val)]], columns=['Topology', 'Metric', 'Value'])
# result_graph = result_graph.append(pd.DataFrame([['1 layer', 'R-Squared', np.mean(rsquare_val)]],
#                                                 columns=['Topology', 'Metric', 'Value']))
#
# result_graph = result_graph.append(pd.DataFrame([['2 layers', 'MAPE', np.mean(cvscores_val)]],
#                                                 columns=['Topology', 'Metric', 'Value']))
#
# result_graph = result_graph.append(pd.DataFrame([['2 layers', 'R-Squared', np.mean(rsquare_val)]],
#                                                 columns=['Topology', 'Metric', 'Value']))
#
# result_graph = result_graph.append(pd.DataFrame([['3 layer', 'MAPE', np.mean(cvscores_val)]],
#                                                 columns=['Topology', 'Metric', 'Value']))
#
# result_graph = result_graph.append(pd.DataFrame([['3 layer', 'R-Squared', np.mean(rsquare_val)]],
#                                                 columns=['Topology', 'Metric', 'Value']))
# result_graph = result_graph.round(2)

# fig = make_subplots(specs=[[{"secondary_y": True}]])
#
# fig.add_trace(
#     go.Bar(
#         x=result_graph.loc[result_graph['Metric'] == 'MAPE']["Topology"],
#         y=result_graph.loc[result_graph['Metric'] == 'MAPE']["Value"], name="MAPE",
#         text=result_graph.loc[result_graph['Metric'] == 'MAPE']["Value"], textposition='outside', offsetgroup=0),
#         secondary_y=False
#     )
#
# fig.add_trace(
#     go.Bar(
#         x=result_graph.loc[result_graph['Metric'] == 'R-Squared']["Topology"],
#         y=result_graph.loc[result_graph['Metric'] == 'R-Squared']["Value"], name="R-Squared",
#         text=result_graph.loc[result_graph['Metric'] == 'R-Squared']["Value"], textposition='outside', offsetgroup=1),
#          secondary_y=True
#     )
#
# fig.update_yaxes(title_text="Mean Absolute Percentage Error (MAPE)", range=[0, 50], secondary_y=False)
# fig.update_yaxes(title_text="R-Squared (%)", range=[50, 100], secondary_y=True)
# fig.update_layout(title='Topology comparison', xaxis_title="Topology", barmode='group')
# pyo.plot(fig)

# reg_graph = pd.DataFrame()
# reg_graph['l2'] = history_dict['val_mae']
# reg_graph['l1'] = history_dict['val_mae']
# reg_graph['l1_l2'] = history_dict['val_mae']

# plt.plot(epochs[5:], reg_graph['l2'][5:], 'b', label='l2', color='blue')
# plt.plot(epochs[5:], reg_graph['l1'][5:], 'b', label='l1', color='red')
# plt.plot(epochs[5:], reg_graph['l1_l2'][5:], 'b', label='l1_l2', color='black')
# plt.title('Validation MAE - Regularizers')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.show()






