########################################################################################################################
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,KFold, cross_val_score
from itertools import combinations_with_replacement
from keras.layers import Dropout
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras import layers, models, regularizers


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

def combination_layers(min_neurons,max_neurons,n_layers):
    l = []
    for i in range(min_neurons,max_neurons):
        l.append(i)
    layersize = list(combinations_with_replacement(l,n_layers))
    return layersize


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

X_test = df_test.drop(columns=['Points', 'Team', 'Season', 'League'])
y_test = df_test['Points']

scaler = StandardScaler().fit(X_train)
scaler_X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
scaler_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Feature Selection
variables = ['Goals_against','Corners_against','Shots_p/goal','Shots_target_against','Fouls','Shots_precision_against',
             'Shots','Total_cards_against','Shots_target','Corners','Corners_p/goal_against', 'Corners_p/goal']

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
#########################################   SET A ENVIRONMENT FOR RANDOM STATE  ########################################
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
# for later versions:
tf.compat.v2.random.set_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


################################################################## MODEL ###############################################
# https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras link for random state keras

#Define model
def build_model_grid(dense_layer_sizes,activation = 'relu', optimizer = 'RMSprop'):#dropout = 0.2
    model = models.Sequential()
    model.add(layers.Dense(dense_layer_sizes[0], activation=activation, input_shape=(scaler_X_train.shape[1],)))
    # model.add(Dropout(dropout))
    for units in dense_layer_sizes[1:]:
        model.add(layers.Dense(units, activation=activation))
        # model.add(Dropout(dropout), )
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_mae', patience=5),
                  keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_mae', save_best_only=True)]


#Grid Search
k = 5
cv = KFold(n_splits=k, shuffle=True, random_state=15)
Keras_estimator = KerasRegressor(build_fn=build_model_grid)


param_grid = {
    'epochs': [10, 25],#50
    # 'activation': ['relu', 'tanh','sigmoid'], #linear,hard_sigmoid,softmax,softplus,softsign
    'dense_layer_sizes': combination_layers(30,31,1), #(32,32,), (64, 64,)],
    # 'dense_nparams': [32, 64, 72, 128, 154],
    # 'kernel_initializer': ['uniform', 'zeros', 'normal'], #lecun_uniform,glorot_normal,glorot_uniform,he_normal, he_uniform
    # 'batch_size':[2, 16, 32],
    'optimizer':['RMSprop', 'Adam', 'sgd'],#Adagrad, Nadam, Adadelta,'Adamax'
    # 'dropout': [0.5, 0.4, 0.3, 0.2]
}


grid = GridSearchCV(estimator=Keras_estimator, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error',
                    return_train_score = True)
grid_result = grid.fit(scaler_X_train, y_train)


# Summary of results
print('Mean test score: {}'.format(np.mean(grid.cv_results_['mean_test_score'])))
print('Mean train score: {}'.format(np.mean(grid.cv_results_['mean_train_score'])))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

gs_results  = pd.DataFrame(grid_result.cv_results_)

# Define model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(72, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), input_shape=(scaler_X_train.shape[1],)))
    # model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(1))
    # model.add(Dropout(0.2))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_mae', patience=5),
                  keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_mae', save_best_only=True)]

y_train = y_train.to_numpy()
# fix random seed for reproducibility
seed = 15
np.random.seed(seed)

# Set number of folds and epochs
k = 5
num_epochs = 100

kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
#cvscores_train = []
cvscores_val = []
for train, val in kfold.split(scaler_X_train, y_train):
    # Create model
    model = build_model()
	# Fit the model
    history = model.fit(scaler_X_train.loc[train], y_train[train], validation_data=(scaler_X_train.loc[val], y_train[val]),
                        callbacks=callbacks_list, epochs=num_epochs, verbose=0)
	# Evaluate the model
#    scores_train = model.evaluate(scaler_X_train.loc[train], y_train[train], verbose=0)
	scores_val = model.evaluate(scaler_X_train.loc[val], y_train[val], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores_val[1]*100))
 #   cvscores_train.append(scores_train[1] * 100)
	cvscores_val.append(scores_val[1] * 100)
#print("MAE training score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_train), np.std(cvscores_train)))
print("MAE validation score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_val), np.std(cvscores_val)))


# Plot train and validation MAE
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(len(dict_history["epoch"])-2)]
#
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

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

