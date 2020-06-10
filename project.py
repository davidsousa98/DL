from preprocessing import correlation_matrix, X_train, X_test, y_train, y_test, df_train,\
    df_test, df_league_games
import pandas as pd
import numpy as np
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

# Classification table for 'Premier LEague'
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

