# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import datetime as dt
from datetime import datetime
pd.set_option('display.max_rows', 50)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# %% [markdown]
# Tables Schema and Relationships
# 

# %%

from IPython.display import Image
Image(filename='..\drawing\\relationship.png')


# %%
# Importing all the CSVs as dataframes
df_circuits = pd.read_csv('..\data\circuits.csv', index_col=False)
df_constructor_results = pd.read_csv(
    '..\data\constructor_results.csv', index_col=False)
df_constructor_standings = pd.read_csv(
    '..\data\constructor_standings.csv', index_col=False)
df_constructors = pd.read_csv('..\data\constructors.csv', index_col=False)
df_driver_standings = pd.read_csv(
    '..\data\driver_standings.csv', index_col=False)
df_drivers = pd.read_csv('..\data\drivers.csv', index_col=False)
df_lap_times = pd.read_csv('..\data\lap_times.csv', index_col=False)
df_pit_stops = pd.read_csv('..\data\pit_stops.csv', index_col=False)
df_qualifying = pd.read_csv('..\data\qualifying.csv', index_col=False)
df_races = pd.read_csv('..\data\\races.csv', index_col=False)
df_results = pd.read_csv('..\data\\results.csv', index_col=False)
df_seasons = pd.read_csv('..\data\seasons.csv', index_col=False)
df_sprint_results = pd.read_csv('..\data\sprint_results.csv', index_col=False)
df_status = pd.read_csv('..\data\status.csv', index_col=False)


# %%
# Merging results with Racers, Drivers, Constructors and Status
df = (df_results.merge(df_races[['raceId', 'name', 'date']], on='raceId', how='left')
      .merge(df_drivers[['driverId', 'driverRef', 'nationality', 'dob']], on='driverId', how='left')
      .merge(df_constructors[['constructorId', 'name', 'nationality']], on='constructorId', how='left')
      .merge(df_status[['statusId', 'status']], on='statusId', how='left')
      )
df_eda = df.rename(columns={"name_x": "RacerName", "name_y": "ConstructorName", "nationality_x": "RacerNacionality", "nationality_y": "ConstructorNacionality"})


# %%
#renaming Columns
df = df[['resultId', 'name_x', 'date', 'driverRef', 'nationality_x', 'dob', 'name_y', 'nationality_y', 'status', 'number', 'positionOrder', 'points', 'laps', 'milliseconds', 'fastestLap',
         'rank', 'fastestLapTime', 'fastestLapSpeed', 'status']].rename(columns={"name_x": "RacerName", "name_y": "ConstructorName", "nationality_x": "RacerNacionality", "nationality_y": "ConstructorNacionality"})
df = df.replace('\\N', 0)  # Replacing nulls per 0


# %%
# Converting objects in Categories and Datetime
cat_cols = ['RacerName', 'driverRef', 'RacerNacionality',
            'ConstructorName', 'ConstructorNacionality', 'positionOrder', 'rank', 'status']
df[cat_cols] = df[cat_cols].astype('category')
df['fastestLapTime'] = pd.to_datetime(df['fastestLapTime'], infer_datetime_format=True)
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df['dob'] = pd.to_datetime(df['dob'], infer_datetime_format=True)

# Sort data by race date
df.sort_values(by='date', inplace=True)

# Create a reference to the original dataframe for plotting



# %% [markdown]
# Since Lewis Hamilton is the current biggest winner in F1 I decided to get some of the highlight of him and some data regards Ciscuits / Performances

# %%
px.bar(df_eda.query('position == "1"')
       .groupby(["driverRef"])["RacerName"].agg({'count'})
       .sort_values('count', ascending=False).head(20), title='Top 20 Race Winners'
       )


# %% [markdown]
# Racers and its winning circuits
# 

# %%
(df_eda.query('position == "1"')
 .groupby(["driverRef", "RacerName"])["driverRef"].agg({'count'})
 .query('count >= 1')
 .sort_values('driverRef', ascending=True)
 )


# %% [markdown]
# Number of unique circuits the racers won
# 

# %%
px.bar(df_eda.query('position == "1"')
       .groupby(["driverRef"])["RacerName"].agg({'nunique'})
       .query('nunique >= 1')
       .sort_values('nunique', ascending=False), title='Ranking of Racers that won different circuits ( uniques )'
       )


# %% [markdown]
# Circuits and it`s respective winners along the time
# 

# %%
(df_eda.query('position == "1"')
 .groupby(["RacerName", "driverRef"])["RacerName"].agg({'count'})
 .query('count >= 1')
 .sort_values('RacerName', ascending=False)
 )


# %% [markdown]
# Ranking circuits that had repetitive winners
# 

# %%
(df_eda.query('position == "1"')
 .groupby(["RacerName", "driverRef"])["RacerName"].agg({'count'})
 .query('count >= 2')
 .sort_values('count', ascending=False)
 )


# %% [markdown]
# How many unique circuits have Lewis Hamilton already won?
# 
# How many unique circuits have Lewis Hamilton did not win yet?
# 

# %%
circuits = df_eda['RacerName'].unique()
hamilton_circuits = df_eda.query('position == "1" and driverRef == "hamilton"')['RacerName'].unique()

# Count the number of circuits won and lost by Hamilton
num_wins = len([circuit for circuit in circuits if circuit in hamilton_circuits])
num_losses = len([circuit for circuit in circuits if circuit not in hamilton_circuits])

# Print the results using f-strings
print(f"Lewis Hamilton has won in {num_wins} circuits")
print(f"Lewis Hamilton has not won in {num_losses} circuits")

# %% [markdown]
# Validation dataset ( races of 2022 )

# %%
validation = df[df["date"].dt.year == 2022]
data = df[df["date"].dt.year != 2022]

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# %%

le = preprocessing.LabelEncoder()

# %% [markdown]
# Defining data types for validation

# %%
def preprocess_data(X):
    """
    Preprocesses the given dataframe X by applying label encoding and
    converting datetime columns to numeric format.

    Args:
        X: a pandas dataframe containing the input features

    Returns:
        The preprocessed pandas dataframe
    """
    X = X.copy()
    label_cols = ['RacerName', 'driverRef', 'RacerNacionality',
                  'ConstructorName', 'ConstructorNacionality', 'status']
    X[label_cols] = X[label_cols].apply(preprocessing.LabelEncoder().fit_transform)
    X['fastestLapTime'] = pd.to_numeric(pd.to_datetime(X['fastestLapTime']))
    X['date'] = pd.to_numeric(pd.to_datetime(X['date']))
    X['dob'] = pd.to_numeric(pd.to_datetime(X['dob']))
    return X

X_val = validation[['resultId', 'RacerName', 'date', 'driverRef', 'RacerNacionality', 'dob', 'ConstructorName', 'ConstructorNacionality',
                    'number', 'laps', 'milliseconds', 'fastestLap', 'fastestLapTime', 'fastestLapSpeed', 'status']]
y_val = validation[['points']]

X_val = preprocess_data(X_val)

X = data[['resultId', 'RacerName', 'date', 'driverRef', 'RacerNacionality', 'dob', 'ConstructorName', 'ConstructorNacionality',
        'number','laps', 'milliseconds', 'fastestLap','fastestLapTime', 'fastestLapSpeed', 'status']]
y = data[['points']]

X = preprocess_data(X)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
rfc = RandomForestRegressor(max_depth=2, random_state=0)
rfc.fit(X_train, Y_train.values.ravel())


# %%

predictions = rfc.predict(X_test)
predictions = [round(x) for x in predictions]
Y_test['predictions'] = predictions
print('RMSE of the Random Forest Model for test data ( Considering ALL Years) : {0}'.format(mean_squared_error(Y_test['points'], predictions)))


# %%
predictions_val = rfc.predict(X_val)
validation = validation.assign(predicted_RF=np.round(predictions_val))
rgb = GradientBoostingRegressor(random_state=0)
rgb.fit(X_train, Y_train.values.ravel())
rgb_pred = rgb.predict(X_val)
rgb_pred = [round(x) for x in rgb_pred]

# %%
validation = validation.assign(predicted_rgb=np.round(rgb_pred))
print('RMSE of the Gradient Boost Model on the Validation data: {0}'.format(mean_squared_error(validation['points'], rgb.predict(X_val))))


# %% [markdown]
# Validation Data Predictions Visualization

# %%
gb_data = validation.groupby(['driverRef'])['points','predicted_RF','predicted_rgb'].agg(
    {'sum'}).reset_index().sort_values(('points', 'sum'), ascending=False)[:20]
s = gb_data.style.bar(color='#d65f5f')
s


# %%
print('RMSE of the Random Forrest Model on the Validation data : {0}'.format(mean_squared_error(validation['points'], rfc.predict(X_val))))

print('RMSE of the Gradient Boost Model on the Validation data: {0}'.format(mean_squared_error(validation['points'], rgb.predict(X_val))))

# %%
rfc.max_depth

# %%
from sklearn.model_selection import GridSearchCV 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5,10,15,20,50,80, 90, 100, 110],
    'max_features': [2, 3, 4, 5],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12, 15],
    'n_estimators': [50, 100, 200, 300, 1000, 10000]
}# Create a based model
rf = RandomForestRegressor()# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# %%
#grid_search.fit(X_train, Y_train.values.ravel())

# %%
print('''{'bootstrap': True,
 'max_depth': 100,
 'max_features': 5,
 'min_samples_leaf': 3,
 'min_samples_split': 10,
 'n_estimators': 300}''')

# %%
print('RMSE of the Gradient Boost Model on the Validation data: {0}'.format(mean_squared_error(validation['points'], grid_search.predict(X_val))))

# %%
validation = validation.assign(predicted_GRID=np.round(grid_search.predict(X_val)))
gb_data = validation.groupby(['driverRef'])['points','predicted_RF','predicted_rgb','predicted_GRID'].agg(
    {'sum'}).reset_index().sort_values(('points', 'sum'), ascending=False)[:20]
s = gb_data.style.bar(color='#d65f5f')
s

# %% [markdown]
# The Performance after gridsearch did not improve the results against the validation data.
# 
# It`s possible to already admit that we could not create a model capable of predict the results of F1 starting in 2022 with a good RMSE.
# The key thing is that this year was the game changed, and Redbull assumed the leadership domiantion and Mercedes is currently strugling with its car performance.


