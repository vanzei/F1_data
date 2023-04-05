#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
pd.set_option('display.max_rows', 50)


# In[4]:


from IPython.display import Image
Image(filename='..\drawing\\relationship.png') 


# In[5]:


df_circuits = pd.read_csv('..\data\circuits.csv', index_col=False)
df_constructor_results = pd.read_csv('..\data\constructor_results.csv', index_col=False)
df_constructor_standings = pd.read_csv('..\data\constructor_standings.csv', index_col=False)
df_constructors = pd.read_csv('..\data\constructors.csv', index_col=False)
df_driver_standings = pd.read_csv('..\data\driver_standings.csv', index_col=False)
df_drivers = pd.read_csv('..\data\drivers.csv', index_col=False)
df_lap_times = pd.read_csv('..\data\lap_times.csv', index_col=False)
df_pit_stops = pd.read_csv('..\data\pit_stops.csv', index_col=False)
df_qualifying = pd.read_csv('..\data\qualifying.csv', index_col=False)
df_races = pd.read_csv('..\data\\races.csv', index_col=False)
df_results = pd.read_csv('..\data\\results.csv', index_col=False)
df_seasons = pd.read_csv('..\data\seasons.csv', index_col=False)
df_sprint_results = pd.read_csv('..\data\sprint_results.csv', index_col=False)
df_status = pd.read_csv('..\data\status.csv', index_col=False)


# In[6]:


df = (df_results.merge(df_races[['raceId','name','date']],on='raceId', how='left')
    .merge(df_drivers[['driverId','driverRef','nationality','dob']],on='driverId', how='left')
    .merge(df_constructors[['constructorId','name','nationality']],on='constructorId', how='left')
    .merge(df_status[['statusId','status']],on='statusId', how='left')
)


# In[7]:


df = df[['resultId', 'name_x', 'date', 'driverRef','nationality_x', 'dob', 'name_y', 'nationality_y', 'status', 'number', 'grid','position', 'positionOrder', 'points', 'laps','milliseconds', 'fastestLap', 'rank', 'fastestLapTime','fastestLapSpeed','status']].rename(columns={"name_x":"RacerName", "name_y":"ConstructorName","nationality_x":"RacerNacionality","nationality_y":"ConstructorNacionality"})


# In[8]:


df.head()


# In[9]:


df = df.replace('\\N', 0) # Replacing nulls per 0


# In[10]:


#for col1 in ['resultId', 'RacerName', 'driverRef', 'RacerNacionality',
#       'ConstructorName', 'ConstructorNacionality', 'status', 'number', 'grid',
#       'position', 'positionOrder', 'points', 'rank', 'status']:
#    df[col1] = df[col1].astype('category')
#df = df.astype({'milliseconds':'int64'})
#df['fastestLapTime'] = df['fastestLapTime'].apply(pd.to_datetime)
#df['date'] = df['date'].apply(pd.to_datetime)
#df['dob'] = df['dob'].apply(pd.to_datetime)


# In[11]:


df.info()


# In[12]:


px.bar(df.query('position == "1"')
.groupby(["driverRef"])["RacerName"].agg({'count'})
.sort_values('count',ascending=False).head(20), title='Top 20 Race Winners'
)


# Racers and its winning circuits

# In[13]:


(df.query('position == "1"')
.groupby(["driverRef","RacerName"])["driverRef"].agg({'count'})
.sort_values('driverRef',ascending=True)
)


# Number of unique circuits the racers won

# In[14]:


(df.query('position == "1"')
.groupby(["driverRef"])["RacerName"].agg({'nunique'})
.sort_values('nunique',ascending=False)
)


# Circuits and it`s respective winners along the time

# In[15]:


(df.query('position == "1"')
.groupby(["RacerName","driverRef"])["RacerName"].agg({'count'})
.sort_values('RacerName',ascending=False)
)


# Ranking circuits that had repetitive winners

# In[16]:


(df.query('position == "1"')
.groupby(["RacerName","driverRef"])["RacerName"].agg({'count'})
.query('count >= 2')
.sort_values('count',ascending=False)
)


# In[17]:


circuits = df['RacerName'].unique()
hamilton_circuits = df.query('position == "1" and driverRef == "hamilton"')['RacerName'].unique()
len(circuits)


# In[21]:


i=0
j=0
for cir in circuits:
    if cir not in hamilton_circuits:
        j+=1
        print(cir)
    else:
        i+=1
print(f'Lewis Hamilton already won in {i} Circuits')
print(f'Lewis Hamilton have not win in {j} Circuits')

