import matplotlib
import matplotlib.units as units
import pandas as pd
import numpy as np


print 'Read into data...'
print
data = pd.read_csv('data.csv')
print data.shot_made_flag.head(100)


def feature_engineering(df):
  
    
    for i in ['game_event_id','game_id', 'team_id','shot_id', 'team_name']: 
        #drop unnecessary ids, drop team name, because it's always lakers
        df.drop(i, 1)
    
    df = pd.to_datetime()
    
    
    

train = data[-pd.isnull(data.shot_made_flag)] #training observations are those withpit shot_made_flag as NaN
test = data[pd.isnull(data.shot_made_flag)]
test.drop('shot_made_flag', 1)

X = train[-train['shot_made_flag']] #features
y = train['shot_made_flag'] #labels
X.drop('shot_made_flag', 1)


print 'start exploring features...'
