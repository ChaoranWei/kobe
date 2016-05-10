import matplotlib
import matplotlib.units as units
import pandas as pd
import numpy as np


print 'Read into data...'
print
data = pd.read_csv('data.csv')
#print data.head(100)


def feature_engineering(df):
  
    
    for i in ['game_event_id','game_id', 'team_id','shot_id', 'team_name']: 
        #drop unnecessary ids, drop team name, because it's always lakers
        df = df.drop(i, 1)
    
    df['game_date'] = pd.to_datetime(df['game_date']) #processing date string
    for i in ['year', 'month','day']:
        df[i] = getattr(df['game_date'].dt, i)
        
    df = df.drop('season',1)
    df = df.drop('game_date',1)
    
    #turn categorical data into dummy variables
    matchup = df['matchup'].values
    for i,j in enumerate(matchup):
        if '@' in j:
            matchup[i] = 'away'
        else:
            matchup[i] = 'home'
            
    for i in ['action_type','combined_shot_type','period','shot_type','shot_zone_area','shot_zone_basic','shot_zone_range','opponent', 'matchup']:
        dummy = pd.get_dummies(df[i])
        df = pd.concat([df, dummy],axis=1)
        df = df.drop(i,1)
    
    return df

print 'start exploring features...'
data = feature_engineering(data)
    

train = data[-pd.isnull(data.shot_made_flag)] #training observations are those without shot_made_flag as NaN
test = data[pd.isnull(data.shot_made_flag)]
test.drop('shot_made_flag', 1)

y = train['shot_made_flag'] #labels
X = train.drop('shot_made_flag', 1) #features
print train.head(10)
print train.columns.values
