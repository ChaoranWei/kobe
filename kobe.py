import matplotlib
import matplotlib.units as units
import pandas as pd
import numpy as np
import math


print 'Read into data...'
print
data = pd.read_csv('data.csv')
data.set_index('shot_id', inplace=True)


def feature_engineering(df):
  
    
    for i in ['game_event_id','game_id', 'team_id', 'team_name']: 
        #drop unnecessary ids, drop team name, because it's always lakers
        df = df.drop(i, 1)
    
    df['game_date'] = pd.to_datetime(df['game_date']) #processing date string

    for i in ['year', 'month','day']:
        df[i] = getattr(df['game_date'].dt,i)

    df = df.drop('season',1)
    df = df.drop('game_date',1)
    
    #turn categorical data into dummy variables
    matchup = df['matchup'].values
    for i,j in enumerate(matchup):
        if '@' in j:
            matchup[i] = 'away'
        else:
            matchup[i] = 'home'
            
    for i in ['action_type','period','shot_type','shot_zone_area','shot_zone_basic','shot_zone_range','opponent', 'matchup','combined_shot_type']:
        dummy = pd.get_dummies(df[i])
        df = pd.concat([df, dummy],axis=1)
        df = df.drop(i,1)
    #value = np.add(np.multiply(df['loc_y'].values,df['loc_y'].values),np.multiply(df['loc_x'].values,df['loc_x'].values))
    #df['distance'] = np.sqrt(value)
    #did not drop combined_shot_type
    
    #change columns with same name
    temp = []
    for i in list(df.columns.values):
        if i not in temp:
            temp.append(i)
        else:
            df = df.drop(i,1)
        
        
    return df

print 'start exploring features...'
data = feature_engineering(data)

'''
print 'some cool visualizations: '
print

import matplotlib.pyplot as plt

#from https://www.kaggle.com/arjoonn/kobe-bryant-shot-selection/preliminary-exploration

#misses and shots
plt.figure(figsize=(2*7, 7*(84.0/50.0)))
plt.subplot(121)
h = data.loc[data.shot_made_flag == 1]
plt.scatter(h.loc_x, h.loc_y, color='green', alpha = 0.05)
plt.title('Shots Made')
ax = plt.gca() #get axes instances
ax.set_ylim([-50, 900])

plt.subplot(122)
h = data.loc[data.shot_made_flag == 0]
plt.scatter(h.loc_x, h.loc_y, color='red', alpha=0.05)
plt.title('Shots missed')
ax = plt.gca()
ax.set_ylim([-50, 900])
#plt.show()
plt.savefig('shots_made_and_missed.png')


#shot types
groups = data.groupby('combined_shot_type')

fig, ax = plt.subplots(figsize=(2*7*0.8, 7*(84.0/60.0)*0.8))
ax.margins(0.05)
alpha = 0.2
alphas, n = [], float(len(data.combined_shot_type))
for u in [i[0] for i in groups]:
    d = len(data.loc[data.combined_shot_type == u, 'combined_shot_type'])
    alphas.append(np.log1p(d)) #transparency, more shots he makes, more opaque the shot is
for (name, group), alp in zip(groups, alphas):
    ax.plot(group.loc_x, group.loc_y,
            marker='.', linestyle='', ms=12,
            label=name, alpha=0.5)
ax.legend()
#plt.show()
plt.savefig('combined_shot_type_layout.png')

def get_acc(df, against):
    ct = pd.crosstab(df.shot_made_flag, df[against]).apply(lambda x:x/x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(against)
    plt.ylabel('% shots made')
    plt.savefig(against + '_vs_accuracy.png')
'''




##########implementation of machine learning algorithms################################
    

train = data[-pd.isnull(data.shot_made_flag)] #training observations are those without shot_made_flag as NaN
test = data[pd.isnull(data.shot_made_flag)]
test = test.drop('shot_made_flag', 1)

y = train['shot_made_flag'] #labels
X = train.drop('shot_made_flag', 1) #features
#print train.head(10)
#print train.columns.values

#kfold cross validation

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.lda import LDA

'''
kfold = KFold(n=len(X), n_folds = 10, random_state = 2288)

models = [('LR',LogisticRegression()),('lda', LDA()), ('GBC', GradientBoostingClassifier(n_estimators=100, random_state=2288, max_depth = 4, learning_rate = 0.1, max_features = 10)), ('RF', RandomForestClassifier(n_estimators = 1000, max_features=20,criterion = 'entropy', max_depth = 8, bootstrap = True, warm_start=True, random_state=245))]
#Naive Bayes: -12, LR: -0.61, RF: -0.60+, GBC: 0.60+
#, ('SVM',SVC(probability=True))
#, ('RF', RandomForestClassifier())

for name, model in models:
    print 'training ' + name + '...'
    results = cross_val_score(model, X, y, cv=kfold, scoring = 'log_loss')
    print name + ': ' + str(results.mean()) + ' +/- ' + str(results.std())
 
#above cross validation shows that ' ' is the best estimator

model = GradientBoostingClassifier(n_estimators=100, random_state=2288, max_depth = 4, learning_rate = 0.1, max_features = 10)

model.fit(X, y)
preds = model.predict_proba(test)

submission = pd.DataFrame()
submission["shot_id"] = test.index 
submission["shot_made_flag"]= preds[:,1]

submission.to_csv("sub.csv",index=False)
'''
#xgboost
from xgboost.sklearn import XGBClassifier

clf_xgb = XGBClassifier(max_depth=7, learning_rate=0.012, n_estimators=1000, subsample=0.62, colsample_bytree=0.6, seed=1)
clf_xgb.fit(X, y)

preds = clf_xgb.predict_proba(test)
submission = pd.DataFrame()
submission["shot_id"] = test.index 
submission["shot_made_flag"]= preds[:,1]

submission.to_csv("sub.csv",index=False)