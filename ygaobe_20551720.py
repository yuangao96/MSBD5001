
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn as sl
import numpy as np


# In[2]:


train = pd.read_csv('D:\\personal\\train.csv')
test = pd.read_csv('D:\\personal\\test.csv')


# In[3]:


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
pen = preprocessing.LabelEncoder()
train['penalty'] = pen.fit_transform(list(train['penalty'].values))
test['penalty'] = pen.fit_transform(list(test['penalty'].values))

#train['penalty'] = OneHotEncoder(sparse = False).fit_transform(train[['penalty']])
#test['penalty'] = OneHotEncoder(sparse = False).fit_transform(test[['penalty']])

#train['max_iter'] = (train['max_iter'] - train['max_iter'].mean())/train['max_iter'].std()
#train['random_state'] = (train['random_state'] - train['random_state'].mean())/train['random_state'].std()
#train['n_samples'] = (train['n_samples'] - train['n_samples'].mean())/train['n_samples'].std()
#train['n_features'] = (train['n_features'] - train['n_features'].mean())/train['n_features'].std()

#test['max_iter'] = (test['max_iter'] - test['max_iter'].mean())/test['max_iter'].std()
#test['random_state'] = (test['random_state'] - test['random_state'].mean())/test['random_state'].std()
#test['n_samples'] = (test['n_samples'] - test['n_samples'].mean())/test['n_samples'].std()
#test['n_features'] = (test['n_features'] - test['n_features'].mean())/test['n_features'].std()


# In[4]:


from sklearn.model_selection import train_test_split
features = list(train.columns[1:2]) + list(train.columns[3:5]) +list(train.columns[6:14])
#features = list(train.columns[1:14])
#print(features)
y = train['time']
x_new = train[features]
x_new_train,x_new_test,y_train,y_test = train_test_split(x_new,y,test_size=0.3,random_state=0)


# In[6]:


import xgboost as xgb

xlf = xgb.XGBRegressor( 
                        max_depth=5, 
                        learning_rate=0.9, 
                        n_estimators=500,
                        silent=True, 
                        objective='reg:tweedie',
                        nthread=-1, 
                        gamma=0.03,
                        min_child_weight=5, 
                        max_delta_step=0,
                        subsample=0.6, 
                        colsample_bytree=0.9, 
                        colsample_bylevel=1, 
                        reg_alpha=2, 
                        reg_lambda=3, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None
                      )
xlf.fit(x_new_train, y_train)
test_y_predicted = xlf.predict(x_new_test)
#print(test_y_predicted)

#test_y_predicted_right = []   
#for items in test_y_predicted:
#    if items > 0:
#        test_y_predicted_right.append(items)
#test_y_predicted[test_y_predicted < 0] = min(test_y_predicted_right)

from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_y_predicted))


# In[7]:


xlf_y_1 = xlf.predict(test[features])
print(xlf_y_1)
import csv
output_1 = open('D:\\personal\\output_1.csv','w',newline='')
writer = csv.writer(output_1)
writer.writerows(map(lambda x: [x],xlf_y_1))
output_1.close()

df_example = pd.read_csv('D:\\personal\\output_1.csv',header = None,names=['time'])
df_example.index.name = 'Id'
df_example.to_csv('D:\\personal\\11_21.csv',encoding='utf-8')

