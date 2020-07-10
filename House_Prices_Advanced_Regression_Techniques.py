#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("C:/Users/jtani/OneDrive/Documents/train.csv")


# In[3]:


import numpy as np


# In[4]:


data_test=pd.read_csv("C:/Users/jtani/OneDrive/Documents/test.csv")


# In[5]:


data['enc_street'] = pd.get_dummies(data.Street, drop_first=True)
data_test['enc_street'] = pd.get_dummies(data_test.Street, drop_first=True)


# In[6]:


def encode(x):
 return 1 if x == 'Partial' else 0
data['enc_condition'] = data.SaleCondition.apply(encode)
data_test['enc_condition'] = data_test.SaleCondition.apply(encode)


# In[7]:


d = data.select_dtypes(include=[np.number]).interpolate().dropna()


# In[9]:


sum(d.isnull().sum() != 0)


# In[10]:


y = np.log(d.SalePrice)
X = d.drop(['SalePrice', 'Id'], axis=1)


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model=lr.fit(x_train,y_train)


# In[15]:


print ("R^2 is: \n", model.score(x_test, y_test))


# In[16]:


predictions = model.predict(x_test)


# In[17]:


from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[18]:


import matplotlib.pyplot as plt
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# In[19]:


submission = pd.DataFrame()
submission['Id'] = data_test.Id


# In[20]:


feats = data_test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()


# In[21]:


predictions = model.predict(feats)


# In[22]:


final_predictions = np.exp(predictions)


# In[23]:


submission['SalePrice'] = final_predictions
submission.head()


# In[24]:


submission.to_csv('submission1.csv', index=False)


# In[25]:


pwd


# In[ ]:




