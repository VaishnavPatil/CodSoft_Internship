#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_csv("Churn_Modelling.csv")


# In[5]:


print(data.head())


# In[6]:


print(data.info())


# In[7]:


print(data.isnull().sum())


# In[8]:


data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# In[9]:


data = pd.get_dummies(data, drop_first=True)


# In[10]:


data = data.astype(int)


# In[11]:


print(data.head())


# In[12]:


print(data['Exited'].value_counts())


# In[13]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Exited', data=data, palette='pastel')
plt.title('Customer Churn Distribution', fontsize=16)
plt.xlabel('Exited', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# In[14]:


X = data.drop('Exited', axis=1)
y = data['Exited']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('Training Shape: ', X_train.shape)
print('Testing Shape: ', X_test.shape)


# In[17]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[18]:


threshold = 0.5


# In[19]:


y_train_classified = [1 if value > threshold else 0 for value in y_train]


# In[20]:


LR = LogisticRegression()
LR.fit(X_train_scaled, y_train_classified)


# In[21]:


y_test_classified = [1 if value > threshold else 0 for value in y_test]
accuracy1 = accuracy_score(y_test_classified, LR.predict(X_test_scaled))
print("Logistic Regression Model Accuracy:", accuracy1)


# In[22]:


svm = SVC()
svm.fit(X_train_scaled, y_train_classified)


# In[23]:


accuracy2 = accuracy_score(y_test_classified, svm.predict(X_test_scaled))
print("Support Vector Machine Model Accuracy:", accuracy2)


# In[24]:


rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train_classified)


# In[25]:


accuracy3 = accuracy_score(y_test_classified, rf.predict(X_test_scaled))
print("Random Forest Model Accuracy:", accuracy3)


# In[26]:


dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train_classified)


# In[27]:


accuracy4 = accuracy_score(y_test_classified, dt.predict(X_test_scaled))
print("Decision Tree Model Accuracy:", accuracy4)


# In[28]:


KNN = KNeighborsClassifier()
KNN.fit(X_train_scaled, y_train_classified)


# In[29]:


accuracy5 = accuracy_score(y_test_classified, KNN.predict(X_test_scaled))
print("K-Nearest Neighbors Model Accuracy:", accuracy5)


# In[30]:


GBC = GradientBoostingClassifier()
GBC.fit(X_train_scaled, y_train_classified)


# In[31]:


accuracy6 = accuracy_score(y_test_classified, GBC.predict(X_test_scaled))
print("Gradient Boosting Model Accuracy:", accuracy6)


# In[32]:


performance_summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors', 'Gradient Boosting'],
    'Accuracy': [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6]
})


# In[33]:


print(performance_summary)


# In[ ]:




