#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[2]:


data = pd.read_csv('spam.csv', encoding='ISO-8859-1')


# In[3]:


print(data.head())


# In[4]:


print(data.info())


# In[5]:


data = data.drop(columns=data.columns[2:5])


# In[6]:


data.columns = ['Category', 'Message']


# In[7]:


print(data.head())


# In[8]:


print(data.isnull().sum())


# In[14]:


category_counts = data['Category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

plt.figure(figsize=(8, 6))
sns.barplot(x='Category', y='Count', data=category_counts)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Category Distribution')

for i, count in enumerate(category_counts['Count']):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()


# In[15]:


data['spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)


# In[16]:


print(data.head())


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(data.Message, data.spam, test_size=0.2)


# In[18]:


featurer = CountVectorizer()


# In[19]:


X_train_count = featurer.fit_transform(X_train.values)


# In[20]:


print(X_train_count)


# In[21]:


model = MultinomialNB()


# In[22]:


model.fit(X_train_count, y_train)


# In[23]:


X_test_count = featurer.transform(X_test)


# In[24]:


model_accuracy = model.score(X_test_count, y_test)
print("Model Accuracy:", model_accuracy)


# In[25]:


clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])


# In[26]:


clf.fit(X_train, y_train)


# In[27]:


pipeline_accuracy = clf.score(X_test, y_test)
print("Pipeline Accuracy:", pipeline_accuracy)


# In[28]:


pretrained_model = model
new_sentences = [
    "Your account has been debited with $100. Please verify by texting the password 'MIX' to 85069. Get Usher and Britney. FML"
]


# In[29]:


new_sentences_count = featurer.transform(new_sentences)


# In[30]:


predictions = pretrained_model.predict(new_sentences_count)


# In[31]:


for sentence, prediction in zip(new_sentences, predictions):
    if prediction == 1:
        print(f"'{sentence}' is a spam message.")
    else:
        print(f"'{sentence}' is not a spam message.")


# In[ ]:




