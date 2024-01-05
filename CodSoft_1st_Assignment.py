#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


train_data_path = "train_data.txt"
train_columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
train_data = pd.read_csv(train_data_path, sep=':::', names=train_columns)
display(train_data.head())
print(train_data.shape)


# In[3]:


test_data_path = "test_data.txt"
test_columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
test_data = pd.read_csv(test_data_path, sep=':::', names=test_columns)
print(display(test_data.head()))
print(test_data.shape)


# In[4]:


test_solution_data_path = "test_data_solution.txt"
test_solution_data = pd.read_csv(test_solution_data_path, sep=':::', names=test_columns)
print(display(test_solution_data.head()))
print(test_solution_data.shape)


# In[7]:


genre_counts = train_data['GENRE'].value_counts()


# In[8]:


plt.figure(figsize=(20, 8))


# In[9]:


genre_counts.plot(kind='barh')


# In[10]:


plt.title('Number of Movies per Genre')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')


# In[25]:


train_data['description_length'] = train_data['DESCRIPTION'].apply(len)

plt.figure(figsize=(15, 10))

sns.pointplot(x='GENRE', y='description_length', data=train_data, capsize=.2, dodge=True, join=False)
plt.title('Description Length by Genre')
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Description Length')

plt.show()


# In[29]:


top_genres = train_data['GENRE'].value_counts().head(10)


# In[30]:


plt.figure(figsize=(10, 10))


# In[31]:


plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10.colors)
plt.title('Top 10 Most Frequent Genres')


# In[32]:


plt.show()


# In[33]:


(train_data['DESCRIPTION'].fillna("", inplace=True), test_data['DESCRIPTION'].fillna("", inplace=True))


# In[34]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100000)


# In[35]:


X_train = tfidf_vectorizer.fit_transform(train_data['DESCRIPTION'])
X_test = tfidf_vectorizer.transform(test_data['DESCRIPTION'])


# In[36]:


label_encoder = LabelEncoder()


# In[37]:


y_train = label_encoder.fit_transform(train_data['GENRE'])


# In[38]:


y_test = label_encoder.transform(test_solution_data['GENRE'])


# In[39]:


X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[40]:


clf = LinearSVC()


# In[41]:


clf.fit(X_train_sub, y_train_sub)


# In[42]:


y_val_pred = clf.predict(X_val)


# In[43]:


print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))


# In[44]:


y_test_pred = clf.predict(X_test)


# In[45]:


print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))


# In[46]:


from sklearn.naive_bayes import MultinomialNB

mnb_classifier = MultinomialNB()

mnb_classifier.fit(X_train, y_train)


# In[48]:


mnb_classifier.predict(X_test)


# In[50]:


from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=700)

# Train the classifier on the training data
lr_classifier.fit(X_train, y_train)


# In[51]:


y_test_pred_lr = lr_classifier.predict(X_test)


# In[ ]:


def predict_movie(description, vectorizer, classifier, label_encoder):

    description_vectorized = vectorizer.transform([description])

    pred_label = classifier.predict(description_vectorized)

    predicted_genre = label_encoder.inverse_transform(pred_label)[0]

    return predicted_genre


# In[1]:


sample_descr_for_movie = "A movie where police chases the criminal and shoots him"
predicted_genre_1 = predict_movie(sample_descr_for_movie, t_v, clf, label_encoder)
print(predicted_genre_1)

sample_descr_for_movie1 = "A movie where a person chases a girl to get married with him but the girl refuses him."
predicted_genre_2 = predict_movie(sample_descr_for_movie1, t_v, clf, label_encoder)
print(predicted_genre_2)


# In[ ]:




