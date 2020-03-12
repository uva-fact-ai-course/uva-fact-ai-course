#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups
data20 = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))
print(list(enumerate(data20.target_names)))


# In[2]:


import numpy as np
baseball = np.where(data20.target == 9)[0]
hockey = np.where(data20.target == 10)[0]


# In[3]:


strings = [data20.data[i] for i in list(baseball) + list(hockey)]
target = [0 if data20.target[i] == 9 else 1 for i in list(baseball) + list(hockey)]


# In[4]:


from Transparency.preprocess.vectorizer import cleaner
import re

def cleaner_20(text) :
    text = cleaner(text)
    text = re.sub(r'(\W)+', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

strings = [cleaner_20(s) for s in strings]
strings, target = zip(*[(s, t) for s, t in zip(strings, target) if len(s) != 0])


# In[5]:


from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(range(len(strings)), stratify=target, test_size=0.2, random_state=13478)
train_idx, dev_idx = train_test_split(train_idx, stratify=[target[i] for i in train_idx], test_size=0.2, random_state=13478)


# In[6]:


X_train = [strings[i] for i in train_idx]
X_dev = [strings[i] for i in dev_idx]
X_test = [strings[i] for i in test_idx]

y_train = [target[i] for i in train_idx]
y_dev = [target[i] for i in dev_idx]
y_test = [target[i] for i in test_idx]

texts = { 'train' : X_train, 'test' : X_test, 'dev' : X_dev }
labels = { 'train' : y_train, 'test' : y_test, 'dev' : y_dev }


# In[7]:


import pandas as pd
df_texts = []
df_labels = []
df_exp_splits = []

for key in ['train', 'test', 'dev'] :
    df_texts += texts[key]
    df_labels += labels[key]
    df_exp_splits += [key] * len(texts[key])
    
df = pd.DataFrame({'text' : df_texts, 'label' : df_labels, 'exp_split' : df_exp_splits})
df.to_csv('20News_sports_dataset.csv', index=False)


# In[8]:


get_ipython().run_line_magic('run', '"../preprocess_data_BC.py" --data_file 20News_sports_dataset.csv --output_file ./vec_20news_sports.p --word_vectors_type fasttext.simple.300d --min_df 2')


# In[ ]:




