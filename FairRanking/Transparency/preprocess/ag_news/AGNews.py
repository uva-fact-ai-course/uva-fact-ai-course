#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm_notebook


# In[2]:


df = {}
keys = ['train', 'test']
for k in keys :
    df[k] = pd.read_csv('' + k + '.csv', header=None)
    df[k] = df[k][df[k][0].isin([1, 3])]


# In[3]:


from Transparency.preprocess.vectorizer import cleaner


# In[4]:


for k in keys :
    texts = list(df[k][2])
    for i in tqdm_notebook(range(len(texts))) :
        texts[i] = cleaner(texts[i])
    df[k]['text'] = texts


# In[5]:


for k in keys :
    df[k][0] = [1 if (x == 3) else 0 for x in list(df[k][0])]


# In[6]:


import pandas as pd
df_texts = []
df_labels = []
df_exp_splits = []

for key in ['train', 'test'] :
    df_texts += list(df[key]['text'])
    df_labels += list(df[key][0])
    df_exp_splits += [key] * len(list(df[key]['text']))
    
df = pd.DataFrame({'text' : df_texts, 'label' : df_labels, 'exp_split' : df_exp_splits})
df.to_csv('agnews_dataset.csv', index=False)


# In[7]:


import pandas as pd
df = pd.read_csv('agnews_dataset.csv')

from sklearn.model_selection import train_test_split
train_idx, dev_idx = train_test_split(df.index[df.exp_split == 'train'], test_size=0.15, random_state=16377)
df.loc[dev_idx, 'exp_split'] = 'dev'
df.to_csv('agnews_dataset_split.csv', index=False)


# In[8]:


get_ipython().run_line_magic('run', '"../preprocess_data_BC.py" --data_file agnews_dataset_split.csv --output_file ./vec_agnews.p --word_vectors_type fasttext.simple.300d --min_df 5')


# In[ ]:





# In[ ]:




