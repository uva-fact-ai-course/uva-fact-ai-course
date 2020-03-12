#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip')
get_ipython().system('unzip snli_1.0.zip')


# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# In[9]:


keys = ['train', 'test', 'dev']


# In[10]:


import json


# In[11]:


data = {k:[] for k in keys}
for k in keys :
    for line in open('snli_1.0/snli_1.0_' + k + '.jsonl').readlines() :
        data[k].append(json.loads(line))


# In[12]:


from Transparency.preprocess.vectorizer import cleaner
from tqdm import tqdm_notebook


# In[13]:


p, q, a = {}, {}, {}

for k in keys :
    p[k] = [cleaner(x['sentence1']) for x in tqdm_notebook(data[k]) if x['gold_label'] != '-']
    q[k] = [cleaner(x['sentence2']) for x in tqdm_notebook(data[k]) if x['gold_label'] != '-']
    a[k] = [cleaner(x['gold_label']) for x in tqdm_notebook(data[k]) if x['gold_label'] != '-']


# In[14]:


entity_list = ['neutral', 'contradiction', 'entailment']
f = open('entity_list.txt', 'w')
f.write("\n".join(entity_list))
f.close()


# In[15]:


import pandas as pd
df_paragraphs = []
df_questions = []
df_answers = []
df_exp_splits = []

for k in keys :
    df_paragraphs += p[k]
    df_questions += q[k]
    df_answers += a[k]
    df_exp_splits += [k] * len(p[k])
    
df = {'paragraph' : df_paragraphs, 'question' : df_questions, 'answer' : df_answers, 'exp_split' : df_exp_splits}
df = pd.DataFrame(df)


# In[16]:


df.to_csv('snli_dataset.csv', index=False)


# In[1]:


get_ipython().run_line_magic('run', '"../preprocess_data_QA.py" --data_file snli_dataset.csv --output_file ./vec_snli.p --all_answers_file entity_list.txt --word_vectors_type glove.840B.300d --min_df 3')


# In[ ]:




