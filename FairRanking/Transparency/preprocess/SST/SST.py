#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
a = nltk.corpus.BracketParseCorpusReader(".", "(train|dev|test)\.txt")

text = {}
labels = {}
keys = ['train', 'dev', 'test']
for k in keys :
    text[k] = [x.leaves() for x in a.parsed_sents(k+'.txt') if x.label() != '2']
    labels[k] = [int(x.label()) for x in a.parsed_sents(k+'.txt') if x.label() != '2']
    print(len(text[k]))
    
import spacy
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
import re

def tokenize(text) :
    text = " ".join(text)
    text = text.replace("-LRB-", '')
    text = text.replace("-RRB-", " ")
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    tokens = " ".join([t.text.lower() for t in nlp(text)])
    return tokens

for k in keys :
    text[k] = [tokenize(t) for t in text[k]]
    labels[k] = [1 if x >= 3 else 0 for x in labels[k]]


# In[2]:


import pandas as pd
df_texts = []
df_labels = []
df_exp_split = []

for k in keys :
    df_texts += text[k]
    df_labels += labels[k]
    df_exp_split += [k]*len(text[k])
    
df = pd.DataFrame({'text' : df_texts, 'label' : df_labels, 'exp_split' : df_exp_split}) 


# In[3]:


df.to_csv('sst_dataset.csv', index=False)


# In[4]:


get_ipython().run_line_magic('run', '"../preprocess_data_BC.py" --data_file sst_dataset.csv --output_file ./vec_sst.p --word_vectors_type fasttext.simple.300d --min_df 1')


# In[ ]:




