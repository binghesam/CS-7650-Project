#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import random
import re
import matplotlib.pyplot as plt
import pickle


# In[5]:


pairs_full = []
with open('biased.full') as f:
    for line in f:
        items = re.sub(' ##','',line).split('\t')
        if len(items[1].split())< 100:
            pairs_full.append((items[0],items[1],items[2]))
print('len(pairs_full)',len(pairs_full))


# In[36]:


idx_filtered = None


# In[37]:


idx_filtered = pickle.load(open( "./idx_filtered", "rb" ) )


# In[38]:


if idx_filtered:
    pairs_short = []
    for i in idx_filtered:
        pairs_short.append(pairs_full[i])
    n_choice = len(pairs_short)
else:
    n_choice = int(len(pairs_full)/1) # choose a small size to test result
    pairs_short = random.sample(pairs_full,k=n_choice) # bias neutral pairs


# Randomly mix biased and neutral statement together with 0 as neutral and 1 as biased

# In[39]:


bias_sent = [pair[1] for pair in pairs_short]
neutral_sent = [pair[2] for pair in pairs_short]


# In[1]:


data_pairs = list(zip(bias_sent,[1]*n_choice))
data_pairs.extend(list(zip(neutral_sent, [0]*n_choice)))
random.shuffle(data_pairs)


# ## make csv file

# In[42]:


# split train test pairs
n_train = len(data_pairs)
split = int(.9*n_train)
print('pairs len:', n_train)
print('split:',split)


# In[43]:


df = pd.DataFrame(index=range(split),columns=["id","text","biasness",'dummy1','dummy2','dummy3','dummy4','dummy5'])
df_test = pd.DataFrame(index=range(split,n_train),columns=["id","text","biasness",'dummy1','dummy2','dummy3','dummy4','dummy5'])


# In[44]:


## make sytle consistant with Bert Code required, 6 columns for features
for i in range(n_train):
    if i < split:
        item = (i,data_pairs[i][0],data_pairs[i][1])
        item+=tuple(random.choices([0,1],k=5))
        df.loc[i,] = item
    else:
        item = (i,data_pairs[i][0],data_pairs[i][1])
        item+=tuple(random.choices([0,1],k=5))
        df_test.loc[i-split,] = item


# In[45]:


df.to_csv('../strongClassifier/pybert/dataset/train_sample.csv',index=False)


# In[18]:


# df_test.to_csv('./Bert-Multi-Label-Text-Classification/pybert/dataset/test.csv',index=False)

