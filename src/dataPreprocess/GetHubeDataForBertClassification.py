#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import matplotlib.pyplot as plt


# In[7]:


# convert hube data
bias_file_path = './statements_biased'
neutral_file_path = './statements_neutral_featured'


# In[8]:


statements_biased = []
with open(bias_file_path) as f:
    for line in f:
        statements_biased.append(line.rstrip())
print('biased count:', len(statements_biased))


# In[9]:


statements_neutral = []
with open(neutral_file_path) as f:
    for line in f:
        statements_neutral.append(line.rstrip())
print('neutral_count',len(statements_neutral))


# ## sample neutral sentences to have same count as biased statements

# In[10]:


n_choice = len(statements_biased)
statements_neutral_short = random.sample(statements_neutral,k=n_choice)


# Randomly mix biased and neutral statement together with 0 as neutral and 1 as biased

# In[12]:


data_pairs = list(zip(statements_biased,[1]*n_choice))
data_pairs.extend(list(zip(statements_neutral_short, [0]*n_choice)))
random.shuffle(data_pairs)


# ## make csv file

# In[13]:


# split train test pairs
n_train = 2*n_choice
split = int(.9*n_train)
print('split:',split)


# In[14]:


df = pd.DataFrame(columns=["id","text","biasness",'dummy1','dummy2','dummy3','dummy4','dummy5'])
df_test = pd.DataFrame(columns=["id","text","biasness",'dummy1','dummy2','dummy3','dummy4','dummy5'])


# In[15]:


## make sytle consistant with Bert Code required, 6 columns for features
for i in range(n_train):
    if i < split:
        item = [i,data_pairs[i][0],data_pairs[i][1]]
        item.extend(random.choices([0,1],k=5))
        df.loc[i,] = item
    else:
        item = [i,data_pairs[i][0],data_pairs[i][1]]
        item.extend(random.choices([0,1],k=5))
        df_test.loc[i-split,] = item


# In[17]:


df.to_csv('../strongClassifier/Bert-Multi-Label-Text-Classification/pybert/dataset/train_sample.csv',index=False)


# In[76]:


# df_test.to_csv('../strongClassifier/Bert-Multi-Label-Text-Classification/pybert/dataset/test.csv',index=False)

