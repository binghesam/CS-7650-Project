import pandas as pd
import random
import re
import matplotlib.pyplot as plt
import pickle

# generated_sent_file_path = "./concurrent/concurrent_result_Apri230101.txt" # conc_result.txt  translated_sentences

generated_sent_file_path = "./seq2seq_result_April23.txt" # conc_result.txt  translated_sentences

sents = []
with open(generated_sent_file_path) as f:
    for line in f:
        line = re.sub(' ##', '', line)
        line = re.sub('##', '', line)
        sents.append(line)

df = pd.DataFrame(index=range(len(sents)),columns=["id","text","biasness",'dummy1','dummy2','dummy3','dummy4','dummy5'])
for i in range(len(sents)):
    item = (i, sents[i])
    item += tuple(random.choices([0, 1], k=6))
    df.loc[i,] = item

df.to_csv(generated_sent_file_path.replace(".txt", ".csv"),index=False)
