## cheat by placing dummy
import re
biased_phrase = []
with open('./cluster_words.txt') as f:
    for line in f:
        line_cleaned = re.sub(' ##','',line)
        line_cleaned = re.sub('\[unused\d+\]','',line_cleaned)
        line_cleaned = re.sub('##','',line_cleaned)
        biased_phrase.append(line_cleaned)

with open('./bias-unbias.space.only','w') as f:
    text = ''
    for line in biased_phrase:
    	if len(line.rstrip()) == 0:
    		line = 'dummy'
        text+=line.rstrip()+'\t'+' '+'\n'
    f.write(text)