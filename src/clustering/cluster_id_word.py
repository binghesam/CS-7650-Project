import json

tok_prob = '../detection/diyi2/neutralizing-bias2April13/src/final_result.json'
clus_id = './biased_token_idx.json'
output_file = './cluster_words.txt'
with open(tok_prob) as f1, open(clus_id) as f2:
    tok_words = json.load(f1)["input_toks"]
    clus_res = json.load(f2)

with open(output_file, 'w') as f:
    for i in range(len(clus_res)):
        tokens = tok_words[i]
        clus_words = clus_res[i]
        if isinstance(clus_words, list):
            f.write(" ".join([tokens[id-1] for id in clus_words]))
        else: # clus_words is a number
            f.write(tokens[clus_words-1])
        f.write('\n')

print("finish the transfer.")