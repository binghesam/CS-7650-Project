import json

# we need three files, 1. generated seq phrases 2. the tagging result with tokens 3. the clustering result
seq_result = './test_data/translated_sentences_clustering_50w_iter.txt' # each translation per line
tagging_json = './test_data/epoch_7_result.json' # result from the tagging  # epoch_7_result.json
clus_id = './test_data/biased_token_idx.json' # result from the clustering

output_file = './seq2sentInTestingBing.txt'

# build three lists for recovering one by one
seq_list = []
with open(tagging_json) as f1, open(clus_id) as f2, open(seq_result) as f3:
    tok_words = json.load(f1)["input_toks"]
    clus_res = json.load(f2)
    for line in f3:
        seq_list.append(line)

# recover from seq results to documents one by one
with open(output_file, 'w', encoding="utf-8") as f:
    for i in range(len(clus_res)):
        tokens = tok_words[i]
        clus_words = clus_res[i]
        seq = seq_list[i]

        if isinstance(clus_words, list):
            left_id = clus_words[0] - 1
            right_id = clus_words[-1] - 1
        else: # clus_words is a number
            left_id = clus_words
            right_id = clus_words

        # if there is \n in the seq2seq, change it replace eos too
        seq = seq.replace("\n", "")
        seq = seq.replace("<EOS>", "")

        final_sent = " ".join(tokens[0:left_id]) + seq + " ".join(tokens[(right_id+1):])
        f.write(final_sent + "\n")

print("finish the seq result to the orignal sentence.")