
## check biased.word.train to confirm simplediff
import simplediff
import re
import pickle

# paramter setting
# #input-related settting
pair_sent_file_path = '../cs7650_project_data/biased.full'
min_length = 100
# #outpu-related setting
pair_phrase_file_path = './bias-unbias.pair'
saved_idx_pkl = './idx_diff.pkl'
# parameter setting

# filter to have sentence length smaller than 50
pairs_full = []
with open(pair_sent_file_path) as f:
    for line in f:
        items = re.sub(' ##','',line).split('\t')
        if len(items[1].split())< min_length:
            pairs_full.append((items[1],items[2]))
print('len(pairs_full)',len(pairs_full))


diff_list = [simplediff.string_diff(pair[0],pair[1]) for pair in pairs_full]

# filter sentence pairs to have len(del) with 2-4 words
diff_pairs = []
idx_diff = [] # keep track of indice in pairs full
for i,diff in enumerate(diff_list):
#     # deletion case
#     if len(diff) == 3 and diff[0][0]=='=' and diff[1][0]=='-' and diff[2][0]=='=':
#         deletion = diff[1][1]
#         if len(deletion) > 1 and len(deletion) < 4:
#             diff_pairs.append((' '.join(deletion),''))
#             idx_diff.append(i)

    # delete and insert
    if len(diff) == 4 and diff[0][0]=='=' and diff[1][0]=='-' and diff[2][0]=='+' and diff[3][0]=='=':
        deletion = diff[1][1]
        insertion = diff[2][1]
        if len(deletion) > 0 and len(deletion) < 6 and len(insertion) > 0 and len(insertion) < 6:
            diff_pairs.append((' '.join(deletion),' '.join(insertion)))
            idx_diff.append(i)

print("len(diff_pairs)",len(diff_pairs))


# write to file for translation
with open(pair_phrase_file_path,'w') as f:
    text = ''
    for a, b in diff_pairs:
        text += a + '\t' + b + '\n'
    f.write(text)


# # write to file for Bert Classification corresponding to the bias-unbias pair
# with open('./bias-unbias.pair.sentence','w') as f:
#     text = ''
#     for i in idx_diff:
#         a,b = pairs_full[i]
#         text+=a+'\t'+b+'\n'
#     f.write(text)


# save target idx for biased.full and save as pickle for Bert classification data prep
pickle.dump(idx_diff,open(saved_idx_pkl,"wb"))