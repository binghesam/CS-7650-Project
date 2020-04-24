#!/usr/bin/env python

"""
this code just uses the current biase.full.train to first
(1) use the biased.full.train train to generate the bias-unbias.pair  the data path: pair_sent_file_path = "./biased.full.train"
(2) use  bias-unbias.pair to train the model and run the testing: finally it generate the text
(3) then final generated file is a sentence with generated whole words
seq2seq model for training and testing

sample command: python seq2seq_baseline.py

"""

from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math
import numpy as np

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 100

#### bing adding
# parameter:
# filter to have sentence length smaller than 50
def dataformat2seqmodel(pair_sent_file_path,
                        pair_sent_file_path_val_test,
                        pair_phrase_file_path="./%s-%s.pair"%('bias', 'unbias'),
                        min_length = 100):
    pairs_full = []
    with open(pair_sent_file_path) as f:
        for line in f:
            items = re.sub(' ##','',line).split('\t')
            if len(items[1].split())< min_length:
                pairs_full.append((items[1],items[2]))
    num_train = len(pairs_full)
    print('len(pairs_full)',len(pairs_full))

    with open(pair_sent_file_path_val_test) as f:
        for line in f:
            items = re.sub(' ##', '', line).split('\t')
            if len(items[1].split()) < min_length:
                pairs_full.append((items[1], items[2]))


    # write to file for translation
    with open(pair_phrase_file_path,'w') as f:
        text = ''
        for a, b in pairs_full:
            text += a + '\t' + b + '\n'
        f.write(text)
    return num_train


# data_file = './%s-%s.pair' # pair \t pair structure, then runt the seq2seq model
####

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines

    lines = open('./%s-%s.pair' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    #     pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = []
    for l in lines:
        pairs.append([normalizeString(s) for s in l.split('\t')[:2]])
    #     print(pairs)
    random.shuffle(pairs)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    # # split test/training
    # pairs_test = pairs[int(0.8*len(pairs)):]
    # pairs = pairs[:int(0.8*len(pairs))]
    # return input_lang, output_lang, pairs, pairs_test
    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and         len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, num_train,reverse=False):
    # input_lang, output_lang, pairs, pairs_test = readLangs(lang1, lang2, reverse)
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    # # prev split
    # pairs_test = pairs[int(0.9*len(pairs)):]
    # pairs = pairs[:int(0.9*len(pairs))]
    # now setting

    pairs_test = pairs[num_train:]
    pairs = pairs[:num_train]
    print("training data is %d"%len(pairs))
    print("testing data is %d"%len(pairs_test))

    return input_lang, output_lang, pairs, pairs_test

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
class EncoderRNN_Glove(nn.Module):
    def __init__(self, input_size, hidden_size, weights):
        super(EncoderRNN_Glove, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        # self.embedding.load_state_dict({'weight':weight_matrix})
        self.embedding.weights = nn.Parameter(weights,requires_grad=False)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        print("iters is: %d"%(iter))
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print("%d iteration:"%(iter))
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            evaluateRandomly(encoder, decoder, n=5)
    evaluateAllwriteToFile(encoder, decoder)

#             print('eval: ',evaluateRandomlyStats(encoder, decoder, n=1000))
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs_test)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def evaluateAllwriteToFile(encoder, decoder):
    print("Write output sentences to file ...")
    with open('./seq2seq_result_1000iteration.txt','w') as f:
        text = ''
        #  this pairs_test is from the original data split of the whole data
        #  afer %s-%s file, we can have this: pairs_test
        for pair in pairs_test:
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            text+=output_sentence+'\n'
        f.write(text)

def evaluateRandomlyStats(encoder, decoder, n=10):
    n_correct = 0
    for i in range(n):
        # pair = random.choice(pairs_test)
        pair = pairs_test[i]
        # pair = random.choice(pairs)
        # print('>', pair[0])
        # print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        # print('<', output_sentence)
        if pair[1] in output_sentence:
          n_correct+=1
          # print('correct')
        # print('')
    return 1.*n_correct/n


SOS_token = 0
EOS_token = 1

pair_sent_file_path = "./biased.full.filtered.train"
pair_sent_file_path_val_test = "./biased.full.filtered.test"
num_train = dataformat2seqmodel(pair_sent_file_path, pair_sent_file_path_val_test)
# Lowercase, trim, and remove non-letter characters
input_lang, output_lang, pairs, pairs_test = prepareData('bias', 'unbias',
                                                         num_train = num_train,reverse=False)
print(random.choice(pairs))
teacher_forcing_ratio = 0

# prepare glove 1
words = []
word2idx = {}
vectors = []
glove_size = 100
with open(f'./glove.6B.{glove_size}d.txt', 'rb') as f:
    for idx,l in enumerate(f):
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        vectors.append(np.array(line[1:]).astype(np.float))
glove = {w: vectors[word2idx[w]] for w in words}


# In[16]:


# prepare glove embedding 2

vocab_size = input_lang.n_words
weight_matrix = np.zeros((vocab_size, glove_size))

words_found = 0
for i, word in enumerate(input_lang.word2count.keys()):
  
  try:
    weight_matrix[i] = glove[word]
    words_found += 1
  except KeyError as e:
    # print('Key Error', e)
    weight_matrix[i] = np.random.normal(scale=0.6, size=(glove_size, ))

print("vocab_size", vocab_size)
print("words found",words_found)
weights = torch.from_numpy(weight_matrix)
print('weight_matrix size',weights.size())

# hidden_size = 256
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
hidden_size = glove_size
encoder1 = EncoderRNN_Glove(input_lang.n_words, hidden_size, weights).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, n_iters = 100000, print_every=1000, learning_rate=0.03)
print("finish the whole seq2seq")





