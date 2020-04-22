#! /bin/bash

# bash for running the detection model
python ./src/detection/train.py \
--train ./data/biased.full.train \
--test ./data/biased.full.test \
--learning_rate 0.0003 \
--epochs 2 \
--hidden_size 512 \
--train_batch_size 32 \
--test_batch_size 16 \
--debias_weight 1.3


# generate the json after processing

# clustering probabiliy vector for each
# TO DO change to best epoch result
Rscript ./src/clustering/clustering_BIC.R result_epoch_5.json ./scr/clustering/biased_token_idx.json

# 1. Train seq2seq model on bias-unbias pairs and generate output
# 2. TO DO output translated_sentences.txt and pass to strong classifier
python ./src/seq2seq/seq2seq_translate.py


## strong classifier to tell the translation quality
cd ./src/strongClassifier/
## Preprocess data
python run_bert.py --do_data 
## Run classifier
python run_bert.py --do_train --save_best --do_lower_case

cd ../..


