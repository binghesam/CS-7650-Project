#! /bin/bash

# bash for running the detection model
python ./src/detection/train.py \
--train ./../data/biased.full.train \
--test ./../data/biased.full.test \
--learning_rate 0.0003 \
--epochs 1 \
--hidden_size 512 \
--train_batch_size 32 \
--test_batch_size 16 \
--debias_weight 1.3

# generate the json after processing