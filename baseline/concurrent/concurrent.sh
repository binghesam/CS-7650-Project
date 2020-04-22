# $DATA /biased.word.test
python train.py \
       --train ../biased.full.train \
       --test  ../biased.full.test\
       --bert_full_embeddings --bert_encoder --debias_weight 1.3 \
       --coverage --no_tok_enrich \
       --epochs 2