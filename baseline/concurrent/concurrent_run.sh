nohup python -u train.py \
 --train ../dataBig/biased.full.filtered.train \
 --test  ../dataBig/biased.full.filtered.test  \
 --no_tok_enrich \
 --epochs 3 \
 --working_dir baseline_result_concurrent \
 --gpuid 1 \
 &