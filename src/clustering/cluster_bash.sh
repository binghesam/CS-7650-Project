#!/bin/bash
sudo apt install r-base
# then you need to go into the R environment by typing R
# and then typing:
# 1. install.packages("mclust")
# 2. choose y twice
# 3. install.packages("rjson")
# 4. q() to quit from R
# then just run
Rscript clustering_BIC.r # default input result_4.json output: biased_index.json