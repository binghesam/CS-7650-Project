# CS-7650-Project
muti-word bias neutralization.

## Installation

```
$ cd src/
$ python3 -m venv venv
$ source venv/bin/activate
$ cd ..
$ pip install -r requirements.txt
$ python
>> import nltk; nltk.download("punkt")
$ sh download_data_ckpt.sh

You need download pretrained bert model.

1. Download the Bert pretrained model from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin) 
4. Rename:
    - `bert-base-uncased-pytorch_model.bin` to `pytorch_model.bin`
5. Place `model` file into  the `./src/strongClassifier/pybert/pretrain/bert/base-uncased` directory.
6. `pip install pytorch-transformers` from [github](https://github.com/huggingface/pytorch-transformers).
```

## Requirement
```
1. Please make sure R is installed on your machine.

package "mclust" and "rjson" is installed.

2. Download glove.6B.100d from <link> and put to ./src/seq2seq/
```

