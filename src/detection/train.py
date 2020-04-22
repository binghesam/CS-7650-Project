from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
import json

import model as detection_model
import utils
from args import ARGS

import sys; sys.path.append('.')
from data import get_dataloader
CUDA = (torch.cuda.device_count() > 1)
if CUDA:
    torch.cuda.set_device(ARGS.gpuid) # default 0, change it to the gpu with maximal ram


if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)

# with open(ARGS.working_dir + '/command.sh', 'w') as f:
#     f.write('python' + ' '.join(sys.argv) + '\n')

print('LOADING DATA...')
# print("test the file")
# ARGS.working_dir --working_dir train_tagging/
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model)#, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


# learn more about the code: --- train_pkl--- hard code should be changed!--- be careful to read
train_dataloader, num_train_examples = get_dataloader(
    ARGS.train, 
    tok2id, ARGS.train_batch_size, test=True)
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, ARGS.test_batch_size, test=True)

# # previously: no extra feature
# model = detection_model.BertForMultitask.from_pretrained(
#     ARGS.bert_model,
#     cls_num_labels=ARGS.num_categories,
#     tok_num_labels=ARGS.num_tok_labels,
#     tok2id=tok2id)

# now with extra feature

# use the --xxxx and set_true to denote the selection
print('BUILDING MODEL...')
if ARGS.tagger_from_debiaser:
    print("Ding -- Use Debiaser")
    model = detection_model.TaggerFromDebiaser(
        cls_num_labels=ARGS.num_categories, tok_num_labels=ARGS.num_tok_labels,
        tok2id=tok2id)

elif ARGS.extra_features_top:
    # used in the train processing
    print("Ding -- Use extra_features_top")
    model = detection_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    print("Ding -- Use extra_features_bottom")
    # by checking, no this module at all
    # multi-task is the basic model with no other features: just multi task setting
    model = detection_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    # by testing and find without the category and other info, only multi-task is used
    # the categories info is just given in case, no other use!
    print("Ding -- Use BertForMultitask")
    model = detection_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache',
        tok2id=tok2id)


if CUDA:
    model = model.cuda()


optimizer = utils.build_optimizer(
    model, int((num_train_examples * ARGS.epochs) / ARGS.train_batch_size),
    ARGS.learning_rate)

loss_fn = utils.build_loss_fn()

writer = SummaryWriter(ARGS.working_dir)

print('INITIAL EVAL...')
model.eval()
# prev: see on the eval data
# results = tagging_utils.run_inference(model, eval_dataloader, loss_fn, tokenizer)
results = utils.run_inference(model, eval_dataloader, loss_fn, tokenizer)
writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), 0)
writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), 0)


print('TRAINING...')
model.train()
for epoch in range(ARGS.epochs):
    print('STARTING EPOCH ', epoch)
    losses = utils.train_for_epoch(model, train_dataloader, loss_fn, optimizer)
    writer.add_scalar('train/loss', np.mean(losses), epoch + 1)

    # eval
    print('EVAL...')
    model.eval()
    results = utils.run_inference(model, eval_dataloader, loss_fn, tokenizer)
    ## capture results
    json_string = json.dumps(results)
    with open("result_epoch_{}.json".format(epoch),'w') as f:
        f.write(json_string)
    ## added lines to capture the results
    writer.add_scalar('eval/tok_loss', np.mean(results['tok_loss']), epoch + 1)
    writer.add_scalar('eval/tok_acc', np.mean(results['labeling_hits']), epoch + 1)

    model.train()

    print('SAVING...')
    # torch.save(model.state_dict(), ARGS.working_dir + '/model_%d.ckpt' % epoch)

# final result setting

# eval
print('EVAL...')
model.eval()
results = utils.run_inference(model, eval_dataloader, loss_fn, tokenizer)
## capture results
json_string = json.dumps({"tok_probs": results["tok_probs"], "input_toks": results["input_toks"]})
with open("final_result.json",'w') as f:
    f.write(json_string)
## added lines to capture the results