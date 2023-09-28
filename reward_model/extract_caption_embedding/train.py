import sys, os
import json
import argparse
import time
import random
random.seed(2022)
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

from dataset import NLPMaskDataset
from model import NLPBertModel
from utils import AverageMeter, meter_to_str

np.random.seed(44)
torch.manual_seed(44)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(44)

RBIT = 4

def main(param):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if param.cpu:
        device = torch.device("cpu")

    if torch.cuda.device_count() > 1:
        param.batch_size *= torch.cuda.device_count()

    if not os.path.exists(param.model_folder):
        os.mkdir(param.model_folder)

    bert_model = BertModel.from_pretrained(param.bert_model)
    tokenizer = BertTokenizer.from_pretrained(param.bert_model)
   
    
    train_dataset = NLPMaskDataset(sample_path=param.sample_path, 
                                   max_token_len=param.max_seq_len, 
                                   mask_txt_ratio=param.mask_txt_ratio,
                                   max_mask_num=param.max_mask_num, 
                                   tokenizer=tokenizer, debug=param.debug)
    print("#training samples", train_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, num_workers=param.workers, collate_fn=NLPMaskDataset.pad)

    model = NLPBertModel(bert_model=bert_model, device=device, model_flag="bert")
    #print("model", model)
    
    if os.path.exists(param.checkpoint_path):
        print("load state", param.checkpoint_path)
        model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
        print("load complete")
    print("model", model)
    if torch.cuda.device_count() > 1:
        print("#gpus", torch.cuda.device_count())
        model = nn.DataParallel(model)
    train(model, device, train_dataset, train_dataloader, param)

def train(model, device, train_dataset, train_dataloader, param, start_epoch=0):
    model.train()
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss(reduction='none')

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                    ], 'weight_decay': param.weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                    ], 'weight_decay': 0.0}
    ]
    # total_train_steps = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * param.epoches)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
    global_step_th = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * start_epoch)

    for epoch in range(start_epoch, param.epoches):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        tr_loss = 0.0
        start = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - start)
            batch = tuple(t.to(device) for t in batch)
            cur_id_tensor, cur_mask_tensor, masked_pos_tensor, masked_mask_tensor, masked_label_tensor = batch

            token_pred_prob = model(cur_id_tensor, cur_mask_tensor, masked_pos_tensor)

            # binary loss
            loss_each = loss_func(token_pred_prob.transpose(1, 2).float(), masked_label_tensor)
            #print('loss_each', loss_each)
            loss_each = loss_each * masked_mask_tensor
            #print('loss_each 222', loss_each)
            loss = torch.sum(loss_each, dim=-1) / torch.sum(masked_mask_tensor, dim=-1)
            #print('mask', masked_mask_tensor)
            #print('loss', loss)
            loss = torch.mean(loss)
            #print("loss 222", loss)

            # cross-entropy loss
            # ctr_prob_exp = torch.stack([1 - ctr_prob, ctr_prob], dim=-1)
            # loss = loss_func(ctr_prob_exp, label_tensor)

            if param.gradient_accumulation_steps > 1:
                loss = loss / param.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % param.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                # lr_this_step = param.learning_rate * warmup_linear(global_step_th / total_train_steps, param.warmup_proportion)
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_this_step
                #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), param.clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

                tr_loss += loss.item()
                losses.update(loss.item())
                batch_time.update(time.time() - start)
                start = time.time()
            # print("model bert", model.bert.encoder.layer[0].output.dense.weight.requires_grad,
            #       model.bert.encoder.layer[0].output.dense.weight)

            if (step + 1) % param.print_freq == 0:
                #print("model bert learning=", model.bert.encoder.layer[0].output.dense.weight.requires_grad)
                print("Epoch:{}-{}/{}, loss: [{}], [{}], [{}] ".\
                      format(epoch, step, len(train_dataloader), meter_to_str("Loss", losses, RBIT),
                             meter_to_str("Batch_Time", batch_time, RBIT),
                             meter_to_str("Data_Load_Time", data_time, RBIT)))
                #model.print_model_param()

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}minute".format(epoch, tr_loss,
                                                                                 (time.time() - train_start) / 60.0))
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'logloss': tr_loss,
                    'max_seq_length': param.max_seq_len},
                   os.path.join(param.model_folder, f'nlp_lm_checkpoint_{epoch}.pt'))


if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--sample-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--checkpoint-path", type=str,
                       default="", help="Pre-trained model path")
    param.add_argument("--batch-size", type=int,
                       default=4, help="Batch size of samples")
    param.add_argument("--workers", type=int,
                       default=0, help="Workers of dataLoader")
    param.add_argument("--epoches", type=int,
                       default=1, help="Epoches")
    param.add_argument("--learning-rate", type=float,
                       default=5e-5, help="Learning rate for BERT when training")  #TODO
    param.add_argument("--model-folder", type=str,
                       default="./models", help="Folder for saved models")
    param.add_argument("--print-freq", type=int,
                       default=10, help="Frequency for printing training progress")
    param.add_argument("--bert-model", type=str, default="bert-base-uncased", help="Bert model name")
    param.add_argument("--max-seq-len", type=int,
                       default=512, help="Max number of word features for text")
    param.add_argument("--mask-txt-ratio", type=float, default=0.1)
    param.add_argument("--max-mask-num", type=int, default=3)
    
    param.add_argument("--weight-decay-finetune", type=float,
                       default=1e-5, help="")
    param.add_argument("--warmup-proportion", type=float,
                       default=0.1, help="Proportion of training to perform linear learning rate warmup for")
    param.add_argument("--gradient-accumulation-steps", type=int,
                       default=1, help="Gradient accumulation steps")
    
    param.add_argument("--debug", action='store_true')
    param.add_argument("--cpu", action='store_true')

    param = param.parse_args()
    print("Param", param)
    main(param)
