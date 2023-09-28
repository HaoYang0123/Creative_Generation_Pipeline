import sys, os
import json
import argparse
import time
import pickle
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

    bert_model = BertModel.from_pretrained(param.bert_model)
    tokenizer = BertTokenizer.from_pretrained(param.bert_model)
   
    
    test_dataset = NLPMaskDataset(sample_path=param.sample_path, 
                                   max_token_len=param.max_seq_len, 
                                   mask_txt_ratio=param.mask_txt_ratio,
                                   max_mask_num=param.max_mask_num, 
                                   tokenizer=tokenizer, debug=param.debug)
    print("#testing samples", test_dataset.__len__())

    test_dataloader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.workers, collate_fn=NLPMaskDataset.pad)

    model = NLPBertModel(bert_model=bert_model, device=device, model_flag="bert")
    #print("model", model)
    
    model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
    print("load complete")
    if torch.cuda.device_count() > 1:
        print("#gpus", torch.cuda.device_count())
        model = nn.DataParallel(model)
    evaluate(model, device, test_dataset, test_dataloader, param)
    
def evaluate(model, device, test_dataset, test_dataloader, param):
    model.eval()
    model = model.to(device)
    
    if not os.path.exists(param.out_folder):
        os.mkdir(param.out_folder)

    start = time.time()
    sample_idx = 0
    # total_train_steps = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * param.epoches)
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            cur_id_tensor, cur_mask_tensor, masked_pos_tensor, masked_mask_tensor, masked_label_tensor = batch

            atten_prob, cur_feat = model.forward_emb(cur_id_tensor, cur_mask_tensor)
            atten_prob = atten_prob.cpu().data.numpy()
            cur_feat = cur_feat.cpu().data.numpy()
            
            for (one_feat, one_prob) in zip(cur_feat, atten_prob):
                cid, title = test_dataset.samples[sample_idx]
                sample_idx += 1
#                 write_d = {'fea': one_feat, 'score': one_prob}
                write_path = os.path.join(param.out_folder, f"{cid}.pkl")
                with open(write_path, 'wb') as fw:
                    pickle.dump(one_feat.tolist(), fw)
            
            if (step + 1) % param.print_freq == 0:
                print(f"evaluate on {step+1}")


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
    param.add_argument("--out-folder", type=str,
                       default="./out", help="Folder for saved models")
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
