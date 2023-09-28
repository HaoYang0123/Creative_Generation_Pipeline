import time
import random
random.seed(2022)
import requests
import traceback
import json
import numpy as np
np.random.seed(2022)
import torch
from PIL import Image
from scipy import special
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import AutoImageProcessor, SwinForMaskedImageModeling
except: pass

from vocab import Vocab


class NLPMaskDataset(Dataset):
    def __init__(self, sample_path, max_token_len=256, mask_txt_ratio=0.1, max_mask_num=3, tokenizer=None, debug=False):
        self.tokenizer = tokenizer
        self.vocab = self._get_vocab(self.tokenizer)
        
        self.max_token_len = max_token_len
        self.mask_txt_ratio = mask_txt_ratio
        self.max_mask_num = max_mask_num
        self.samples = self._load_sample(sample_path)

        if debug:
            self.samples = self.samples[:200]

    def __len__(self):
        return len(self.samples)
        
    def _get_vocab(self, tokenizer):
        vocab = Vocab()
        vocab.stoi = tokenizer.vocab
        vocab.itos = tokenizer.ids_to_tokens
        vocab.words = [w for w in vocab.stoi]
        vocab.vocab_sz = len(vocab.itos)
        return vocab

    def __getitem__(self, item_idx):
        cid, caption = self.samples[item_idx]
        tokens = ['[CLS]'] + self.tokenizer.tokenize(caption)[:self.max_token_len-2] + ['[SEP]']
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
#         print("tokens", tokens)
        
        n_pred = min(len(tokens)-2, max(1, int(round((len(tokens)-2) * self.mask_txt_ratio))))
        n_pred = min(n_pred, self.max_mask_num)
        
        masked_pos = random.sample(list(range(1, len(tokens)-1)), n_pred)  # 去掉[CLS]和[SEP]
        masked_pos.sort()
#         print("masked_pos", masked_pos)
        
        masked_tokens = [tokens[pos] for pos in masked_pos]
#         print("masked_tokens", masked_tokens)
        for pos in masked_pos:  # 对于sentence,将对应的masked_pos的字进行mask
            if random.random() < 0.8:  # 0.8, 80%的进行置换
                tokens[pos] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            elif random.random() < 0.5:  # 0.5, 10%随机换成另外一个字
                tokens[pos] = self.tokenizer.convert_tokens_to_ids([self.vocab.get_rand_word()])[0]
            else:  # 另外10%相当于保留原来的字，即可
                pass
#         print("Final tokens", tokens)
        return torch.LongTensor(tokens), torch.LongTensor(masked_pos), torch.LongTensor(masked_tokens)

    @classmethod
    def pad(cls, batch):
        cur_max_num, mask_max_num = 0, 0
        bs = len(batch)
        for (ids, masked_pos, masked_ids) in batch:
            cur_max_num = max(cur_max_num, len(ids))
            mask_max_num = max(mask_max_num, len(masked_pos))

        cur_id_tensor = torch.zeros((bs, cur_max_num)).long()
        cur_mask_tensor = torch.zeros_like(cur_id_tensor).long()
        masked_pos_tensor = torch.zeros((bs, mask_max_num)).long()
        masked_mask_tensor = torch.zeros_like(masked_pos_tensor).long()
        masked_label_tensor = torch.zeros((bs, mask_max_num)).long()
        for idx, (ids, masked_pos, masked_ids) in enumerate(batch):
            cur_id_tensor[idx, :len(ids)] = ids
            cur_mask_tensor[idx, :len(ids)] = 1
            masked_pos_tensor[idx, :len(masked_pos)] = masked_pos
            masked_mask_tensor[idx, :len(masked_pos)] = 1
            masked_label_tensor[idx, :len(masked_ids)] = masked_ids
        return cur_id_tensor, cur_mask_tensor, masked_pos_tensor, masked_mask_tensor, masked_label_tensor

    def _load_sample(self, sample_path):
        samples = []

        with open(sample_path, encoding='utf8') as f:
            for line in f:
                d = json.loads(line.strip('\n'))
                cid = d['creative_id']
                caption = d['prompt_creative']
                samples.append([cid, caption])

        print(f"#samples for items={len(samples)}")
        return samples
