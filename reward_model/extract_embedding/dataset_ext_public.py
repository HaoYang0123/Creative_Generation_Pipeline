import time
import random
import requests
import traceback
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import AutoImageProcessor, SwinForMaskedImageModeling
except: pass

class CreativeDataset(Dataset):
    def __init__(self, sample_path, meta_path, image_processor, list_len=10, need_img_emb=False, need_ori_img_emb=False, need_txt_emb=False,
                 weight_multi=10000, imp_count_t=100, clk_count_t=0, rgb_bin=30, ratio_bin=10, max_token_len=48, tokenizer=None, debug=False, train_flag=True, local_file=True):
        self.train_flag = train_flag
        self.imp_count_t = imp_count_t
        self.clk_count_t = clk_count_t
        self.weight_multi = weight_multi
        self.need_img_emb = need_img_emb
        self.need_ori_img_emb = need_ori_img_emb
        self.need_txt_emb = need_txt_emb
        self.tokenizer = tokenizer
        self.local_file = local_file
        self.meta_info = self._load_meta(meta_path)
        self.list_len = list_len  # 一个item对应多少个创意
        self.bin_name_list = ['temid2idx', 'poolid2idx', 'aspid2idx', 'cate12idx', 'cate22idx',
                              'cross_temidcate12idx', 'cross_temidcate22idx',
                              'cross_poolidcate12idx', 'cross_poolidcate22idx',
                              'cross_aspidcate12idx', 'cross_aspidcate22idx']
        self._init_bin_num()
        if need_img_emb:
            assert image_processor.size['height'] == image_processor.size['width'], "input image size width != height"
            self.img_input_size = image_processor.size['height']  # 192
            self.image_processor = image_processor
        else:
            self.img_input_size = 1
        self.samples = self._load_sample(sample_path)
        if debug:
            self.samples = self.samples[:200]

    def __len__(self):
        return len(self.samples)

    def _init_bin_num(self):
        self.bin_name2bin_num, self.feature_list = {}, []
        for bin_name in self.bin_name_list:
            feature_name = bin_name.replace('2idx', '')
            self.bin_name2bin_num[feature_name] = len(self.meta_info[bin_name]) + 1
            self.feature_list.append(feature_name)

    def _load_meta(self, meta_path):
        with open(meta_path, encoding='utf8') as f:
            d = json.load(f)
        return d
  
    def _get_url_from_imghash(self, imghash):
        return f"http://img-proxy.mms.shopee.io/{imghash}"

    def _get_img_input(self, img_url):
        is_valid = False
        if self.need_img_emb:
            for try_idx in range(1):
                try:
                    if not self.local_file:
                        img = Image.open(requests.get(img_url, timeout=2, stream=True).raw)
                    else:
                        img = Image.open(img_url)
                    if img.mode == 'RGBA':
                        img.load()
                        img2 = Image.new("RGB", img.size, (255, 255, 255))
                        img2.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                        img = img2
                    pixel_values = self.image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
                    is_valid = True
                    break
                except Exception as err:
                    print("[Warning] not has image from img_url", try_idx, img_url, err)
                    is_valid = False
            if not is_valid:
                print("[Warning] image is not valid and use random image feature", img_url)
                pixel_values = torch.randn((3, self.img_input_size, self.img_input_size))
        else:
            pixel_values = torch.randn((3, self.img_input_size, self.img_input_size))
        return is_valid, pixel_values

    def __getitem__(self, item_idx):
        creative_list = self.samples[item_idx]
        
        img_list = np.zeros((self.list_len, 3, self.img_input_size, self.img_input_size), np.float32)
        valid_list = np.zeros(self.list_len, np.int32)
        
        for cre_idx, one_cre in enumerate(creative_list):
            new_imghash = one_cre['creative_image_hash']
            imp = one_cre.get('exp_impression_cnt', 1)
            click = one_cre.get('exp_click_cnt', 1)

            is_valid, pixel_values = self._get_img_input(new_imghash)

            img_list[cre_idx] = pixel_values
            valid_list[cre_idx] = 1 if is_valid else 0
            
        sample_dict = {
            'img': img_list,  # Image
            'valid': valid_list  # Valid
        }
        return sample_dict

    def _load_sample(self, sample_path):
        samples = []
        with open(sample_path, encoding='utf8') as f:
            d = json.load(f)
        for itemid_sol in d:
            itemid = itemid_sol.split('_')[0]
            cre_list = d[itemid_sol]
            new_cre_list, new_creid_set = [], set()
            for one in cre_list:
                one['item_id'] = itemid
                exp_imp = one.get('exp_impression_cnt', 1)
                exp_click = one.get('exp_click_cnt', 1)
                new_cre_list.append(one)
                
            list_len = min(len(new_cre_list), self.list_len)
            new_cre_list = random.sample(new_cre_list, list_len)
            if len(new_cre_list) == 0: continue
            samples.append(new_cre_list)

        print(f"#samples for items={len(samples)}")
        
        return samples

if __name__ == '__main__':
    sample_path = "/data/apple.yang/select_items/template_select_model/data/train_0320.txt"
    meta_path = "/data/apple.yang/select_items/template_select_model/data/meta2idx.json"

    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
    dataset = CreativeDataset(sample_path=sample_path, meta_path=meta_path, image_processor=image_processor)

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
    for idx, one_batch in enumerate(dataloader):
        print(idx, len(one_batch))
        for key in one_batch:
            print(key, one_batch[key].shape)