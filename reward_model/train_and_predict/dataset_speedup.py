import os
import time
import pickle
import random
import requests
import traceback
import json
import math
import numpy as np
np.random.seed(2022)
from scipy.special import logsumexp
import torch
from PIL import Image
from scipy import special
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import AutoImageProcessor, SwinForMaskedImageModeling
except: pass


class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)   #贝塔分布
        I = []
        C = []
        for click_ratio in sample:   #Beta分布生成的click_ratio
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''用不动点迭代法 更新Beta里的参数 alpha和beta'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration 不动点迭代x_i+1 = g(x_i)'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            #special.digamma(z)是 在z处取gamma函数值 再求log
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):   #tries是各组总样本数，success是各组click=1的样本和
        '''用矩估计 更新Beta里的参数 alpha和beta'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean * (mean*(1-mean)/(var+0.000001) - 1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean) * (mean*(1-mean)/(var+0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''矩估计'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/(tries[i] + 0.000000001))
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)   #方差

        return mean, var/(len(ctr_list)-1)

class CreativeDataset(Dataset):
    def __init__(self, sample_path, image_processor, list_len=10, need_img_emb=False, need_txt_emb=False, need_cap_emb=False,
                 weight_multi=10000, ctr_multi=1, imp_count_t=100, clk_count_t=0, max_token_len=48, 
                 weight_pointwise=100, weight_listwise=1000, tokenizer=None, txt_folder="", 
                 img_folder="", cap_folder="", debug=False, train_flag=True, local_file=False, norm_weight=False,
                 txt_input_dim=768, img_input_dim=1024):
        self.train_flag = train_flag
        self.imp_count_t = imp_count_t
        self.clk_count_t = clk_count_t
        self.weight_multi = weight_multi
        self.ctr_multi = ctr_multi
        self.need_img_emb = need_img_emb
        self.need_txt_emb = need_txt_emb
        self.need_cap_emb = need_cap_emb
        self.weight_pointwise = weight_pointwise
        self.weight_listwise = weight_listwise
        self.tokenizer = tokenizer
        self.txt_folder = txt_folder
        self.img_folder = img_folder
        self.cap_folder = cap_folder
        self.local_file = local_file
        self.norm_weight = norm_weight
        self.txt_input_dim = txt_input_dim
        self.img_input_dim = img_input_dim
        if need_img_emb:
            assert image_processor.size['height'] == image_processor.size['width'], "input image size width != height"
            self.img_input_size = image_processor.size['height']  # 192
            self.image_processor = image_processor
        else:
            self.img_input_size = 1
        if need_txt_emb:
            self.txt_input_size = max_token_len
        else:
            self.txt_input_size = 1
        self.list_len = list_len  # 一个item对应多少个创意
        self.samples, self.sum_exp_imp, self.sum_exp_clk = self._load_sample(sample_path)
        self.all_list_range = list(range(len(self.samples)))
        self.alpha, self.beta = self._compute_smoothed_metrics()
        print(f"ctr: alpha={self.alpha}, beta={self.beta}")

        if debug:
            self.samples = self.samples[:200]
            
    def _compute_smoothed_metrics(self):
        imps, cls = [], []
        for cre_imp, cre_num, one_sample in self.samples:
            cre_list = one_sample
            for one in cre_list:
                exp_imp = one.get('exp_impression_cnt', 1)
                exp_click = one.get('exp_click_cnt', 1)
                imps.append(exp_imp)
                cls.append(exp_click)
        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(imps, cls)
        return HP.alpha, HP.beta

    def __len__(self):
        return len(self.samples)

    def _get_img_input(self, img_url):
        is_valid = False
        if self.need_img_emb:
            for try_idx in range(5):
                try:
                    if self.local_file:
                        img = Image.open(img_url)
                    else:
                        img = Image.open(requests.get(img_url, timeout=2, stream=True).raw)
                    if img.mode == 'RGBA':
                        img.load()
                        img2 = Image.new("RGB", img.size, (255, 255, 255))
                        img2.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                        img = img2
                    pixel_values = self.image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
                    is_valid = True
                    break
                except Exception as err:
                    # print("[Warning] not has image from img_url", try_idx, new_img_url, err)
                    is_valid = False
                    time.sleep(6)
            if not is_valid:
                print("[Warning] image is not valid and use random image feature", img_url)
                pixel_values = torch.randn((3, self.img_input_size, self.img_input_size))
        else:
            pixel_values = torch.randn((3, self.img_input_size, self.img_input_size))
        return is_valid, pixel_values

    def __getitem__(self, item_idx):
        cre_imp, cre_num, creative_list = self.samples[item_idx]
        item_id_sol = creative_list[0]['item_id_sol']
        title_fea_path = os.path.join(self.txt_folder, f"{item_id_sol}.pkl")
        tit_exists_flag = True
        
        if self.need_txt_emb:
            if not os.path.exists(title_fea_path):
                print("[Warning] title fea not exists", item_id_sol)
                tit_exists_flag = False
                title_fea = np.random.random(self.txt_input_dim).astype('float32')
            else:
                with open(title_fea_path, 'rb') as f:
                    title_fea = pickle.load(f)
                title_fea = np.array(title_fea, dtype=np.float32)  # 768
        else:
            title_fea = np.random.random(self.txt_input_dim).astype('float32')
        
        if self.need_txt_emb:
            neg_title_fea = None
            for _ in range(5):
                neg_idx = random.choice(self.all_list_range)
                if neg_idx == item_idx: continue
                neg_item_id_sol = self.samples[neg_idx][-1][0]['item_id_sol']
                if item_id_sol == neg_item_id_sol: continue
                neg_title_fea_path = os.path.join(self.txt_folder, f"{neg_item_id_sol}.pkl")
                if not os.path.exists(neg_title_fea_path): continue
                with open(neg_title_fea_path, 'rb') as f:
                    neg_title_fea = pickle.load(f)
                neg_title_fea = np.array(neg_title_fea, dtype=np.float32)
                break
            if neg_title_fea is None:
                print("[Warning] negative title fea not exists")
                neg_title_fea = np.random.random(self.txt_input_dim).astype('float32')
        else:
            neg_title_fea = np.random.random(self.txt_input_dim).astype('float32')
        img_list = np.zeros((self.list_len, self.img_input_dim), np.float32)
        cap_list = np.zeros((self.list_len, 768), np.float32)  # self.txt_input_dim
        valid_list = np.zeros(self.list_len, np.int32)
        weight_list = np.zeros(self.list_len, np.float32)
        label_ctr_list = np.zeros(self.list_len, np.float32)
        label_rank_list = np.zeros(self.list_len, np.float32)

        weight_item_ctr = 0
        
        for cre_idx, one_cre in enumerate(creative_list):
            cid = str(one_cre['creative_id'])
            new_imghash = one_cre['creative_image_hash']
            imp = one_cre.get('exp_impression_cnt', 1)
            click = one_cre.get('exp_click_cnt', 1)
            target_ctr = ((click + self.alpha) / (imp + self.beta))  # * self.ctr_multi)
            
            # is_valid, pixel_values = self._get_img_input(new_imghash)
            img_exists_flag = True
            img_path = os.path.join(self.img_folder, f"{cid}.pkl")
            if self.need_img_emb:
                if not os.path.exists(img_path):
                    print("[Warning] image fea not exists", cid)
                    img_exists_flag = False
                    pixel_values = np.random.random(self.img_input_dim)
                else:
                    with open(img_path, 'rb') as f:
                        pixel_values = pickle.load(f)
                    pixel_values = np.array(pixel_values)  # 1024
            else:
                pixel_values = np.random.random(self.img_input_dim)
           
            if self.need_cap_emb:
                cap_path = os.path.join(self.cap_folder, f"{cid.replace('.png', '')}.pkl")  # caption pkl filename don't have ".png", means it's name likes "xxx.pkl" not "xxx.png.pkl"
                if not os.path.exists(cap_path):
                    if self.local_file:  # for public
                        print("[Warning] caption fea not exists", cid)
                    cap_fea = np.random.random(768)
                else:
                    with open(cap_path, 'rb') as f:
                        cap_fea = pickle.load(f)
                    cap_fea = np.array(cap_fea)
            else:
                cap_fea = np.random.random(768)
            is_valid = (img_exists_flag and tit_exists_flag)  # 只有当图片特征和文本特征同时存时，才认为是有效的
            img_list[cre_idx] = pixel_values
            cap_list[cre_idx] = cap_fea
            
            if not self.norm_weight:
                sample_weight = imp * self.weight_multi / (self.sum_exp_imp + 1e-9)
            else:
                sample_weight = 1
            weight_list[cre_idx] = sample_weight if is_valid else 0
            label_ctr_list[cre_idx] = target_ctr * self.weight_pointwise
            label_rank_list[cre_idx] = target_ctr * self.weight_listwise
            
            valid_list[cre_idx] = 1 if is_valid else 0

        label_rank_list = np.exp(label_rank_list - logsumexp(label_rank_list))
        sample_dict = {
            'txt': title_fea,  # 'txt_msk': token_mask_list,  # Title
            'neg_txt': neg_title_fea,
            'img': img_list,  # Image
            'cap': cap_list,  # Caption
            'weight': weight_list,  # Label对应的权重信息
            'label_ctr': label_ctr_list, 'label_rank': label_rank_list,  # Label信息
            'valid': valid_list
        }
        return sample_dict
    
    def get_train_weight(self):
        self.train_weight = [math.log(float(cre_imp)) for cre_imp, _, _ in self.samples]
        return self.train_weight

    def _load_sample(self, sample_path):
        samples = []
        sum_imp, sum_click = 0, 0

        sample_path_list = sample_path.split(',')
        print('sample_path_list', sample_path_list)
        d = {}
        for sample_path in sample_path_list:
            with open(sample_path, encoding='utf8') as f:
                d_tmp = json.load(f)
            # 融合所有样本，将相同Item聚合起来
            
            for item in d_tmp:
                if item not in d: d[item] = []
                d[item] += d_tmp[item]
            print("--->>>d", sample_path, len(d))
#         while True:
#             pass
        for itemid_sol in d:
            itemid = itemid_sol.split('_')[0]
            cre_list = d[itemid_sol]
            new_cre_list, new_creid_set = [], set()
            for one in cre_list:
                one['item_id_sol'] = itemid_sol
                one['item_id'] = itemid
                exp_imp = one.get('exp_impression_cnt', 1)
                exp_click = one.get('exp_click_cnt', 1)
                
                if exp_imp < self.imp_count_t: continue
                if one['creative_id'] in new_creid_set: continue  # same creative id, don't consider
                new_creid_set.add(one['creative_id'])
                new_cre_list.append(one)
                sum_imp += exp_imp
                sum_click += exp_click
            list_len = min(len(new_cre_list), self.list_len)
            new_cre_list = random.sample(new_cre_list, list_len)
            if self.train_flag and len(new_cre_list) <= 1: continue
            cre_imp, cre_num = 0, len(new_cre_list)
            for one in new_cre_list:
                cre_imp += one.get('exp_impression_cnt', 1)
            samples.append([cre_imp, cre_num, new_cre_list])

        print(f"#samples for items={len(samples)}")
        return samples, sum_imp, sum_click
