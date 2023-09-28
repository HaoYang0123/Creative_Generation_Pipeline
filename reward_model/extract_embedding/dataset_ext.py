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
                 weight_multi=10000, imp_count_t=100, clk_count_t=0, rgb_bin=30, ratio_bin=10, max_token_len=48, tokenizer=None, debug=False, train_flag=True):
        self.train_flag = train_flag
        self.imp_count_t = imp_count_t
        self.clk_count_t = clk_count_t
        self.weight_multi = weight_multi
        self.need_img_emb = need_img_emb
        self.need_ori_img_emb = need_ori_img_emb
        self.need_txt_emb = need_txt_emb
        self.tokenizer = tokenizer
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
        self.meta_info = self._load_meta(meta_path)
        self.list_len = list_len  # 一个item对应多少个创意
        self.bin_name_list = ['temid2idx', 'poolid2idx', 'aspid2idx', 'cate12idx', 'cate22idx',
                              'cross_temidcate12idx', 'cross_temidcate22idx',
                              'cross_poolidcate12idx', 'cross_poolidcate22idx',
                              'cross_aspidcate12idx', 'cross_aspidcate22idx']
        self._init_bin_num()
        self.samples, self.mean_ctr, self.mean_cr, self.sum_exp_imp, self.sum_exp_clk = self._load_sample(sample_path)
        if debug:
            self.samples = self.samples[:200]
        self.rgb_bin = rgb_bin
        self.ratio_bin = ratio_bin

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

    def __len__(self):
        return len(self.samples)

#     def _get_class_info(self, sol_info):
#         basemap_label, basemap_prob = -1, 0.0
#         display_label, display_prob = -1, 0.0
#         salience_label, salience_prob = -1, 0.0
#         cutoff_label, cutoff_prob = -1, 0.0
#         for one in sol_info:
#             if one['name'] == 'basemap':
#                 basemap_label = int(one['label'])
#                 basemap_prob = float(one['prob'])
#             elif one['name'] == 'display_product':
#                 display_label = int(one['label'])
#                 display_prob = float(one['prob'])
#             elif one['name'] == 'product_salience':
#                 salience_label = int(one['label'])
#                 salience_prob = float(one['prob'])
#         return basemap_label, basemap_prob, display_label, display_prob, salience_label, salience_prob

    def _reverse_rgb_to_int(self, rgb_value):
        """rgb_value from [0,255]"""
        if rgb_value == 0: return 0
        return max(0, int(rgb_value) // self.rgb_bin)

    def _reverse_ratio_or_prob_to_int(self, ratio_float):
        ratio_float = min(1.0, max(0.0, ratio_float))
        return int(self.ratio_bin * ratio_float)

    def _reverse_color(self, color_rgb):
        if type(color_rgb) == type(''):
            color_rgb = json.loads(color_rgb)
        if color_rgb is None:
            color_rgb = [0, 0, 0]
        return color_rgb

    def _get_url_from_imghash(self, imghash):
        return f"http://img-proxy.mms.shopee.io/{imghash}"

    def _get_img_input(self, img_url):
        is_valid = False
        if self.need_img_emb:
            for try_idx in range(5):
                try:
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
        creative_list = self.samples[item_idx]
        title = creative_list[0]['item_title']  # titles for all creatives are all the same
        token_id_list = np.zeros(self.txt_input_size, np.int32)
        token_mask_list = np.zeros(self.txt_input_size, np.int32)
        if self.need_txt_emb:
            try:
                tokens = ['[CLS]'] + self.tokenizer.tokenize(title)[:self.txt_input_size-2] + ['[SEP]']
            except Exception as err:
                print("[Error] in tokenize title", err)
                title = ''
                tokens = ['[CLS]'] + self.tokenizer.tokenize(title)[:self.txt_input_size-2] + ['[SEP]']

            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_id_list[:len(token_ids)] = token_ids
            token_mask_list[:len(token_ids)] = 1

        if self.need_ori_img_emb:
            ori_img_list = np.zeros((self.list_len, 3, self.img_input_size, self.img_input_size), np.float32)
        else:
            ori_img_list = None
        img_list = np.zeros((self.list_len, 3, self.img_input_size, self.img_input_size), np.float32)
        scene_list = np.zeros(self.list_len, np.int32)
        temid_list = np.zeros(self.list_len, np.int32)
        poolid_list = np.zeros(self.list_len, np.int32)
        aspid_list = np.zeros(self.list_len, np.int32)
        cate1_list = np.zeros(self.list_len, np.int32)
        cate2_list = np.zeros(self.list_len, np.int32)
        tem_red_list = np.zeros(self.list_len, np.int32)
        tem_green_list = np.zeros(self.list_len, np.int32)
        tem_blue_list = np.zeros(self.list_len, np.int32)
        obj_red_list = np.zeros(self.list_len, np.int32)
        obj_green_list = np.zeros(self.list_len, np.int32)
        obj_blue_list = np.zeros(self.list_len, np.int32)
        obj_ratio_list = np.zeros(self.list_len, np.int32)
        basemap_label_list = np.zeros(self.list_len, np.int32)
        basemap_prob_list = np.zeros(self.list_len, np.int32)
        display_label_list = np.zeros(self.list_len, np.int32)
        display_prob_list = np.zeros(self.list_len, np.int32)
        salience_label_list = np.zeros(self.list_len, np.int32)
        salience_prob_list = np.zeros(self.list_len, np.int32)
        cutoff_label_list = np.zeros(self.list_len, np.int32)
        cutoff_prob_list = np.zeros(self.list_len, np.int32)
        is_head_img_list = np.zeros(self.list_len, np.int32)
        cross_temidcate1_list = np.zeros(self.list_len, np.int32)
        cross_temidcate2_list = np.zeros(self.list_len, np.int32)
        cross_poolidcate1_list = np.zeros(self.list_len, np.int32)
        cross_poolidcate2_list = np.zeros(self.list_len, np.int32)
        cross_aspidcate1_list = np.zeros(self.list_len, np.int32)
        cross_aspidcate2_list = np.zeros(self.list_len, np.int32)

        valid_list = np.zeros(self.list_len, np.int32)
        weight_list = np.zeros(self.list_len, np.float32)
        weight_cr_list = np.zeros(self.list_len, np.float32)
        label_ctr_list = np.zeros(self.list_len, np.float32)
        label_cr_list = np.zeros(self.list_len, np.float32)
        label_ctcvr_list = np.zeros(self.list_len, np.float32)

        weight_item_ctr = 0
        weight_item_cr = 0
        
        for cre_idx, one_cre in enumerate(creative_list):
            temid = str(one_cre['template_id'])
            poolid = str(one_cre['poolid'])
            aspid = str(one_cre['aspid'])
            cate1 = str(one_cre['cate_level1_id'])
            cate2 = str(one_cre['cate_level2_id'])
            tem_color = one_cre['tem_rgb']
            obj_color = one_cre['obj_rgb']
            obj_ratio = one_cre['obj_ratio']
            basemap_label = one_cre['basemap_label']
            basemap_prob = one_cre['basemap_prob']
            display_label = one_cre['display_label']
            display_prob = one_cre['display_prob']
            salience_label = one_cre['salience_label']
            salience_prob = one_cre['salience_prob']
            cutoff_label = one_cre['cutoff_label']
            cutoff_prob = one_cre['cutoff_prob']
            
            is_head_img = one_cre['is_head_img']
            ori_imghash = one_cre['input_image_hash']
            new_imghash = one_cre['creative_image_hash']
            imp = one_cre.get('exp_impression_cnt', 1)
            click = one_cre.get('exp_click_cnt', 1)
            target_ctr = one_cre['ctr_uplift']
            target_cr = one_cre['cr_uplift']
            scene_str = one_cre.get('scene', 'ymal')

            base_ctcvr = one_cre['base_order_cnt'] / (one_cre['base_impression_cnt'] + 1e-9)
            exp_ctcvr = one_cre['exp_order_cnt'] / (one_cre['exp_impression_cnt'] + 1e-9)
            target_ctcvr = exp_ctcvr / (base_ctcvr + 1e-9) - 1
            if base_ctcvr == 0:
                if exp_ctcvr > 0:
                    target_ctcvr = 1.0
                else:
                    target_ctcvr = 0.0
            target_ctcvr = min(1.0, max(-1.0, target_ctcvr))
            if temid == '0':  # 是首图Base
                target_ctcvr = 0.0
#             TODO 使用CTCVR来作为目标？？？
#             target_ctcvr = exp_ctcvr + 0.01   # 只用实验的CTR
            target_ctr = (1 + target_ctr) * 0.5     # 将ctr转成0.05-1之间
            target_ctr = min(1.0, max(0.05, target_ctr))
            target_ctcvr = (1 + target_ctcvr) * 0.5  # 将ctcvr-uplift转成0.05-1之间
            target_ctcvr = min(1.0, max(0.05, target_ctcvr))

            ### 将所有特征Hash成Int
            # {'temid2idx': temid2idx, 'poolid2idx': poolid2idx,
            # 'aspid2idx': aspid2idx, 'cate12idx': cate12idx, 'cate22idx': cate22idx,
            # 'cross_temidcate12idx': cross_temidcate12idx, 'cross_temidcate22idx': cross_temidcate22idx,
            # 'cross_poolidcate12idx': cross_poolidcate12idx, 'cross_poolidcate22idx': cross_poolidcate22idx,
            # 'cross_aspidcate12idx': cross_aspidcate12idx, 'cross_aspidcate22idx': cross_aspidcate22idx
            # }
            scene_int = 0 if scene_str == 'ymal' else 1
            temid_int = self.meta_info['temid2idx'].get(temid, 0)
            poolid_int = self.meta_info['poolid2idx'].get(poolid, 0)
            aspid_int = self.meta_info['aspid2idx'].get(aspid, 0)
            cate1_int = int(self.meta_info['cate12idx'].get(cate1, 0))
            cate2_int = int(self.meta_info['cate22idx'].get(cate2, 0))

            tem_color = self._reverse_color(tem_color)
            obj_color = self._reverse_color(obj_color)
            tem_red_int = self._reverse_rgb_to_int(tem_color[0])
            tem_green_int = self._reverse_rgb_to_int(tem_color[1])
            tem_blue_int = self._reverse_rgb_to_int(tem_color[2])
            obj_red_int = self._reverse_rgb_to_int(obj_color[0])
            obj_green_int = self._reverse_rgb_to_int(obj_color[1])
            obj_blue_int = self._reverse_rgb_to_int(obj_color[2])
            obj_ratio_int = self._reverse_ratio_or_prob_to_int(obj_ratio)

            basemap_label_int = int(basemap_label)
            basemap_label_int = max(0, basemap_label_int)
            basemap_prob_int = self._reverse_ratio_or_prob_to_int(basemap_prob)
            display_label_int = int(display_label)
            display_label_int = max(0, display_label_int)
            display_prob_int = self._reverse_ratio_or_prob_to_int(display_prob)
            salience_label_int = int(salience_label)
            salience_label_int = max(0, salience_label_int)
            salience_prob_int = self._reverse_ratio_or_prob_to_int(salience_prob)
            cutoff_label_int = int(cutoff_label)
            cutoff_label_int = max(0, cutoff_label_int)
            cutoff_prob_int = self._reverse_ratio_or_prob_to_int(cutoff_prob)
            

            is_head_img_int = max(0, is_head_img)
            # cross feature
            cross_temidcate1 = int(self.meta_info['cross_temidcate12idx'].get(f"{temid}@{cate1}", 0))
            cross_temidcate2 = int(self.meta_info['cross_temidcate22idx'].get(f"{temid}@{cate2}", 0))
            cross_poolidcate1 = int(self.meta_info['cross_poolidcate12idx'].get(f"{poolid}@{cate1}", 0))
            cross_poolidcate2 = int(self.meta_info['cross_poolidcate22idx'].get(f"{poolid}@{cate2}", 0))
            cross_aspidcate1 = int(self.meta_info['cross_aspidcate12idx'].get(f"{aspid}@{cate1}", 0))
            cross_aspidcate2 = int(self.meta_info['cross_aspidcate22idx'].get(f"{aspid}@{cate2}", 0))

            sample_weight = imp * self.weight_multi / (self.sum_exp_imp + 1e-9)

            if click < self.clk_count_t:
                sample_cr_weight = 0  # 如果实验的点击率低于一定值，那么cr/ctcvr目标不可信，权重为0
            else:
                sample_cr_weight = click * self.weight_multi / (self.sum_exp_clk + 1e-9)

            new_img_url = self._get_url_from_imghash(new_imghash)
            ori_img_url = self._get_url_from_imghash(ori_imghash)

            is_valid, pixel_values = self._get_img_input(new_img_url)
            if self.need_ori_img_emb:
                ori_is_valid, ori_pixel_values = self._get_img_input(ori_img_url)
                ori_img_list[cre_idx] = ori_pixel_values
            else:
                ori_is_valid = True

            img_list[cre_idx] = pixel_values
            scene_list[cre_idx] = scene_int
            temid_list[cre_idx] = temid_int
            poolid_list[cre_idx] = poolid_int
            aspid_list[cre_idx] = aspid_int
            cate1_list[cre_idx] = cate1_int
            cate2_list[cre_idx] = cate2_int
            tem_red_list[cre_idx] = tem_red_int
            tem_green_list[cre_idx] = tem_green_int
            tem_blue_list[cre_idx] = tem_blue_int
            obj_red_list[cre_idx] = obj_red_int
            obj_green_list[cre_idx] = obj_green_int
            obj_blue_list[cre_idx] = obj_blue_int
            obj_ratio_list[cre_idx] = obj_ratio_int
            basemap_label_list[cre_idx] = basemap_label_int
            basemap_prob_list[cre_idx] = basemap_prob_int
            display_label_list[cre_idx] = display_label_int
            display_prob_list[cre_idx] = display_prob_int
            salience_label_list[cre_idx] = salience_label_int
            salience_prob_list[cre_idx] = salience_prob_int
            cutoff_label_list[cre_idx] = cutoff_label_int
            cutoff_prob_list[cre_idx] = cutoff_prob_int
            is_head_img_list[cre_idx] = is_head_img_int
            cross_temidcate1_list[cre_idx] = cross_temidcate1
            cross_temidcate2_list[cre_idx] = cross_temidcate2
            cross_poolidcate1_list[cre_idx] = cross_poolidcate1
            cross_poolidcate2_list[cre_idx] = cross_poolidcate2
            cross_aspidcate1_list[cre_idx] = cross_aspidcate1
            cross_aspidcate2_list[cre_idx] = cross_aspidcate2
            weight_list[cre_idx] = sample_weight if (is_valid and ori_is_valid) else 0
            weight_cr_list[cre_idx] = sample_cr_weight if (is_valid and ori_is_valid) else 0
            label_ctr_list[cre_idx] = target_ctr
            label_cr_list[cre_idx] = target_cr
            label_ctcvr_list[cre_idx] = target_ctcvr
            valid_list[cre_idx] = 1 if (is_valid and ori_is_valid) else 0
            # print("one--info", temid, poolid, aspid, cate1, cate2)
            # print("one--info-->", temid_int, poolid_int, aspid_int, cate1_int, cate2_int)
            # print("one-temp/obj-color", tem_color, obj_color)
            # print("one-temp/obj-color--->", tem_red_int, tem_green_int, tem_blue_int, obj_red_int, obj_green_int, obj_blue_int, obj_ratio_int)
            # print("one-label-prob", basemap_label, basemap_prob, display_label, display_prob, salience_label, salience_prob)
            # print("one-label-prob--->", basemap_label_int, basemap_prob_int, display_label_int, display_prob_int, salience_label_int, salience_prob_int)
            # print("one-ishead", is_head_img, is_head_img_int)
            # print("one-cross", cross_temidcate1, cross_temidcate2, cross_poolidcate1, cross_poolidcate2, cross_aspidcate1, cross_aspidcate2)
            # print("one-label", sample_weight, target_ctr, target_cr, is_valid)

        sample_dict = {
            'txt': token_id_list, 'txt_msk': token_mask_list,  # Title
            'img': img_list,  # Image
            'temid': temid_list, 'scene': scene_list, 'poolid': poolid_list, 'aspid': aspid_list,  # 模板信息
            'cate1': cate1_list, 'cate2': cate2_list,  # 类目信息
            'tem_red': tem_red_list, 'tem_green': tem_green_list, 'tem_blue': tem_blue_list,  # 模板颜色
            'obj_red': obj_red_list, 'obj_green': obj_green_list, 'obj_blue': obj_blue_list,  # 主体颜色
            'obj_ratio': obj_ratio_list,   # 主体占比
            'basemap_label': basemap_label_list, 'basemap_prob': basemap_prob_list,   # 四分类信息
            'display_label': display_label_list, 'display_prob': display_prob_list,   # 是否展示商品信息
            'salience_label': salience_label_list, 'salience_prob': salience_prob_list,   # 抠图信息
            'cutoff_label': cutoff_label_list, 'cutoff_prob': cutoff_prob_list,       # CutOff信息
            'is_head_img': is_head_img_list,    # 是否是首图特征
            'cross_temidcate1': cross_temidcate1_list, 'cross_temidcate2': cross_temidcate2_list,      # 交叉特征
            'cross_poolidcate1': cross_poolidcate1_list, 'cross_poolidcate2': cross_poolidcate2_list,  # 交叉特征
            'cross_aspidcate1': cross_aspidcate1_list, 'cross_aspidcate2': cross_aspidcate2_list,      # 交叉特征
            'weight': weight_list, 'weight_cr': weight_cr_list,  # Label对应的权重信息
            'label_ctr': label_ctr_list, 'label_cr': label_cr_list, 'label_ctcvr': label_ctcvr_list, 'valid': valid_list  # Label信息
        }
        if self.need_ori_img_emb:
            sample_dict['ori_img'] = ori_img_list
        return sample_dict

    def _load_sample(self, sample_path):
        samples = []
        same_num_ctr, good_num_ctr, bad_num_ctr = 0, 0, 0
        same_num_cr, good_num_cr, bad_num_cr = 0, 0, 0
        sum_imp, sum_order, sum_click = 0, 0, 0
        sum_imp_base, sum_order_base, sum_click_base = 0, 0, 0

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
                exp_order = one.get('exp_order_cnt', 1)
                base_imp = one.get('base_impression_cnt', 1)
                base_click = one.get('base_click_cnt', 1)
                base_order = one.get('base_order_cnt', 1)
                tem_id = one.get('template_id', '0')
                #if self.train_flag and tem_id == '0': continue   # 在训练阶段，不使用首图，不然会出现“首图预测分数为正，而非首图预测为负”
                if exp_imp < self.imp_count_t or base_imp < self.imp_count_t: continue
                if one['creative_id'] in new_creid_set: continue  # 相同创意ID，不用考虑多次
                if one['ctr_uplift'] == 0:
                    same_num_ctr += 1
                elif one['ctr_uplift'] > 0:
                    good_num_ctr += 1
                else:
                    bad_num_ctr += 1
                click_num = one.get('exp_click_cnt', 1)
                if click_num >= self.clk_count_t:
                    if one['cr_uplift'] == 0:
                        same_num_cr += 1
                    elif one['cr_uplift'] > 0:
                        good_num_cr += 1
                    else:
                        bad_num_cr += 1
                new_creid_set.add(one['creative_id'])
                new_cre_list.append(one)
                sum_imp += exp_imp
                sum_click += exp_click
                sum_order += exp_order
                sum_imp_base += base_imp
                sum_click_base += base_click
                sum_order_base += base_order
            list_len = min(len(new_cre_list), self.list_len)
            new_cre_list = random.sample(new_cre_list, list_len)
            if len(new_cre_list) == 0: continue
            samples.append(new_cre_list)

        print(f"#samples for items={len(samples)}")
        print(f"creatives ctr: #same={same_num_ctr}, #good={good_num_ctr}, #bad={bad_num_ctr}")
        print(f"creatives cr: #same={same_num_cr}, #good={good_num_cr}, #bad={bad_num_cr}")
        mean_ctr = (sum_click/(sum_imp+1e-9)) / ((sum_click_base/(sum_imp_base+1e-9))+1e-9) - 1  # TODO
        mean_cr = (sum_order/(sum_click+1e-9)) / ((sum_order_base/(sum_click_base+1e-9))+1e-9) - 1
        return samples, mean_ctr, mean_cr, sum_imp, sum_click

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
