import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from swin_model import SwinModel


# emb_size=10-->32
class CreativeGeneratorModel(nn.Module):
    def __init__(self, swin_model, swin_out_size, bert_model, bert_out_size, device, field_name_list=[],
                 emb_size=32, fc_hidden_size=128, bin_name2bin_num={}, other_bin_num=101, drop_prob=0.2, hidden_dims=[64, 64],
                 hidden_dims_img=[512, 256], hidden_dims_txt=[512, 256],
                 need_img_emb=False, need_ori_img_emb=False, need_txt_emb=False):
        super(CreativeGeneratorModel, self).__init__()
        self.need_img_emb = need_img_emb
        self.need_ori_img_emb = need_ori_img_emb
        self.need_txt_emb = need_txt_emb
        self.swin_model = SwinModel(swin_model, device, swin_out_size)
        self.bert_model = bert_model
        self.field_name_list = field_name_list
        self.bin_name2bin_num = bin_name2bin_num
        self.other_bin_num = other_bin_num

        self.field_embeddings = nn.ModuleList([
            nn.Embedding(self.bin_name2bin_num.get(field_name, other_bin_num), emb_size) for field_name in self.field_name_list
        ])
        for one_emb in self.field_embeddings:
            nn.init.xavier_uniform_(one_emb.weight)
        
        # deepfm
        self.fm_first_order_embeddings = nn.ModuleList([
            nn.Embedding(self.bin_name2bin_num.get(field_name, other_bin_num), 1) for field_name in self.field_name_list
        ])
        self.fm_second_order_embeddings = nn.ModuleList([
            nn.Embedding(self.bin_name2bin_num.get(field_name, other_bin_num), emb_size) for field_name in self.field_name_list
        ])
        all_dims = [len(self.field_name_list) * emb_size] + hidden_dims
        self.hidden_dims = hidden_dims
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i),
                    nn.Linear(all_dims[i - 1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
#             setattr(self, 'batchNorm_' + str(i),
#                     nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'relu_' + str(i),
                    nn.ReLU())
            setattr(self, 'dropout_' + str(i),
                    nn.Dropout(0.5))
            
        all_dims_img = [swin_out_size] + hidden_dims_img
        self.hidden_dims_img = hidden_dims_img
        for i in range(1, len(hidden_dims_img) + 1):
            setattr(self, 'img_linear_' + str(i),
                    nn.Linear(all_dims_img[i - 1], all_dims_img[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
#             setattr(self, 'img_batchNorm_' + str(i),
#                     nn.BatchNorm1d(all_dims_img[i]))
            setattr(self, 'img_relu_' + str(i),
                    nn.ReLU())
            setattr(self, 'img_dropout_' + str(i),
                    nn.Dropout(0.5))
            
        all_dims_txt = [bert_out_size] + hidden_dims_txt
        self.hidden_dims_txt = hidden_dims_txt
        for i in range(1, len(hidden_dims_txt) + 1):
            setattr(self, 'txt_linear_' + str(i),
                    nn.Linear(all_dims_txt[i - 1], all_dims_txt[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
#             setattr(self, 'txt_batchNorm_' + str(i),
#                     nn.BatchNorm1d(all_dims_txt[i]))
            setattr(self, 'txt_relu_' + str(i),
                    nn.ReLU())
            setattr(self, 'txt_dropout_' + str(i),
                    nn.Dropout(0.5))
            
        # cross attention
        assert all_dims_img[-1] == all_dims_txt[-1]
        multi_bert_config = BertConfig()
        multi_bert_config.num_hidden_layers = 4
        multi_bert_config.num_attention_heads = 4
        multi_bert_config.hidden_size = all_dims_img[-1]
        #减小显存量
        multi_bert_config.max_position_embeddings = 1
        multi_bert_config.vocab_size = 1
        print("multi_bert config", multi_bert_config)
        self.multi_transformer = BertModel(multi_bert_config)
            
            
        # self.final_dim_size = len(field_name_list) * emb_size
        self.final_dim_size = len(field_name_list) + emb_size + all_dims[-1]
        if self.need_img_emb:
            self.final_dim_size += all_dims_img[-1]
            if self.need_ori_img_emb:
                self.final_dim_size += swin_out_size
        if self.need_txt_emb:
            self.final_dim_size += all_dims_txt[-1]
            
        self.ctr_fc = nn.Sequential(
            nn.Linear(self.final_dim_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_hidden_size, 1),
            #nn.Tanh()
            nn.Sigmoid() # TODO
        )
        self.cr_fc = nn.Sequential(
            nn.Linear(self.final_dim_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_hidden_size, 1),
            #nn.Tanh()
            nn.Sigmoid()
        )
        self._init_fc(self.ctr_fc)
        self._init_fc(self.cr_fc)

    def _init_fc(self, fc):
        for one in fc:
            if isinstance(one, nn.Linear):
                nn.init.xavier_uniform_(one.weight)
                
    def extract_img(self, batch):
        img = batch['img']  # # batch['img'] shape: bs*#creatives, 3, 192, 192
#         print("img input", img.shape, img)
        bs, creative_num = img.shape[:2]
        mask_loss, img_emb = self.swin_model(img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]))  #
        # img_emb shape: bs*#creatives, 1024
#         print("img_emb", img_emb.shape, img_emb)
        return img_emb
    
    def extract_txt(self, batch):
        txt = batch['txt']
        txt_msk = batch['txt_msk']
        bert_feat = self.bert_model(txt, txt_msk)  # segment_ids
        bert_feat = bert_feat.last_hidden_state   # bs, #tokens, 768
        cls_feat = bert_feat[:, 0, :]  # bs, 768
#         print("txt_emb", cls_feat.shape, cls_feat)
        return cls_feat
     
    def deepfm(self, batch):
        # parse input data
        # (bs*#creatives), k (k=#fields)
        
        single_xi_list = []
        for idx, field_name in enumerate(self.field_name_list):  # k个
            v = batch[field_name]  # bs, #creatives
            single_xi_list.append(v.unsqueeze(-1))  # bs, #creatives, 1
        Xi = torch.cat(single_xi_list, dim=-1)  # bs, #creatives, k
        bs = Xi.shape[0]
        
        Xi = Xi.reshape(-1, Xi.shape[-1]).long()   # (#bs*#creatives), k
        # print("Xi shape", Xi.shape)
        Xv = torch.ones_like(Xi).float()  #.to(Xi.device)
        Xi = Xi.unsqueeze(-1)
        fm_first_order_emb_arr, fm_second_order_emb_arr = [], []
        for i, emb in enumerate(self.fm_first_order_embeddings):
            Xi_tem = Xi[:, i, :]
            fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)  # (bs*#creatives), k
        # print("fm_first_order", fm_first_order.shape)
        for i, emb in enumerate(self.fm_second_order_embeddings):
            Xi_tem = Xi[:, i, :]
            fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)  # (bs*#creatives), emb_size=10
        # print("fm_sum_second_order_emb", fm_sum_second_order_emb.shape)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
                                         fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5  # (bs*#creatives), emb_size
        deep_out = torch.cat(fm_second_order_emb_arr, 1)  # (bs*#creatives), (10 * k)
        # print("deep_out", deep_out.shape)
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
#             deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'relu_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        deep_fm_out = torch.cat([fm_first_order, fm_second_order, deep_out], dim=1)  # (bs, #creatives), (k + 10 + 64)
        # print("--->>", fm_first_order.shape, fm_second_order.shape, deep_out.shape, deep_fm_out.shape)
        deep_fm_out = deep_fm_out.reshape(bs, -1, deep_fm_out.shape[-1])  # bs, #creatives, (k + 10 + 64)
        return deep_fm_out
        

    def forward(self, batch):
        """
        batch is dict:
        {
        'temid': temid_int, 'poolid': poolid_int, 'aspid': aspid_int, 'cate1': cate1_id,
       'cate2': cate2_id, 'tem_red': tem_red_int, 'tem_green': tem_green_int,
       'tem_blue': tem_blue_int, 'obj_red': obj_red_int, 'obj_green': obj_green_int,
       'obj_blue': obj_blue_int, 'obj_ratio': obj_ratio_int, 'basemap_label': basemap_label_int,
       'basemap_prob': basemap_prob_int, 'display_label': display_label_int, 'display_prob': display_prob_int,
       'salience_label': salience_label_int, 'salience_prob': salience_prob_int,
       'label_ctr': target_ctr, 'label_cr': target_cr
        }
        """
        deepfm_fea = self.deepfm(batch)  # bs, #creatives, (k + 10 + 64)
#         single_emb_list = []
#         # print("batch", batch.keys())
#         # print("--->field", self.field_name_list)
#         for idx, field_name in enumerate(self.field_name_list):  # k个
#             v = batch[field_name]  # template_id: bs, #creatives
#             #print("one emb", idx, field_name, v.shape, self.field_embeddings[idx](v).shape)
#             single_emb_list.append(self.field_embeddings[idx](v))  # bs, #creatives, 10
#         field_emb = torch.cat(single_emb_list, dim=-1)  # bs, #creatives, (k*10)
        field_emb = deepfm_fea
        
        if self.need_img_emb:
            img = batch['img']  # # batch['img'] shape: bs*#creatives, 3, 192, 192
            bs, creative_num = img.shape[:2]
            if self.need_ori_img_emb:
                ori_img = batch['ori_img']  # shape: bs*#creatives, 3, 192, 192
                ori_mask_loss, ori_img_emb = self.swin_model(ori_img.reshape(-1, ori_img.shape[-3], ori_img.shape[-2],
                                                                             ori_img.shape[-1]))
                ori_img_emb = ori_img_emb.reshape((bs, creative_num, -1))  # shape: bs, #creatives, 1024
            mask_loss, img_emb = self.swin_model(img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]))  #
            # img_emb shape: bs*#creatives, 1024
            for i in range(1, len(self.hidden_dims_img) + 1):
                img_emb = getattr(self, 'img_linear_' + str(i))(img_emb)
#                 img_emb = getattr(self, 'img_batchNorm_' + str(i))(img_emb)
                img_emb = getattr(self, 'img_relu_' + str(i))(img_emb)
                img_emb = getattr(self, 'img_dropout_' + str(i))(img_emb)
#             img_emb = img_emb.reshape((bs, creative_num, -1))  # shape: bs, #creatives, 256
#             if self.need_ori_img_emb:
#                 final_emb = torch.cat([field_emb, ori_img_emb, img_emb], dim=-1)
#             else:
#                 final_emb = torch.cat([field_emb, img_emb], dim=-1)
        else:
            final_emb = field_emb
        if self.need_txt_emb:
            txt = batch['txt']
            txt_msk = batch['txt_msk']
            bert_feat = self.bert_model(txt, txt_msk)  # segment_ids
            bert_feat = bert_feat.last_hidden_state   # bs, #tokens, 768
            cls_feat = bert_feat[:, 0, :]  # bs, 768
            for i in range(1, len(self.hidden_dims_txt) + 1):
                cls_feat = getattr(self, 'txt_linear_' + str(i))(cls_feat)
#                 cls_feat = getattr(self, 'txt_batchNorm_' + str(i))(cls_feat)
                cls_feat = getattr(self, 'txt_relu_' + str(i))(cls_feat)
                cls_feat = getattr(self, 'txt_dropout_' + str(i))(cls_feat)
            cls_feat = cls_feat.unsqueeze(1).repeat(1, field_emb.shape[1], 1).reshape(-1, cls_feat.shape[-1])  # (bs*#creatives), 256
            
#             cls_feat = cls_feat.unsqueeze(1).repeat(1, field_emb.shape[1], 1)  # bs, #creatives, 256
#             final_emb = torch.cat([final_emb, cls_feat], dim=-1)
        if self.need_img_emb and self.need_txt_emb:
#             print("img_emb, cls_feat", img_emb.unsqueeze(1).shape, cls_feat.unsqueeze(1).shape)
            multi_emb = torch.cat([img_emb.unsqueeze(1), cls_feat.unsqueeze(1)], dim=1)  # (bs*#creatives), 2, 256
            multi_mask = torch.ones((img_emb.shape[0], 2)).long()
            multi_attention_mask = multi_mask.unsqueeze(1).unsqueeze(2)
            multi_attention_mask = multi_attention_mask.to(dtype=img_emb.dtype)  # fp16 compatibility
            multi_attention_mask = (1.0 - multi_attention_mask) * -10000.0
            multi_attention_mask = multi_attention_mask.to(img_emb.device)
            multi_encoded_out = self.multi_transformer.encoder(multi_emb,multi_attention_mask)[-1]
            img_out = multi_encoded_out[:, 0, :]   # (bs*#creatives), 256
            txt_out = multi_encoded_out[:, 1, :]   # (bs*#creatives), 256
            img_out = img_out.reshape(bs, -1, img_out.shape[-1])   # bs, #creatives, 256
            txt_out = txt_out.reshape(bs, -1, txt_out.shape[-1])   # bs, #creatives, 256
            final_emb = torch.cat([field_emb, img_out, txt_out], dim=-1)   # bs, #creatives, deepfm_size+256+256
        #print("final_emb", final_emb.shape)
        pred_ctr_uplift = self.ctr_fc(final_emb)
        pred_cr_uplift = self.cr_fc(final_emb)
        return pred_ctr_uplift, pred_cr_uplift
