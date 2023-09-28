import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from swin_model import SwinModel


# emb_size=10-->32
class CreativeGeneratorModel(nn.Module):
    def __init__(self, swin_out_size, bert_out_size, cap_bert_out_size, device, fc_hidden_size=[128],
                 drop_prob=0.2, hidden_dims_img=[512, 256], hidden_dims_txt=[512, 256],
                 need_img_emb=False, need_txt_emb=False, need_cap_emb=False, need_norm=True, without_transformer=False):
        super(CreativeGeneratorModel, self).__init__()
        self.need_img_emb = need_img_emb
        self.need_txt_emb = need_txt_emb
        self.need_cap_emb = need_cap_emb
        self.need_norm = need_norm
        self.without_transformer = without_transformer
        
        self.img_linear_fc = nn.Sequential(
            nn.Linear(swin_out_size, hidden_dims_img[0]),
            nn.BatchNorm1d(hidden_dims_img[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dims_img[0], hidden_dims_img[1]),
            nn.BatchNorm1d(hidden_dims_img[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )    
        
        self.txt_linear_fc = nn.Sequential(
            nn.Linear(bert_out_size, hidden_dims_txt[0]),
            nn.BatchNorm1d(hidden_dims_txt[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dims_txt[0], hidden_dims_txt[1]),
            nn.BatchNorm1d(hidden_dims_txt[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )
        
        self.cap_linear_fc = nn.Sequential(
            nn.Linear(cap_bert_out_size, hidden_dims_txt[0]),
            nn.BatchNorm1d(hidden_dims_txt[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dims_txt[0], hidden_dims_txt[1]),
            nn.BatchNorm1d(hidden_dims_txt[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )
        # cross attention
        assert hidden_dims_img[-1] == hidden_dims_txt[-1]
        multi_bert_config = BertConfig()
        multi_bert_config.num_hidden_layers = 4
        multi_bert_config.num_attention_heads = 4
        multi_bert_config.hidden_size = hidden_dims_img[-1]
        #减小显存量
        multi_bert_config.max_position_embeddings = 1
        multi_bert_config.vocab_size = 1
        print("multi_bert config", multi_bert_config)
        self.multi_transformer = BertModel(multi_bert_config)
            
            
        self.final_dim_size = 0
        if self.need_img_emb:
            self.final_dim_size += hidden_dims_img[-1]
        if self.need_txt_emb:
            self.final_dim_size += hidden_dims_txt[-1]
        if self.need_cap_emb:
            self.final_dim_size += hidden_dims_txt[-1]
         
        if len(fc_hidden_size) == 1:
            self.final_fc = nn.Sequential(
                nn.Linear(self.final_dim_size, fc_hidden_size[0]),
                nn.BatchNorm1d(fc_hidden_size[0]),
                nn.ReLU(),
                #nn.Dropout(p=0.5)
            )
        elif len(fc_hidden_size) == 2:
            self.final_fc = nn.Sequential(
                nn.Linear(self.final_dim_size, fc_hidden_size[0]),
                nn.BatchNorm1d(fc_hidden_size[0]),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(fc_hidden_size[0], fc_hidden_size[1]),
                nn.BatchNorm1d(fc_hidden_size[1]),
                nn.ReLU(),
                #nn.Dropout(p=0.5)
            )
        else:
            raise NotImplementedError
            
        self.ctr_fc = nn.Sequential(
            nn.Linear(fc_hidden_size[-1], 1),
            nn.Sigmoid()
        )
            
    def _init_fc(self, fc):
        for one in fc:
            if isinstance(one, nn.Linear):
                nn.init.xavier_uniform_(one.weight)
                
    def extract_img(self, batch):
        img = batch['img']  # # batch['img'] shape: bs*#creatives, 3, 192, 192
        bs, creative_num = img.shape[:2]
        mask_loss, img_emb = self.swin_model(img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]))  #
        # img_emb shape: bs*#creatives, 1024
        return img_emb
    
    def extract_txt(self, batch):
        txt = batch['txt']
        txt_msk = batch['txt_msk']
        bert_feat = self.bert_model(txt, txt_msk)  # segment_ids
        bert_feat = bert_feat.last_hidden_state   # bs, #tokens, 768
        cls_feat = bert_feat[:, 0, :]  # bs, 768
        return cls_feat
     
    def get_cls_feat(self, cls_feat, creative_num, linear_fc):
        cls_feat = linear_fc(cls_feat)
        if creative_num > 0:
            cls_feat = cls_feat.unsqueeze(1).repeat(1, creative_num, 1).reshape(-1, cls_feat.shape[-1])  # (bs*#creatives), 256
        return cls_feat
        
    def forward(self, batch):
        """
        batch is dict:
        {
        'img': img, 'txt': txt, 'cap': cap, 'label_ctr': target_ctr
        }
        """
        img = batch['img']  # # batch['img'] shape: bs, creatives, 1024
        bs, creative_num = img.shape[:2]
        if self.need_img_emb:
            img_emb = img.reshape(-1, img.shape[-1])
            # img_emb shape: bs*#creatives, 1024
            img_emb = self.img_linear_fc(img_emb)
        else:
            img_emb = None
        if self.need_txt_emb:
            cls_feat = self.get_cls_feat(batch['txt'], creative_num, linear_fc=self.txt_linear_fc)   # (bs*#creatives), 768
            neg_cls_feat = self.get_cls_feat(batch['neg_txt'], creative_num, linear_fc=self.txt_linear_fc)
        else:
            cls_feat = None
            neg_cls_feat = None
            
        if self.need_cap_emb:
            cap_feat = batch['cap'].reshape(-1, batch['cap'].shape[-1])   # (bs*#creatives), 768
            cap_feat = self.get_cls_feat(cap_feat, 0, linear_fc=self.cap_linear_fc)   # (bs*#creatives), 768
        else:
            cap_feat = None
            
        if self.without_transformer:
            img_emb = img_emb.reshape(bs, -1, img_emb.shape[-1])
            cls_feat = cls_feat.reshape(bs, -1, cls_feat.shape[-1])
            cap_feat = cap_feat.reshape(bs, -1, cap_feat.shape[-1])
            final_emb_list = []
            if self.need_img_emb:
                final_emb_list.append(img_emb)
            if self.need_txt_emb:
                final_emb_list.append(cls_feat)
            if self.need_cap_emb:
                final_emb_list.append(cap_feat)
            final_emb = torch.cat(final_emb_list, dim=-1)
        else:
            has_part_num = int(self.need_img_emb) + int(self.need_txt_emb) + int(self.need_cap_emb)
            if has_part_num > 1:
                if self.need_img_emb and self.need_txt_emb and self.need_cap_emb:
                    multi_emb = torch.cat([img_emb.unsqueeze(1), cls_feat.unsqueeze(1), cap_feat.unsqueeze(1)], dim=1)  # (bs*#creatives), 3, 256
                elif self.need_img_emb and self.need_txt_emb:
                    multi_emb = torch.cat([img_emb.unsqueeze(1), cls_feat.unsqueeze(1)], dim=1)  # (bs*#creatives), 3, 256
                elif self.need_img_emb and self.need_cap_emb:
                    multi_emb = torch.cat([img_emb.unsqueeze(1), cap_feat.unsqueeze(1)], dim=1)  # (bs*#creatives), 3, 256
                elif self.need_txt_emb and self.need_cap_emb:
                    multi_emb = torch.cat([cls_feat.unsqueeze(1), cap_feat.unsqueeze(1)], dim=1)  # (bs*#creatives), 3, 256

                multi_mask = torch.ones((bs*creative_num, has_part_num)).long()
                multi_attention_mask = multi_mask.unsqueeze(1).unsqueeze(2)
                multi_attention_mask = multi_attention_mask.to(dtype=img_emb.dtype)  # fp16 compatibility
                multi_attention_mask = (1.0 - multi_attention_mask) * -10000.0
                multi_attention_mask = multi_attention_mask.to(img_emb.device)
                multi_encoded_out = self.multi_transformer.encoder(multi_emb, multi_attention_mask)[-1]

                if self.need_img_emb and self.need_txt_emb and self.need_cap_emb:
                    img_out = multi_encoded_out[:, 0, :]   # (bs*#creatives), 256
                    txt_out = multi_encoded_out[:, 1, :]   # (bs*#creatives), 256
                    cap_out = multi_encoded_out[:, 2, :]   # (bs*#creatives), 256
                    img_out = img_out.reshape(bs, -1, img_out.shape[-1])   # bs, #creatives, 256
                    txt_out = txt_out.reshape(bs, -1, txt_out.shape[-1])   # bs, #creatives, 256
                    cap_out = cap_out.reshape(bs, -1, cap_out.shape[-1])   # bs, #creatives, 256
                    final_emb = torch.cat([img_out, txt_out, cap_out], dim=-1)   # bs, #creatives, 256+256+256
                elif self.need_img_emb and self.need_txt_emb:
                    img_out = multi_encoded_out[:, 0, :]   # (bs*#creatives), 256
                    txt_out = multi_encoded_out[:, 1, :]   # (bs*#creatives), 256
                    img_out = img_out.reshape(bs, -1, img_out.shape[-1])   # bs, #creatives, 256
                    txt_out = txt_out.reshape(bs, -1, txt_out.shape[-1])   # bs, #creatives, 256
                    final_emb = torch.cat([img_out, txt_out], dim=-1)    # bs, #creatives, 256+256
                elif self.need_img_emb and self.need_cap_emb:
                    img_out = multi_encoded_out[:, 0, :]   # (bs*#creatives), 256
                    cap_out = multi_encoded_out[:, 1, :]   # (bs*#creatives), 256
                    img_out = img_out.reshape(bs, -1, img_out.shape[-1])   # bs, #creatives, 256
                    cap_out = cap_out.reshape(bs, -1, cap_out.shape[-1])   # bs, #creatives, 256
                    final_emb = torch.cat([img_out, cap_out], dim=-1)   # bs, #creatives, 256+256
                elif self.need_txt_emb and self.need_cap_emb:
                    txt_out = multi_encoded_out[:, 0, :]   # (bs*#creatives), 256
                    cap_out = multi_encoded_out[:, 1, :]   # (bs*#creatives), 256
                    txt_out = txt_out.reshape(bs, -1, txt_out.shape[-1])   # bs, #creatives, 256
                    cap_out = cap_out.reshape(bs, -1, cap_out.shape[-1])   # bs, #creatives, 256
                    final_emb = torch.cat([txt_out, cap_out], dim=-1)   # bs, #creatives, 256+256
            elif has_part_num == 1:
                if self.need_img_emb:
                    final_emb = img_emb.reshape(bs, -1, img_emb.shape[-1])  # bs, #creatives, 256
                elif self.need_txt_emb:
                    final_emb = cls_feat.reshape(bs, -1, cls_feat.shape[-1])  # bs, #creatives, 256
                elif self.need_cap_emb:
                    final_emb = cap_feat.reshape(bs, -1, cap_feat.shape[-1])  # bs, #creatives, 256
            else:
                raise NotImplementedError
        #print("final_emb", final_emb.shape)
#         pred_ctr = self.ctr_fc(final_emb)
        final_emb = final_emb.reshape(-1, final_emb.shape[-1])  # (bs,#creatives), (256+256+256)
        
        final_emb = self.final_fc(final_emb)   # (bs,#creatives), 10
        # pred_ctr = self.ctr_fc(final_emb)      # (bs,#creatives), 1
        
        final_emb = final_emb.reshape(bs, creative_num, final_emb.shape[-1])
        # pred_ctr = pred_ctr.reshape(bs, creative_num, pred_ctr.shape[-1]).squeeze(-1)
        # print("final_emb", final_emb.shape)
        pred_ctr = torch.mean(final_emb, -1)
        # print("pred_ctr", pred_ctr.shape)
        # print("--->>>", img_emb.shape, cls_feat.shape, neg_cls_feat.shape)
        
        return pred_ctr, img_emb, cls_feat, neg_cls_feat, final_emb
