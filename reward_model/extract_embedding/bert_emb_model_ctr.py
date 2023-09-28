import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertModel
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertPreTrainingHeads


class NLPBertModel(nn.Module):
    def __init__(self, hash_path, bert_model, device, model_flag="bert",
                 dense_field_size=32, dense_field_num=101, list_field_size=6, emb_size=6,
                 feat_hidden_size=768, hidden_size=128, dropout_prob=0.1, max_his_len=55):
        super(NLPBertModel, self).__init__()

        self.user_position_embedding = nn.Embedding(max_his_len, hidden_size)
        nn.init.xavier_uniform_(self.user_position_embedding.weight)
        self.start_bro_pos = 0
        self.start_add_pos = 30
        self.start_buy_pos = 40

        # self.title_des_type_embedding = nn.Embedding(2, feat_hidden_size)
        # nn.init.xavier_uniform_(self.title_des_type_embedding.weight)

        self.model_flag = model_flag
        self.bert = bert_model
        # print('config', self.bert.config)
        # if model_flag == "bert":
        #     self.cls = BertPreTrainingHeads(self.bert.config)  # , self.bert.embeddings.word_embeddings.weight
        #     self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        # elif model_flag == "distilbert":
        #     print("using distil bert =======")
        #     self.activation = GELUActivation()
        #     self.vocab_transform = nn.Linear(self.bert.config.dim, self.bert.config.dim)
        #     self.vocab_layer_norm = nn.LayerNorm(self.bert.config.dim, eps=1e-12)
        #     self.vocab_projector = nn.Linear(self.bert.config.dim, self.bert.config.vocab_size)
        #     #print("ssss", self.bert.embeddings.word_embeddings.weight.shape)
        #     self.vocab_projector.weight = self.bert.embeddings.word_embeddings.weight
        # else:
        #     raise NotImplementedError
        self.device = device
        self.dense_field_size = dense_field_size
        self.feat_hidden_size = feat_hidden_size
        #self.cur_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.title_fc = nn.Sequential(nn.Linear(feat_hidden_size, hidden_size))
        self.des_fc = nn.Sequential(nn.Linear(feat_hidden_size, hidden_size))

        self.title_des_fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_bro_fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_add_fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_buy_fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_bro_fc_item = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_add_fc_item = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_buy_fc_item = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_qry_fc_item = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.his_fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

        nn.init.xavier_uniform_(self.title_des_fc[0].weight)
        nn.init.xavier_uniform_(self.his_bro_fc[0].weight)
        nn.init.xavier_uniform_(self.his_add_fc[0].weight)
        nn.init.xavier_uniform_(self.his_buy_fc[0].weight)
        nn.init.xavier_uniform_(self.his_bro_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_add_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_buy_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_qry_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_fc[0].weight)

    def _load_hash(self, hash_path):
        with open(hash_path) as f:
            d = json.load(f)
        return d

    def print_model_param(self):
        print("cur_fc", torch.max(self.cur_fc[0].weight), torch.min(self.cur_fc[0].weight))
        print("his_bro_fc", torch.max(self.his_bro_fc[0].weight), torch.min(self.his_bro_fc[0].weight))
        print("his_add_fc", torch.max(self.his_add_fc[0].weight), torch.min(self.his_add_fc[0].weight))
        print("his_buy_fc", torch.max(self.his_buy_fc[0].weight), torch.min(self.his_buy_fc[0].weight))
        print("his_bro_fc_item", torch.max(self.his_bro_fc_item[0].weight), torch.min(self.his_bro_fc_item[0].weight))
        print("his_add_fc_item", torch.max(self.his_add_fc_item[0].weight), torch.min(self.his_add_fc_item[0].weight))
        print("his_buy_fc_item", torch.max(self.his_buy_fc_item[0].weight), torch.min(self.his_buy_fc_item[0].weight))
        print("his_fc", torch.max(self.his_fc[0].weight), torch.min(self.his_fc[0].weight))
        print("item_fc", torch.max(self.item_fc[0].weight), torch.min(self.item_fc[0].weight))
        print("user_fc", torch.max(self.user_fc[0].weight), torch.min(self.user_fc[0].weight))
        print("final_fc", torch.max(self.final_fc[0].weight), torch.min(self.final_fc[0].weight))

    def get_atten_score_for_item_old(self, cur_id_tensor, cur_mask_tensor):
        """
        get attention scores: bs, #tokens
        """
        token_prob, cur_att_feat = self.get_bert_token_score(cur_id_tensor, cur_mask_tensor, self.cur_fc)  # bs, #tokens
        itm_feat = self.item_fc(cur_att_feat)  # bs, 128
        return token_prob, itm_feat

    def get_atten_score_for_item(self, cur_id_tensor, cur_mask_tensor):
        token_prob, cur_att_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)  # bs, #tokens, 768
        #print("token_prob", token_prob)
        itm_feat = self.item_fc(cur_att_feat)  # bs, 128
        return token_prob, itm_feat

    def get_bert_feat_for_item(self, cur_id_tensor, cur_mask_tensor, is_title):
        token_prob, cur_att_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)  # bs, #tokens, 768
        if is_title:
            cur_att_feat = self.title_fc(cur_att_feat)
        else:
            cur_att_feat = self.des_fc(cur_att_feat)
        return token_prob, cur_att_feat

    def get_title_des_feat(self, title_feat, des_feat):
        cur_item_feat = torch.stack([title_feat, des_feat], dim=1)
        cur_mask_tmp_tensor = torch.ones((title_feat.shape[0], 2), dtype=torch.int32, device=self.device)
        cur_att_feat = self.attention_by_weight(cur_item_feat, cur_mask_tmp_tensor, self.title_des_fc)
        return cur_att_feat   # bs, 128

    def norm_matrix(self, emb, dim=1):
        """
        特征归一化
        :param emb: 输入的特征，bs * dim
        :param dim: 按行或列进行归一化
        :return:
        """
        emb_norm = emb.norm(p=2, dim=dim, keepdim=True)
        return emb.div(emb_norm)

    def get_atten_score_for_user(self, his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor, his_buy_msk_tensor, \
                his_bro_fea_tensor, his_add_fea_tensor, his_buy_fea_tensor, query_id_tensor, query_mask_tensor):
        bs = his_msk_tensor.shape[0]
        his_bro_att_feat = his_bro_fea_tensor
        his_add_att_feat = his_add_fea_tensor
        his_buy_att_feat = his_buy_fea_tensor

        his_bro_att_item_feat = self.attention_by_weight(his_bro_att_feat, his_bro_msk_tensor,
                                                         self.his_bro_fc_item)  # bs, 768
        his_add_att_item_feat = self.attention_by_weight(his_add_att_feat, his_add_msk_tensor,
                                                         self.his_add_fc_item)  # bs, 768
        his_buy_att_item_feat = self.attention_by_weight(his_buy_att_feat, his_buy_msk_tensor,
                                                         self.his_buy_fc_item)  # bs, 768
        bert_feat = self.bert(query_id_tensor, query_mask_tensor)  # bs, #tokens, 768
        bert_feat = bert_feat.last_hidden_state
        cls_feat = bert_feat[:, 0, :].unsqueeze(-1)  # bs, 768, 1
        bert_feat_norm = self.norm_matrix(bert_feat, dim=-1)
        cls_feat_norm = self.norm_matrix(cls_feat, dim=1)
        atten_score = torch.bmm(bert_feat_norm, cls_feat_norm).squeeze(-1)  # bs, #tokens
        # print("score", atten_score)
        extended_attention_mask = (1.0 - query_mask_tensor) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        query_probs = nn.Softmax(dim=-1)(atten_score)

        user_stack_feat = torch.stack([his_bro_att_item_feat, his_add_att_item_feat, his_buy_att_item_feat, bert_feat[:, 0, :]],
                                      dim=1)  # bs, 4, 768
        his_prob, usr_att_feat = self.attention_score_by_weight(user_stack_feat, his_msk_tensor, self.his_fc)
        usr_feat = self.user_fc(usr_att_feat)  # bs, 128

        return query_probs, his_prob, usr_feat

    def forward_lm(self, cur_id_tensor, cur_mask_tensor, masked_pos_tensor):
        # segment_ids = torch.zeros_like(cur_mask_tensor).long()
        bert_feat = self.bert(cur_id_tensor, cur_mask_tensor)
        bert_feat = bert_feat.last_hidden_state
        mask_feature = torch.gather(bert_feat, 1,
                                    masked_pos_tensor.unsqueeze(2).expand(-1, -1, bert_feat.shape[-1]))
        #print("mask_feature", bert_feat.shape, mask_feature.shape, masked_pos_tensor.shape)
        if self.model_flag == "bert":
            prediction_scores, _ = self.cls(mask_feature, bert_feat)
        else:
            prediction_logits = self.vocab_transform(mask_feature)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_scores = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            #print('score', prediction_scores.shape)
        #print("score", prediction_scores.shape)
        return prediction_scores

    def get_field_emb_fea(self, field_tensor_tuple):
        other_dense_tensor = field_tensor_tuple[-1]
        field_tensor_tuple = field_tensor_tuple[:-1]
        single_tensor = field_tensor_tuple[:len(field_tensor_tuple)-2*self.list_field_size]  # each shape: bs
        list_tensor = field_tensor_tuple[len(field_tensor_tuple)-2*self.list_field_size:
                                               len(field_tensor_tuple)-self.list_field_size]   # each shape: bs, #fields
        list_mask_tensor = field_tensor_tuple[len(field_tensor_tuple)-self.list_field_size:]   # each shape: bs, #fields
        # print("single_input", single_tensor)
        # for idx, v in enumerate(single_tensor):
        #     print(idx, v, self.field_name_list[idx], self.hash_dict[self.field_name_list[idx]])
        #     hh = self.field_embeddings[idx](v)
        #     print(hh)
        single_emb_tensor = [self.field_embeddings[idx](v) for idx, v in enumerate(single_tensor)]  # bs, 6
        list_emb_tensor = []
        for idx, v in enumerate(list_tensor):
            one_emb = self.field_embeddings[len(single_tensor)+idx](v)  # bs, #fields, 6

            # mean pool
            one_emb = one_emb * list_mask_tensor[idx].unsqueeze(-1)  # compute weight
            one_emb = torch.sum(one_emb, dim=1) / torch.sum(list_mask_tensor[idx], dim=1).unsqueeze(-1)  # bs, 6

            # atten pool
            # one_emb = self.attention_by_weight(one_emb, list_mask_tensor[idx], self.field_weight[idx])
            # print("---", one_emb.shape)
            list_emb_tensor.append(one_emb)

        other_dense_tensor = other_dense_tensor.long()
        # print("other_dense ", other_dense_tensor.shape)
        dense_emb_tensor = []
        for idx in range(other_dense_tensor.shape[1]):
            one_emb = self.dense_field_embeddings[idx](other_dense_tensor[:, idx])  # bs, 6
            # print('----', idx, one_emb.shape)
            dense_emb_tensor.append(one_emb)
        # other_dense_emb = self.dense_field_embeddings[other_dense_tensor.long()]  # bs, 32, 6
        # print("other_dense_emb", other_dense_emb.shape)
        base_fea = torch.hstack(single_emb_tensor + list_emb_tensor + dense_emb_tensor)  # bs, 12*6
        # base_fea = torch.cat((base_fea, other_dense_tensor), dim=-1)
        # print("base_fea", base_fea.shape)
        return base_fea

    def mean_title_des_fea(self, bs, his_fea_tensor, his_des_tensor):
        his_tit_att_feat = his_fea_tensor.reshape(-1, his_fea_tensor.shape[-1])  # (bs*30), 768
        his_des_att_feat = his_des_tensor.reshape(-1, his_des_tensor.shape[-1])  # (bs*30), 768
        tit_des_feat = torch.stack([his_tit_att_feat, his_des_att_feat], dim=1)  # (bs*30), 2, 768
        his_feat = torch.mean(tit_des_feat, dim=1)  # (bs*30), 768
        his_feat = his_feat.reshape(bs, -1, self.feat_hidden_size)  # bs, 30, 768
        return his_feat

    def atten_title_des_fea(self, bs, his_fea_tensor, his_des_tensor, his_des_msk_tensor):
        # his_fea_tensor/his_des_tensor shape: bs, 30, 768
        # his_des_msk_tensor shape: bs, 30
        his_tit_att_feat = his_fea_tensor.reshape(-1, his_fea_tensor.shape[-1])  # (bs*30), 128
        his_des_att_feat = his_des_tensor.reshape(-1, his_des_tensor.shape[-1])  # (bs*30), 128
        his_des_msk_tensor = his_des_msk_tensor.reshape(-1)  # bs*30
        his_tit_mask_tensor = torch.ones(his_tit_att_feat.shape[0], dtype=torch.long,
                                             device=self.device)
        his_mask_tmp_tensor = torch.stack([his_tit_mask_tensor, his_des_msk_tensor], dim=1)  # (bs*30), 2
        # print("bor mask", his_bro_mask_tmp_tensor.shape, torch.stack([his_bro_tit_att_feat, his_bro_des_att_feat], dim=1).shape)
        #his_item_feat = self.add_text_type_embedding(his_tit_att_feat.shape[0], his_tit_att_feat, his_des_att_feat)
        his_item_feat = torch.stack([his_tit_att_feat, his_des_att_feat], dim=1)  # (bs*30), 2, 128
        his_att_feat = self.attention_by_weight(his_item_feat, his_mask_tmp_tensor, self.title_des_fc)  # (bs*30), 128
        his_att_feat = his_att_feat.reshape(bs, -1, his_att_feat.shape[-1])  # bs, 30, 128
        return his_att_feat

    def add_pos_emb(self, bs, his_emb, start_pos=0):
        # his_emb: bs, 30, 768
        his_bro_long = torch.arange(start_pos, start_pos+his_emb.shape[1]).repeat(bs, 1).to(
            self.device)  # bs, 30 --> [[0,1,2,...],[0,1,2,...],...]
        his_bro_pos = self.user_position_embedding(his_bro_long)  # bs, 30, 768
        return his_bro_pos + his_emb  # bs, 30, 768

    def get_query_prob(self, bert_feat, query_mask_tensor):
        cls_feat = bert_feat[:, 0, :].unsqueeze(-1)  # bs, 768, 1
        bert_feat_norm = self.norm_matrix(bert_feat, dim=-1)
        cls_feat_norm = self.norm_matrix(cls_feat, dim=1)
        atten_score = torch.bmm(bert_feat_norm, cls_feat_norm).squeeze(-1)  # bs, #tokens
        # print("score", atten_score)
        extended_attention_mask = (1.0 - query_mask_tensor) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        query_probs = nn.Softmax(dim=-1)(atten_score)
        return query_probs

    def add_text_type_embedding(self, bs, title_emb, des_emb):
        # title_emb/ des_emb: bs, 768
        type_long = torch.arange(0, 2).repeat(bs, 1).to(
            self.device)  # bs, 2 --> [[0,1],[0,1],...]
        type_emb = self.title_des_type_embedding(type_long)  # bs, 2, 768
        item_emb = torch.stack([title_emb, des_emb], dim=1)  # bs, 2, 768
        return item_emb + type_emb

    def forward(self, input_tensor, return_mid_res=False):
        """
        print("cur_id", cur_id_tensor.shape): bs, #tokens
        print("cur_mask_tensor", cur_mask_tensor.shape): bs, #tokens
        print("his_msk_tensor", his_msk_tensor.shape): bs, 4
        print("his_bro_msk_tensor", his_bro_msk_tensor.shape): bs, 30
        print("his_add_msk_tensor", his_add_msk_tensor.shape): bs, 10
        print("his_buy_msk_tensor", his_buy_msk_tensor.shape): bs, 10
        print("his_bro_id_tensor", his_bro_id_tensor.shape): bs*30, #tokens
        print("his_add_id_tensor", his_add_id_tensor.shape): bs*10, #tokens
        print("his_buy_id_tensor", his_buy_id_tensor.shape): bs*10, #tokens
        print("his_bro_m_tensor", his_bro_m_tensor.shape): bs*30, #tokens
        print("his_add_m_tensor", his_add_m_tensor.shape): bs*10, #tokens
        print("his_buy_m_tensor", his_buy_m_tensor.shape): bs*10, #tokens
        print("query_id_tensor", query_id_tensor.shape): bs, #tokens
        print("query_mask_tensor", query_mask_tensor.shape): bs, #tokens
        """
        cur_fea_tensor, cur_des_fea_tensor, cur_des_msk_tensor, his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor, his_buy_msk_tensor, \
        his_bro_des_msk_tensor, his_add_des_msk_tensor, his_buy_des_msk_tensor, \
        bro_fea_tensor, bro_des_fea_tensor, add_fea_tensor, add_des_fea_tensor, buy_fea_tensor, buy_des_fea_tensor = input_tensor

        bs = cur_fea_tensor.shape[0]
        #cur_att_feat = self.get_bert_feat(cur_id_tensor, cur_mask_tensor, self.cur_fc)  # bs, 768
        cur_tit_mask_tensor = torch.ones(bs, dtype=torch.long, device=self.device)
        cur_mask_tmp_tensor = torch.stack([cur_tit_mask_tensor, cur_des_msk_tensor], dim=1)  # bs, 2
        #cur_item_feat = self.add_text_type_embedding(bs, cur_fea_tensor, cur_des_fea_tensor)  # bs, 2, 768
        if cur_fea_tensor.shape[-1] == self.feat_hidden_size:
            cur_fea_tensor = self.title_fc(cur_fea_tensor)  # bs, 768 --> bs, 128
            cur_des_fea_tensor = self.des_fc(cur_des_fea_tensor)  # bs, 768 --> bs, 128
        cur_item_feat = torch.stack([cur_fea_tensor, cur_des_fea_tensor], dim=1)
        cur_att_feat = self.attention_by_weight(cur_item_feat, cur_mask_tmp_tensor, self.title_des_fc)
        # cur_att_feat = torch.mean(torch.stack([cur_fea_tensor, cur_des_fea_tensor], dim=1), dim=1)

        if bro_fea_tensor.shape[-1] == self.feat_hidden_size:
            bro_fea_tensor = self.title_fc(bro_fea_tensor)
            add_fea_tensor = self.title_fc(add_fea_tensor)
            buy_fea_tensor = self.title_fc(buy_fea_tensor)
            bro_des_fea_tensor = self.des_fc(bro_des_fea_tensor)
            add_des_fea_tensor = self.des_fc(add_des_fea_tensor)
            buy_des_fea_tensor = self.des_fc(buy_des_fea_tensor)

        his_bro_att_feat = self.atten_title_des_fea(bs, bro_fea_tensor, bro_des_fea_tensor, his_bro_des_msk_tensor)  # bs, 30, 768
        his_add_att_feat = self.atten_title_des_fea(bs, add_fea_tensor, add_des_fea_tensor, his_add_des_msk_tensor)  # bs, 10, 768
        his_buy_att_feat = self.atten_title_des_fea(bs, buy_fea_tensor, buy_des_fea_tensor, his_buy_des_msk_tensor)  # bs, 10, 768
        # his_bro_att_feat = bro_fea_tensor
        # his_add_att_feat = add_fea_tensor
        # his_buy_att_feat = buy_fea_tensor
        # his_bro_att_feat = self.mean_title_des_fea(bs, bro_fea_tensor, bro_des_fea_tensor)
        # his_add_att_feat = self.mean_title_des_fea(bs, add_fea_tensor, add_des_fea_tensor)
        # his_buy_att_feat = self.mean_title_des_fea(bs, buy_fea_tensor, buy_des_fea_tensor)

        his_bro_att_feat = self.add_pos_emb(bs, his_bro_att_feat, self.start_bro_pos)
        his_add_att_feat = self.add_pos_emb(bs, his_add_att_feat, self.start_add_pos)
        his_buy_att_feat = self.add_pos_emb(bs, his_buy_att_feat, self.start_buy_pos)

        #print("his", his_bro_att_feat.shape, his_add_att_feat.shape, his_buy_att_feat.shape)
        # print("---", cur_att_feat.shape, his_bro_att_feat.shape, his_add_att_feat.shape, his_buy_att_feat.shape)
        his_bro_att_item_feat = self.attention_by_weight(his_bro_att_feat, his_bro_msk_tensor, self.his_bro_fc_item)  # bs, 768
        his_add_att_item_feat = self.attention_by_weight(his_add_att_feat, his_add_msk_tensor, self.his_add_fc_item)  # bs, 768
        his_buy_att_item_feat = self.attention_by_weight(his_buy_att_feat, his_buy_msk_tensor, self.his_buy_fc_item)  # bs, 768

        # his_bro_att_item_feat = torch.mean(his_bro_att_feat, dim=1)  # bs, 768
        # his_add_att_item_feat = torch.mean(his_add_att_feat, dim=1)  # bs, 768
        # his_buy_att_item_feat = torch.mean(his_buy_att_feat, dim=1)  # bs, 768

        # print("---", query_id_tensor.shape, query_mask_tensor.shape, query_feat.shape)
        user_stack_feat = torch.stack([his_bro_att_item_feat, his_add_att_item_feat, his_buy_att_item_feat], dim=1)  # bs, 4, 768
        # user_mask = torch.ones((bs, 3)).long().to(self.device)
        #print("user attention start")
        #usr_att_feat = self.attention_by_weight(user_stack_feat, his_msk_tensor, self.his_fc)
        his_prob, usr_att_feat = self.attention_score_by_weight(user_stack_feat, his_msk_tensor, self.his_fc)
        # his_prob = None
        # usr_att_feat = torch.mean(user_stack_feat, dim=1)  # bs, 768

        # itm_feat = self.item_fc(cur_att_feat)  # bs, 128
        # usr_feat = self.user_fc(usr_att_feat)  # bs, 128
        itm_feat = cur_att_feat
        usr_feat = usr_att_feat

        # pred_prob = nn.Sigmoid()(torch.sum(itm_feat * usr_feat, dim=-1))  # bs

        # base_feat = self.get_field_emb_fea(field_tensor_tuple)
        # final_feat = torch.cat([itm_feat, usr_feat, base_feat], dim=-1)  # bs, (2*128 + #fields*6)
        # pred_prob = nn.Sigmoid()(self.final_fc(final_feat)).squeeze(-1)  # bs

        #pred_prob = nn.CosineSimilarity(dim=-1)(itm_feat, usr_feat)
        #pred_prob = nn.Sigmoid()(pred_prob)
        pred_prob = nn.Sigmoid()(torch.sum(itm_feat * usr_feat, dim=-1))  # bs
        if return_mid_res:
            return pred_prob, his_prob, usr_feat, itm_feat
        return pred_prob

    def forward_onlyItem(self, input_tensor, return_mid_res=False):
        cur_att_feat, his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor, his_buy_msk_tensor, \
            his_bro_att_feat, his_add_att_feat, his_buy_att_feat = input_tensor

        bs = cur_att_feat.shape[0]

        his_bro_att_feat = self.add_pos_emb(bs, his_bro_att_feat, self.start_bro_pos)
        his_add_att_feat = self.add_pos_emb(bs, his_add_att_feat, self.start_add_pos)
        his_buy_att_feat = self.add_pos_emb(bs, his_buy_att_feat, self.start_buy_pos)

        his_bro_att_item_feat = self.attention_by_weight(his_bro_att_feat, his_bro_msk_tensor, self.his_bro_fc_item)  # bs, 768
        his_add_att_item_feat = self.attention_by_weight(his_add_att_feat, his_add_msk_tensor, self.his_add_fc_item)  # bs, 768
        his_buy_att_item_feat = self.attention_by_weight(his_buy_att_feat, his_buy_msk_tensor, self.his_buy_fc_item)  # bs, 768

        user_stack_feat = torch.stack([his_bro_att_item_feat, his_add_att_item_feat, his_buy_att_item_feat], dim=1)  # bs, 4, 768

        his_prob, usr_att_feat = self.attention_score_by_weight(user_stack_feat, his_msk_tensor, self.his_fc)

        itm_feat = cur_att_feat
        usr_feat = usr_att_feat

        pred_prob = nn.Sigmoid()(torch.sum(itm_feat * usr_feat, dim=-1))  # bs
        if return_mid_res:
            return pred_prob, his_prob, usr_feat, itm_feat
        return pred_prob

    def get_bert_feat(self, ids, mask, weight):
        segment_ids = torch.zeros_like(mask).long()
        bert_feat = self.bert(ids, mask, segment_ids)
        bert_feat = bert_feat.last_hidden_state
        atten_feat = self.attention_by_weight(bert_feat, mask, weight)
        return atten_feat

    def get_bert_cls_feat(self, ids, mask):
        # segment_ids = torch.zeros_like(mask).long()
        bert_feat = self.bert(ids, mask)  # segment_ids
        bert_feat = bert_feat.last_hidden_state
        return bert_feat[:, 0, :]

    def get_bert_cls_atten_feat(self, ids, mask):
        bert_feat = self.bert(ids, mask)  # segment_ids
        bert_feat = bert_feat.last_hidden_state  # bs, #tokens, 768
        cls_feat = bert_feat[:, 0, :].unsqueeze(-1)  # bs, 768, 1
        bert_feat_norm = self.norm_matrix(bert_feat, dim=-1)
        cls_feat_norm = self.norm_matrix(cls_feat, dim=1)

        atten_score = torch.bmm(bert_feat_norm, cls_feat_norm).squeeze(-1)  # bs, #tokens
        # print("score", atten_score)
        extended_attention_mask = (1.0 - mask) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        atten_probs = nn.Softmax(dim=-1)(atten_score)
        # print("mask", mask)
        # print("probs", atten_probs)
        atten_feat = atten_probs.unsqueeze(-1) * bert_feat
        atten_feat = torch.sum(atten_feat, dim=1)
        return atten_probs, atten_feat

    def get_bert_token_score(self, ids, mask, weight):
        segment_ids = torch.zeros_like(mask).long()
        bert_feat = self.bert(ids, mask, segment_ids)
        bert_feat = bert_feat.last_hidden_state
        atten_prob, atten_feat = self.attention_score_by_weight(bert_feat, mask, weight)
        return atten_prob, atten_feat

    def attention_score_by_weight(self, feat, mask, weight):
        atten_score = weight(feat).squeeze(-1)  # bs, #tokens
        extended_attention_mask = (1.0 - mask) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        atten_probs = nn.Softmax(dim=-1)(atten_score)  # bs, #tokens
        atten_feat = atten_probs.unsqueeze(-1) * feat
        atten_feat = torch.sum(atten_feat, dim=1)
        return atten_probs, atten_feat

    def attention_by_weight(self, feat, mask, weight):
        """
        feat: bs, n, 768
        mask: bs, n
        weight: 768
        """
        atten_score = weight(feat).squeeze(-1)  # bs, #tokens
        extended_attention_mask = (1.0 - mask) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        atten_probs = nn.Softmax(dim=-1)(atten_score)
        #print("atten_probs", atten_probs)
        atten_feat = atten_probs.unsqueeze(-1) * feat
        atten_feat = torch.sum(atten_feat, dim=1)
        return atten_feat
