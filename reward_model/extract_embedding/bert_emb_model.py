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
                 feat_hidden_size=768, hidden_size=128, dropout_prob=0.1):
        super(NLPBertModel, self).__init__()
        # shopid_hash = self.hash_dict['shopid']
        # # userid not use
        # cat1V2_hash = self.hash_dict['cat1V2']
        # cat2V2_hash = self.hash_dict['cat2V2']
        # cat3V2_hash = self.hash_dict['cat3V2']
        # IntentionL0_hash = self.hash_dict['IntentionL0']
        # IntentionL1_hash = self.hash_dict['IntentionL1']
        # user_age_hash = self.hash_dict['user_age']
        # user_gender_hash = self.hash_dict['user_gender']
        # rt_shopids_hash = self.hash_dict['rt_shopids']
        # rt_cat1_hash = self.hash_dict['rt_cat1']
        # rt_cat2_hash = self.hash_dict['rt_cat2']
        # rt_cat3_hash = self.hash_dict['rt_cat3']
        self.list_field_size = list_field_size
        self.field_name_list = ['shopid', 'cat1V2', 'cat2V2', 'cat3V2', 'IntentionL0', 'IntentionL1', 'user_age', 'user_gender', 'PhoneBrand',
                                'rt_shopids', 'rt_cat1', 'rt_cat2', 'rt_cat3', 'rt_inten0', 'rt_inten1']
        self.hash_dict = self._load_hash(hash_path)

        # field embedding
        self.field_embeddings = nn.ModuleList([
            nn.Embedding(len(self.hash_dict[field_name])+1, emb_size) for field_name in self.field_name_list
        ])
        # self.field_weight = nn.ModuleList([
        #     nn.Sequential(nn.Linear(emb_size, 1), nn.Tanh()) for _ in range(list_field_size)
        # ])
        self.dense_field_embeddings = nn.ModuleList([
            nn.Embedding(dense_field_num, emb_size) for _ in range(dense_field_size)
        ])
        for one_emb in self.field_embeddings:  #, self.field_weight):
            nn.init.xavier_uniform_(one_emb.weight)
            #nn.init.xavier_uniform_(one_weight[0].weight)

        self.model_flag = model_flag
        self.bert = bert_model
        print('config', self.bert.config)
        if model_flag == "bert":
            self.cls = BertPreTrainingHeads(self.bert.config)  # , self.bert.embeddings.word_embeddings.weight
            self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        elif model_flag == "distilbert":
            print("using distil bert =======")
            self.activation = GELUActivation()
            self.vocab_transform = nn.Linear(self.bert.config.dim, self.bert.config.dim)
            self.vocab_layer_norm = nn.LayerNorm(self.bert.config.dim, eps=1e-12)
            self.vocab_projector = nn.Linear(self.bert.config.dim, self.bert.config.vocab_size)
            #print("ssss", self.bert.embeddings.word_embeddings.weight.shape)
            self.vocab_projector.weight = self.bert.embeddings.word_embeddings.weight
        else:
            raise NotImplementedError
        self.device = device
        self.dense_field_size = dense_field_size
        self.feat_hidden_size = feat_hidden_size
        self.cur_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_bro_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_add_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_buy_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_bro_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_add_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_buy_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_qry_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())

        self.item_fc = nn.Sequential(nn.Linear(feat_hidden_size, hidden_size))
        self.user_fc = nn.Sequential(nn.Linear(feat_hidden_size, hidden_size))

        self.final_fc = nn.Sequential(
            nn.Linear(2 * hidden_size + emb_size * len(self.field_name_list) + emb_size * self.dense_field_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, 1)
            )

        nn.init.xavier_uniform_(self.cur_fc[0].weight)
        nn.init.xavier_uniform_(self.his_bro_fc[0].weight)
        nn.init.xavier_uniform_(self.his_add_fc[0].weight)
        nn.init.xavier_uniform_(self.his_buy_fc[0].weight)
        nn.init.xavier_uniform_(self.his_bro_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_add_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_buy_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_qry_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_fc[0].weight)
        nn.init.xavier_uniform_(self.item_fc[0].weight)
        nn.init.xavier_uniform_(self.user_fc[0].weight)
        nn.init.xavier_uniform_(self.final_fc[0].weight)
        nn.init.xavier_uniform_(self.final_fc[-1].weight)

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

    def get_bert_feat_for_item(self, cur_id_tensor, cur_mask_tensor, is_title=True):
        token_prob, cur_att_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)  # bs, #tokens, 768
        return token_prob, cur_att_feat

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

    def forward(self, input_tensor):
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
        cur_id_tensor, cur_mask_tensor, his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor, his_buy_msk_tensor, \
        his_bro_id_tensor, his_add_id_tensor, his_buy_id_tensor, his_bro_m_tensor, his_add_m_tensor, \
        his_buy_m_tensor, query_id_tensor, query_mask_tensor = input_tensor[:14]
        field_tensor_tuple = input_tensor[14:]

        bs = cur_id_tensor.shape[0]
        #cur_att_feat = self.get_bert_feat(cur_id_tensor, cur_mask_tensor, self.cur_fc)  # bs, 768
        _, cur_att_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)  # bs, #tokens, 768
        #print("cur_att_feat", cur_att_feat.shape)
        #his_bro_att_feat = self.get_bert_feat(his_bro_id_tensor, his_bro_m_tensor, self.his_bro_fc).reshape(bs, -1, self.feat_hidden_size)  # bs, 30, 768
        #his_add_att_feat = self.get_bert_feat(his_add_id_tensor, his_add_m_tensor, self.his_add_fc).reshape(bs, -1, self.feat_hidden_size)  # bs, 10, 768
        #his_buy_att_feat = self.get_bert_feat(his_buy_id_tensor, his_buy_m_tensor, self.his_buy_fc).reshape(bs, -1, self.feat_hidden_size)  # bs, 10, 768
        _, his_bro_att_feat = self.get_bert_cls_atten_feat(his_bro_id_tensor, his_bro_m_tensor)
        his_bro_att_feat = his_bro_att_feat.reshape(bs, -1, self.feat_hidden_size)
        _, his_add_att_feat = self.get_bert_cls_atten_feat(his_add_id_tensor, his_add_m_tensor)
        his_add_att_feat = his_add_att_feat.reshape(bs, -1, self.feat_hidden_size)
        _, his_buy_att_feat = self.get_bert_cls_atten_feat(his_buy_id_tensor, his_buy_m_tensor)
        his_buy_att_feat = his_buy_att_feat.reshape(bs, -1, self.feat_hidden_size)
        #print("his", his_bro_att_feat.shape, his_add_att_feat.shape, his_buy_att_feat.shape)
        # print("---", cur_att_feat.shape, his_bro_att_feat.shape, his_add_att_feat.shape, his_buy_att_feat.shape)
        his_bro_att_item_feat = self.attention_by_weight(his_bro_att_feat, his_bro_msk_tensor, self.his_bro_fc_item)  # bs, 768
        his_add_att_item_feat = self.attention_by_weight(his_add_att_feat, his_add_msk_tensor, self.his_add_fc_item)  # bs, 768
        his_buy_att_item_feat = self.attention_by_weight(his_buy_att_feat, his_buy_msk_tensor, self.his_buy_fc_item)  # bs, 768
        #query_feat = self.get_bert_feat(query_id_tensor, query_mask_tensor, self.his_qry_fc_item)  # bs, 768
        query_feat = self.get_bert_cls_feat(query_id_tensor, query_mask_tensor)
        # print("---", query_id_tensor.shape, query_mask_tensor.shape, query_feat.shape)
        user_stack_feat = torch.stack([his_bro_att_item_feat, his_add_att_item_feat, his_buy_att_item_feat, query_feat], dim=1)  # bs, 4, 768
        # user_mask = torch.ones((bs, 3)).long().to(self.device)
        #print("user attention start")
        usr_att_feat = self.attention_by_weight(user_stack_feat, his_msk_tensor, self.his_fc)

        itm_feat = self.item_fc(cur_att_feat)  # bs, 128
        usr_feat = self.user_fc(usr_att_feat)  # bs, 128

        # pred_prob = nn.Sigmoid()(torch.sum(itm_feat * usr_feat, dim=-1))  # bs

        base_feat = self.get_field_emb_fea(field_tensor_tuple)
        final_feat = torch.cat([itm_feat, usr_feat, base_feat], dim=-1)  # bs, (2*128 + #fields*6)
        pred_prob = nn.Sigmoid()(self.final_fc(final_feat)).squeeze(-1)
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
