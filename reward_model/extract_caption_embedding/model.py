import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertModel
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertPreTrainingHeads


class NLPBertModel(nn.Module):
    def __init__(self, bert_model, device, model_flag="bert"):
        super(NLPBertModel, self).__init__()

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

    def norm_matrix(self, emb, dim=1):
        """
        特征归一化
        :param emb: 输入的特征，bs * dim
        :param dim: 按行或列进行归一化
        :return:
        """
        emb_norm = emb.norm(p=2, dim=dim, keepdim=True)
        return emb.div(emb_norm)

    def forward_emb(self, cur_id_tensor, cur_mask_tensor):
        atten_probs, atten_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)
        return atten_probs, atten_feat

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

    def forward(self, cur_id_tensor, cur_mask_tensor, masked_pos_tensor):
        # segment_ids = torch.zeros_like(cur_mask_tensor).long()
        bert_feat = self.bert(cur_id_tensor, cur_mask_tensor)
        bert_feat = bert_feat.last_hidden_state
        mask_feature = torch.gather(bert_feat, 1,
                                    masked_pos_tensor.unsqueeze(2).expand(-1, -1, bert_feat.shape[-1]))
        #print("mask_feature", bert_feat.shape, mask_feature.shape, masked_pos_tensor.shape)
        #print("ffflag", self.model_flag)
        if self.model_flag == "bert":
            prediction_scores, _ = self.cls(mask_feature, bert_feat)
            #print("pred", prediction_scores.shape)
        else:
            prediction_logits = self.vocab_transform(mask_feature)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_scores = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            #print('score', prediction_scores.shape)
        #print("score", prediction_scores.shape)
        return prediction_scores