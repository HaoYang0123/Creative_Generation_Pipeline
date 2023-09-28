#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """计算三元组损失，
    loss=(cos(x,y) - cos(x',y) - margin) + (cos(x,y) - cos(x,y') - margin)
    according to: https://en.wikipedia.org/wiki/Triplet_loss"""

    def __init__(self, margin, lambda_value, device):
        """
        :param margin: 三元组损失中的margin参数（一般设置为0.1）
        :param lambda_value: hardest negative adv中设置的lambda值（一般设置为1）
        :param device: 计算设备（CPU或者GPU）
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.lambda_value = lambda_value
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

    def forward(self, anc, pos, neg, loss_type):
        """
        :param anc: anchor特征，bs * dim
        :param pos: positive特征，bs * dim
        :param neg: negative特征，bs * dim
        :param loss_type: string, 使用Loss的方式（e.g., base, online semi-hard, online hardest, online hardest adv)
        anc[i]和pos[i]为对应的正样本
        anc[i]和neg[j] (for every j) 为对应的负样本
        :return: 1) cost损失，2) index选择的最难负样本
        """
        anc = self.norm_matrix(anc)  # 按行对各样本进行特征归一化
        pos = self.norm_matrix(pos)  # 按行对各样本进行特征归一化
        neg = self.norm_matrix(neg)  # 按行对各样本进行特征归一化
        n = anc.size(0)  # 样本数量

        if loss_type == "base":
            sim_anc_pos = torch.cosine_similarity(anc, pos, dim=1)  # shape: bs
            sim_anc_neg = torch.cosine_similarity(anc, neg, dim=1)  # shape: bs
            # print("----", sim_anc_pos, sim_anc_neg)
            cost = (self.margin + sim_anc_neg - sim_anc_pos).clamp(min=0)
            index = torch.LongTensor([i for i in range(n)]).to(anc.device)
            return cost.mean(), index
        elif loss_type == "online semi-hard":
            sim_anc_pos = torch.cosine_similarity(anc, pos, dim=1).view(n, 1)
            scores = torch.mm(anc, neg.t())  # 负样本的打分，shape: bs * bs
            d = sim_anc_pos.expand_as(scores)  # 正样本打分
            mask = (scores < d).float()  # 只取那些负样本打分比正样本打分低的负样本
            cost = (mask * (self.margin + scores - d)).clamp(min=0)
            cost, index = cost.max(1)  # cost如果全部是0，其实这一个样本不能要了
            return cost.mean(), index
        elif loss_type == "online hardest":
            sim_anc_pos = torch.cosine_similarity(anc, pos, dim=1).view(n, 1)
            scores = torch.mm(anc, neg.t())  # 负样本的打分，shape: bs * bs
            d = sim_anc_pos.expand_as(scores)  # 正样本打分
            cost = (self.margin + scores - d).clamp(min=0)
            cost, index = cost.max(1)
            return cost.mean(), index
        elif loss_type == "online hardest adv":
            sim_anc_pos = torch.cosine_similarity(anc, pos, dim=1).view(n, 1)
            scores = torch.mm(anc, neg.t())  # 负样本的打分，shape: bs * bs
            d = sim_anc_pos.expand_as(scores)  # 正样本打分
            cost = (self.margin + scores - d).clamp(min=0)
            cost, index = cost.max(1)  # cost记录最像负样本与正样本的loss，index记录对应的最像负样本的索引
            mask = (cost > self.margin).float()  # 找到最像负样本的score>正样本的情况
            cost1 = cost * (1 - mask)  # 去掉那些最像负样本的score>正样本的情况
            cost2 = self.lambda_value * ((scores + 1).gather(1, index.view(-1, 1)).squeeze(1)) * mask
            # 最像负样本的score>正样本的情况，仅使用负样本的score
            cost = cost1 + cost2
            # print("scores", scores)  # 负样本打分
            # print("d", d)  # 正样本打分
            # print("cost1", cost1)
            # print("cost2", cost2)
            return cost.mean(), index
        elif loss_type == "online hardest adv v2" or loss_type == "online hardest v2":
            Pos = torch.cosine_similarity(anc, pos, dim=1)
            scores = torch.mm(anc, neg.t())  # 负样本的打分，shape: bs * bs

            V_neg, I_neg = scores.max(1)
            Neg = scores[torch.arange(0, n), I_neg]

            # Mask hard/easy triplets
            HardTripletMask = ((Neg > Pos) | (Neg > 0.8))
            EasyTripletMask = ((Neg < Pos) & (Neg < 0.8))

            # triplets
            Triplet_val = torch.stack([Pos, Neg], 1)

            if loss_type == "online hardest adv v2":
                loss_hardtriplet = Neg[HardTripletMask].sum()
                loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask, :] / 0.1, dim=1)[:, 0].sum()

                N_hard = HardTripletMask.float().sum()
                N_easy = EasyTripletMask.float().sum()
                if torch.isnan(loss_hardtriplet) or N_hard == 0:
                    loss_hardtriplet, N_hard = 0, 0
                    print("No hard triplets in the batch")
                if torch.isnan(loss_easytriplet) or N_easy == 0:
                    loss_easytriplet, N_easy = 0, 0
                    print("No easy triplets in the batch")

                N = N_hard + N_easy
                # print("N-N", N_easy, N_hard, loss_easytriplet, loss_hardtriplet)
                if N == 0: N = 1
                loss = (loss_easytriplet + self.lambda_value * loss_hardtriplet) / N
                return loss, I_neg
            else:
                loss = -F.log_softmax(Triplet_val / 0.1, dim=1)[:, 0].mean()
                return loss, I_neg
        else:
            raise NotImplementedError
        return None

    def forward_ori(self, img, txt):
        """
        :param img: 图片特征，bs * dim
        :param txt: 文本特征，bs * dim
        img[i]和txt[i]为对应的正样本
        img[i]和txt[j] (every j != i) 为对应的负样本
        :return:
        """
        img = self.norm_matrix(img)  # 按行对各样本进行特征归一化
        txt = self.norm_matrix(txt)  # 按行对各样本进行特征归一化
        n = img.size(0)  # 样本数量
        scores = torch.mm(img, txt.t())
        # scores形状为 bs * bs, scores[i][j]表示第i个图片特征与第j个文本间的cosine相似度打分
        diagonal = scores.diag().view(n, 1)  # 对角线为正样本的打分
        d1 = diagonal.expand_as(scores)  # 正样本打分
        d2 = diagonal.t().expand_as(scores)  # 正样本打分

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(n) > .5
        I = Variable(mask).to(self.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if self.max_violation:  # 只考虑最相似的负样本的打分
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.mean() + cost_im.mean()
