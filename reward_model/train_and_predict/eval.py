import json
import os
import copy
import sys
import numpy as np
import random
from sklearn.metrics import mean_squared_error


class UpliftDataLine:
    def __init__(self, json_data):
        self.creative_id = json_data["creative_id"]
        self.pred_ctr = json_data["pred_ctr"]
        self.gt_ctr = json_data["gt_ctr"]
        self.new_imghash = json_data["new_imghash"]
        self.imp = json_data["imp"]
        self.click = json_data["click"]
        self.itemid = json_data["item_id"]


class UpliftData:
    def __init__(self, data_path):
        tmp_data = open(data_path).read().strip().split("\n")
        self.item_data_dict_total = dict()
        self.hit_rate = dict()
        self.uplift_hit_rate = dict()
        self.item_hr_uplift_list = []
        self.item_imp_weight = []

        for line in tmp_data:
            uplift_data_line = UpliftDataLine(json.loads(line))
            if uplift_data_line.itemid not in self.item_data_dict_total:
                self.item_data_dict_total[uplift_data_line.itemid] = []
            self.item_data_dict_total[uplift_data_line.itemid].append(uplift_data_line)

        print("SYS_LOG : UpliftData:" + data_path + " load done!")

    def compute_base_ctr(self):
        base_imp, base_click = 0, 0
        for item_id in self.item_data_dict_total:
            val_list = self.item_data_dict_total[item_id]
            if len(val_list) <= 1: continue
            for one in val_list:
                base_imp += one.imp
                base_click += one.click
        return base_click / base_imp

    def RankScoreByCtr(self, topk=10):
        base_ctr = self.compute_base_ctr()
        print("base_ctr", base_ctr)
        click_num_list = [0 for _ in range(topk)]
        imp_num_list = [0 for _ in range(topk)]
        for item_id in self.item_data_dict_total:
            val_list = self.item_data_dict_total[item_id]
            if len(val_list) <= 1: continue
            pred_ctr_sorted_list = sorted(val_list, key=lambda x:x.pred_ctr, reverse=True)
            for t in range(topk):
                retain_order_list = pred_ctr_sorted_list[:(t+1)]
                sum_click, sum_imp = 0, 0
                for one in retain_order_list:
                    sum_click += one.click
                    sum_imp += one.imp
                click_num_list[t] += sum_click
                imp_num_list[t] += sum_imp
        ctr_top_list = [o/i for o,i in zip(click_num_list, imp_num_list)]
        ctr_uplift_top_list = [round(v/base_ctr-1,4) for v in ctr_top_list]
        print('ctr', [round(v,4) for v in ctr_top_list])
        print('ctr top5', ctr_uplift_top_list[:5])
        return np.mean(np.array(ctr_uplift_top_list[:5])), [round(v,4) for v in ctr_top_list], ctr_uplift_top_list[:5]

    def compute_mse(self, ):
        y_pred, y_true= [], []
        for item_id in self.item_data_dict_total:
            val_list = self.item_data_dict_total[item_id]
            for one_cre in val_list:
                y_pred.append(one_cre.pred_ctr)
                y_true.append(one_cre.gt_ctr)
        return mean_squared_error(y_true, y_pred)
            
if __name__ == "__main__":
    infolder = sys.argv[1]  #
    best_score = -1
    best_epoch = -1
    best_mse = -1
    best_ctr, best_uplift = None, None
    best_list = []
    if not os.path.exists(infolder):
        for idx in range(30):
            inpath = infolder + "." + str(idx) + '-' + str(idx)
            if not os.path.exists(inpath): continue
            print("inpath", inpath)
            ymal_uplift_data = UpliftData(inpath)
            print("\n")
            score, ctr, ctr_uplift = ymal_uplift_data.RankScoreByCtr(topk=20)
            mse = ymal_uplift_data.compute_mse()
            best_list.append([score, idx, ctr, ctr_uplift])
            if score >= best_score:
                best_score = score
                best_epoch = idx
                best_ctr = ctr
                best_uplift = ctr_uplift
                best_mse = mse
    else:
        ymal_uplift_data = UpliftData(infolder)
        print("\n")
        score, ctr, ctr_uplift = ymal_uplift_data.RankScoreByCtr(topk=20)
        mse = ymal_uplift_data.compute_mse()
        if score >= best_score:
            best_score = score
            best_epoch = 0
            best_ctr = ctr
            best_uplift = ctr_uplift
            best_mse = mse


    print("Best epoch", best_epoch, best_score)
    print("Best ctr", best_ctr)
    print("Best ctr uplift", best_uplift)
    print("MSE", best_mse)
    
