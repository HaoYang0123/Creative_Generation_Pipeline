import pygsheets
import json
import random
import numpy as np


def acc_eval(itemid2info, epoch, param, eval_step=0):
    print("--->>>", param.outpath + "." + str(epoch) + "-" + str(eval_step))
    fw = open(param.outpath + "." + str(epoch) + "-" + str(eval_step), 'w', encoding='utf8')
    for itemid in itemid2info:
        all_creatives_list = itemid2info[itemid]
        for one_cre in all_creatives_list:
            fw.write(json.dumps(one_cre) + '\n')      
    fw.close()

    