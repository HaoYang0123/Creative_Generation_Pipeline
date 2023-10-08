import os, sys
import numpy as np
import pickle
import json,copy


img_folder = sys.argv[1]  #"img_fea_pretrain_public"
train_path = "train.json"
test_path = "test.json"
find_dim = 1024
out_folder = sys.argv[2]  #"title_fea_pretrain_public"

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

with open(train_path) as f:
    train_d = json.load(f)

with open(test_path) as f:
    test_d = json.load(f)

all_d = copy.deepcopy(train_d)

for item in test_d:
    if item not in all_d: all_d[item] = []
    all_d[item] += test_d[item]

print("#train", len(train_d), "#test", len(test_d))
print("#all", len(all_d))

itemid2img_list = {}

for item in all_d:
    cre_list = all_d[item]
    img_list = []
    for one in cre_list:
        img_list.append(one['creative_id'])
    itemid2img_list[item] = img_list

for item in itemid2img_list:
    img_list = itemid2img_list[item]
    fea_list = []
    for imgid in img_list:
        img_path = os.path.join(img_folder, f"{imgid}.pkl")
        if not os.path.exists(img_path):
            print("No exists", img_path)
            continue
        with open(img_path, 'rb') as f:
            fea = pickle.load(f)
            if len(fea) != find_dim: 
                print("not size", imgid)
                continue
            #print("type", type(fea))
            fea_list.append(fea)
    #print("ffff", np.array(fea_list).shape)
    item_title_fea = np.mean(np.array(fea_list), axis=0)
    #print("--->>", item_title_fea.shape)
    title_path = os.path.join(out_folder, f"{item}.pkl")
    with open(title_path, 'wb') as fw:
        pickle.dump(item_title_fea.tolist(), fw)
