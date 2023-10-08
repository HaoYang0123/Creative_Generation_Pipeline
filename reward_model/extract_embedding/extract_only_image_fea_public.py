import sys, os
import json
import argparse
import time
import pickle
import random
random.seed(2022)
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import AutoImageProcessor, SwinForMaskedImageModeling
except: pass

from dataset_ext_public import CreativeDataset
from model_ext import CreativeGeneratorModel
from swin_emb_model import SwinModel as SwinEmbModel
from transformers import DistilBertTokenizer, DistilBertModel
from bert_emb_model import NLPBertModel as BertEmbModel
from utils import AverageMeter, meter_to_str, get_opt, update_opt, compute_pairwise_loss_v1

np.random.seed(44)
torch.manual_seed(44)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(44)

RBIT = 4

def main(param):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if param.cpu:
        device = torch.device("cpu")

    if param.need_img_emb:
        image_processor = AutoImageProcessor.from_pretrained(param.swin_model)
        swin_model = SwinForMaskedImageModeling.from_pretrained(param.swin_model)
    else:
        image_processor = None
        swin_model = None
    if param.need_txt_emb:
        bert_model = DistilBertModel.from_pretrained(param.bert_model)
        tokenizer = DistilBertTokenizer.from_pretrained(param.bert_model)
    else:
        bert_model = None
        tokenizer = None

    if os.path.exists(param.test_sample_path):
        test_dataset = CreativeDataset(sample_path=param.test_sample_path, meta_path=param.meta_path,
                                       image_processor=image_processor, list_len=param.list_len,
                                       need_img_emb=param.need_img_emb, need_txt_emb=param.need_txt_emb,
                                       imp_count_t=param.imp_count_test,
                                       tokenizer=tokenizer, debug=param.debug)
        test_dataloader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.workers)
    else:
        test_dataset, test_dataloader = None, None
        print("no input test data =======")
        sys.exit(-1)

    field_name_list = param.field_name_str.split(',')
    model = CreativeGeneratorModel(swin_model=swin_model, swin_out_size=param.swin_out_size,
                                   bert_model=bert_model, bert_out_size=param.bert_out_size, device=device,
                                   field_name_list=field_name_list,
                                   emb_size=param.emb_size,
                                   fc_hidden_size=param.fc_hidden_size,
                                   bin_name2bin_num=test_dataset.bin_name2bin_num,
                                   other_bin_num=param.other_bin_num,
                                   drop_prob=param.dropout,
                                   need_img_emb=param.need_img_emb,
                                   need_txt_emb=param.need_txt_emb)
    
    if param.load_pretrain:  # 如果要Load Pretrain模型，则把使用过商品图片和标题预训练过的Swin和Bert进行加载
        if os.path.exists(param.swin_emb_checkpoint_path):
            print("load state from swin-emb", param.swin_emb_checkpoint_path)
            if not os.path.exists(param.swin_emb_meta_path):
                print("[Erorr] no input swin emb meta path")
                sys.exit(-1)
            with open(param.swin_emb_meta_path, encoding='utf8') as f:
                swin_meta_info = json.load(f)
            swin_emb_model = SwinEmbModel(swin_model=swin_model,
                              image_processor=image_processor,
                              device=device, feat_size=param.swin_out_size,
                              class_num_1=len(swin_meta_info['level1']) + 1,
                              class_num_2=len(swin_meta_info['level2']) + 1,
                              class_num_3=len(swin_meta_info['level3']) + 1)
            swin_emb_model.load_state_dict(torch.load(param.swin_emb_checkpoint_path)["model_state"])
            model.swin_model.swin_model = swin_emb_model.swin_model
            print("load swin-emb complete")
    else:
        print("no load swin and bert pretrain model.")
    model = model.to(device)
    extract_feature(model, test_dataset, test_dataloader, device, 0, param)

def extract_feature(model, test_dataset, test_dataloader, device, epoch, param):
    save_folder_img = param.save_folder_img
    if not os.path.exists(save_folder_img):
        os.mkdir(save_folder_img)
    model.eval()
    tmp_pidx = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader)):
            for name in batch:
                batch[name] = batch[name].to(device)
                
            img_fea = model.extract_img(batch)  # (bs*#creatives), img_dim
            
            bs = batch['valid'].shape[0]
            img_fea = img_fea.reshape(bs, -1, img_fea.shape[-1])   # bs, #creatives, img_dim
            # print("--->>>", img_fea.shape)
            img_fea_list = img_fea.tolist()
            
            # get itemid
            valid_list = batch['valid']  # bs, #creatives
            for tidx in range(img_fea.shape[0]):
                cre_valid_list = valid_list[tidx]   # #creatives
                img_flist = img_fea_list[tidx]  #  #creatives, img_dim

                for one_cre, valid, img in zip(test_dataset.samples[tmp_pidx], cre_valid_list, img_flist):
                    creative_id = one_cre['creative_id']
                    if int(valid) != 1:
                        print("[Warning] not creative", creative_id)
                        continue
                    write_path = os.path.join(save_folder_img, f"{creative_id}.pkl")
                    with open(write_path, 'wb') as fw:
                        pickle.dump(img, fw)
                    
                tmp_pidx += 1


if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--meta-path", type=str,
                       default=r"", help="Data path hash")
    param.add_argument("--outpath", type=str,
                       default=r"", help="Data path hash")
    param.add_argument("--save-folder-img", type=str,
                       default=r"", help="save img feature folder")
    param.add_argument("--write-ratio", type=float, default=1.2, help="")
    param.add_argument("--write-google-name", type=str, default="ctr_cr_uplift_exp", help="")
    param.add_argument("--test-sample-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--checkpoint-path", type=str,
                       default="", help="Pre-trained model path")
    param.add_argument("--swin-emb-checkpoint-path", type=str,
                       default="", help="Swin embedding model path")
    param.add_argument("--swin-emb-meta-path", type=str, default="", help="Swin embedding config path")
    param.add_argument("--bert-emb-checkpoint-path", type=str, default="", help="Bert embedding model path")
    param.add_argument("--bert-emb-meta-path", type=str, default="", help="Bert embedding config path")  # meta-->config
    param.add_argument("--batch-size", type=int,
                       default=4, help="Batch size of samples")
    param.add_argument("--workers", type=int,
                       default=0, help="Workers of dataLoader")
    param.add_argument("--print-freq", type=int,
                       default=10, help="Frequency for printing training progress")

    # model parameters:
    param.add_argument("--swin-model", type=str,
                       default="microsoft/swin-base-patch4-window7-224", help="Swin model name")  # microsoft/swin-tiny-patch4-window7-224
    param.add_argument("--fix-swin", action='store_true')
    param.add_argument("--num-step-swin", type=int, default=100000000, help="")
    param.add_argument("--swin-out-size", type=int, default=1024, help="")   # 768
    param.add_argument("--bert-model", type=str, default="cahya/distilbert-base-indonesian", help="Bert model name")
    param.add_argument("--fix-bert", action='store_true')
    param.add_argument("--num-step-bert", type=int, default=100000000, help="")
    param.add_argument("--bert-out-size", type=int, default=768, help="")
    param.add_argument("--field-name-str", type=str,
                       default="scene,temid,poolid,aspid,cate1,cate2,tem_red,tem_green,tem_blue,obj_red,obj_green,obj_blue,"
                               "obj_ratio,basemap_label,basemap_prob,display_label,display_prob,salience_label,salience_prob,is_head_img,"
                               "cross_temidcate1,cross_temidcate2,cross_poolidcate1,cross_poolidcate2,cross_aspidcate1,cross_aspidcate2", help="")
    param.add_argument("--list-len", type=int, default=10, help="")
    param.add_argument("--emb-size", type=int, default=10, help="")
    param.add_argument("--fc-hidden-size", type=int, default=128, help="")
    param.add_argument("--other-bin-num", type=int, default=101, help="")
    param.add_argument("--dropout", type=float, default=0.1, help="")
    param.add_argument("--lambda-pointwise", type=float, default=0.5, help="weight for pointwise loss")
    param.add_argument("--lambda-ctr", type=float, default=0.5, help="weight of ctr and (1-lambda-ctr) is for weight of cr")
    # param.add_argument("--mask-txt-ratio", type=float, default=0.1)
    # param.add_argument("--max-mask-num", type=int, default=3)

    param.add_argument("--need-img-emb", action='store_true')
    param.add_argument("--need-txt-emb", action='store_true')
    param.add_argument("--debug", action='store_true')
    param.add_argument("--cpu", action='store_true')
    param.add_argument("--load-pretrain", action='store_true')

    param.add_argument("--imp-count-test", type=int, default=100, help="impression number threshold for testing data")

    param = param.parse_args()
    print("Param", param)
    main(param)
