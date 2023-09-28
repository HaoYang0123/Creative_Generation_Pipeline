import sys, os
import json
import argparse
import time
import traceback
import random
random.seed(2022)
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
try:
    from transformers import AutoImageProcessor, SwinForMaskedImageModeling
except: pass

from dataset_speedup import CreativeDataset
from model_speedup import CreativeGeneratorModel
from swin_emb_model import SwinModel as SwinEmbModel
from transformers import DistilBertTokenizer, DistilBertModel
from bert_emb_model import NLPBertModel as BertEmbModel
from utils import AverageMeter, meter_to_str, get_opt, update_opt, compute_pairwise_loss
from eval_util import acc_eval
from triplet_loss import TripletLoss

np.random.seed(44)
torch.manual_seed(44)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(44)

RBIT = 4

def main(param):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if param.cpu:
        device = torch.device("cpu")

    if torch.cuda.device_count() > 1:
        param.batch_size *= torch.cuda.device_count()

    if not os.path.exists(param.model_folder):
        os.mkdir(param.model_folder)

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

    train_dataset = CreativeDataset(sample_path=param.sample_path,
                                    image_processor=image_processor, list_len=param.list_len,
                                    need_img_emb=param.need_img_emb, need_txt_emb=param.need_txt_emb,
                                    need_cap_emb=param.need_cap_emb, imp_count_t=param.imp_count_train,
                                    weight_pointwise=param.weight_pointwise, weight_listwise=param.weight_listwise,
                                    tokenizer=tokenizer, txt_folder=param.txt_folder, img_folder=param.img_folder, 
                                    cap_folder=param.cap_folder, debug=param.debug, train_flag=True, 
                                    local_file=param.local_file, norm_weight=param.norm_weight, 
                                    txt_input_dim=param.txt_input_dim, img_input_dim=param.swin_out_size)
    print("#training samples", train_dataset.__len__())
    if param.public:
        print("use sample weight for public")
        train_sample_weight = train_dataset.get_train_weight()
        train_sampler = WeightedRandomSampler(train_sample_weight,
                                                  len(train_sample_weight),
                                                  replacement=False)
        train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size, 
                                      sampler=train_sampler, num_workers=param.workers)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size, num_workers=param.workers)

    if param.test_sample_path.find(',') >= 0 or os.path.exists(param.test_sample_path):
        test_dataset = CreativeDataset(sample_path=param.test_sample_path,
                                       image_processor=image_processor, list_len=param.list_len,
                                       need_img_emb=param.need_img_emb, need_txt_emb=param.need_txt_emb,
                                       need_cap_emb=param.need_cap_emb, imp_count_t=param.imp_count_test,
                                       weight_pointwise=param.weight_pointwise, weight_listwise=param.weight_listwise,
                                       tokenizer=tokenizer, txt_folder=param.txt_folder, img_folder=param.img_folder,
                                       cap_folder=param.cap_folder, debug=param.debug, train_flag=False, 
                                       local_file=param.local_file, norm_weight=param.norm_weight, 
                                       txt_input_dim=param.txt_input_dim, img_input_dim=param.swin_out_size)
        test_dataloader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.workers)
    else:
        test_dataset, test_dataloader = None, None

    fc_hidden_size = [int(v.strip()) for v in param.fc_hidden_size.split(',')]
    model = CreativeGeneratorModel(swin_out_size=param.swin_out_size, bert_out_size=param.txt_input_dim,
                                   cap_bert_out_size=param.cap_bert_out_size, device=device,
                                   fc_hidden_size=fc_hidden_size, drop_prob=param.dropout,
                                   need_img_emb=param.need_img_emb,
                                   need_txt_emb=param.need_txt_emb,
                                   need_cap_emb=param.need_cap_emb,
                                   without_transformer=param.without_transformer)
    print("model", model)
    if os.path.exists(param.checkpoint_path):
        print("load state", param.checkpoint_path)
        model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
        print("load complete")
    print("model", model)
    if torch.cuda.device_count() > 1:
        print("#gpus", torch.cuda.device_count())
        model = nn.DataParallel(model)
    train(model, device, train_dataset, train_dataloader, test_dataset, test_dataloader, param)

def _round_list(list_list, r=2):
    for idx in range(len(list_list)):
        for jdx in range(len(list_list[idx])):
            list_list[idx][jdx] = round(list_list[idx][jdx], r)
    

def train(model, device, train_dataset, train_dataloader, test_dataset, test_dataloader, param, start_epoch=0):
    model.train()
    model = model.to(device)

    sim_loss_func = TripletLoss(margin=0.2, lambda_value=1.0, device=device)
    criterion = nn.LogSoftmax(dim=1)  # ListWise loss
    aux_criterion = nn.MSELoss(reduction='none')  # PointWise loss
    pairwise_criterion = nn.MarginRankingLoss(margin=0.0)  # PairWise loss for top1/top2

    eval_step = max(1, int(len(train_dataloader) * param.eval_freq))
    save_step = max(1, int(len(train_dataloader) * param.save_freq))
    print("evaluate step=====", eval_step)
    print("save step=====", save_step)
    optimizer = get_opt(model, param)  # 根据Swin/Bert是否更新的参数，来得到相应的优化器

    train_step = 0
    for epoch in range(start_epoch, param.epoches):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        tr_loss = 0.0
        start = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_pointwise = AverageMeter()
        losses_listwise = AverageMeter()
        losses_sim = AverageMeter()
        
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            train_step += 1
            data_time.update(time.time() - start)
            for name in batch:
                batch[name] = batch[name].to(device)

            try:
                pred_ctr, img_emb, tit_emb, neg_tit_emb, final_emb = model(batch)
                # img_emb/tit_emb: (bs*#cre), 256
            except Exception as err:
                print("[Warning] batch1norm don't support for batch_size=1", err, traceback.format_exc())
                continue

            # 计算CTR目标的Loss：1）ListWise Loss；2）PointWise Loss
            list_loss_ctr = -torch.sum(criterion(pred_ctr) * batch['label_rank'].detach() * batch['weight']) / \
                       (torch.sum(batch['weight']) + 1e-9)  # 计算ListWise Loss
            aux_loss_ctr = aux_criterion(pred_ctr.float(), batch['label_ctr'].float())
            aux_loss_ctr = torch.sum(aux_loss_ctr * batch['weight']) / (torch.sum(batch['weight']) + 1e-9) # 根据创意的曝光量来加权，计算PointWise Loss
            # print("Loss ctr", loss_ctr, aux_loss_ctr)
            if param.lambda_sim > 0 and img_emb is not None and tit_emb is not None and neg_tit_emb is not None:
                loss_sim, _ = sim_loss_func(img_emb, tit_emb, neg_tit_emb, param.loss_type)
            else:
                loss_sim = torch.FloatTensor([0.0]).to(device)
            loss = (1 - param.lambda_pointwise) * list_loss_ctr + param.lambda_pointwise * aux_loss_ctr  # + param.lambda_sim * loss_sim

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tr_loss += loss.item()
            losses.update(loss.item())
            losses_pointwise.update(aux_loss_ctr.item())
            losses_listwise.update(list_loss_ctr.item())
            losses_sim.update(loss_sim.item())
            batch_time.update(time.time() - start)
            start = time.time()
            # print("model bert", model.bert.encoder.layer[0].output.dense.weight.requires_grad,
            #       model.bert.encoder.layer[0].output.dense.weight)

            if (step + 1) % param.print_freq == 0:
                print("Epoch:{}-{}/{}, loss: [{}], loss-pointwise[{}], loss-listwise[{}], loss-sim[{}], [{}], [{}] ".\
                      format(epoch, step, len(train_dataloader), meter_to_str("Loss", losses, RBIT),
                             meter_to_str("", losses_pointwise, RBIT),
                             meter_to_str("", losses_listwise, RBIT),
                             meter_to_str("", losses_sim, RBIT),
                             meter_to_str("Batch_Time", batch_time, RBIT),
                             meter_to_str("Data_Load_Time", data_time, RBIT)))
            update_opt(model, optimizer, param, train_step)
            if (step + 1) % eval_step == 0 and test_dataloader:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save({'epoch': epoch, 'model_state': model.module.state_dict(), 'loss': tr_loss},
                               os.path.join(param.model_folder, f'checkpoint_{epoch}_{(step + 1)}.pt'))
                else:
                    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'loss': tr_loss},
                           os.path.join(param.model_folder, f'checkpoint_{epoch}_{(step+1)}.pt'))
                evaluate(model, test_dataset, test_dataloader, device, epoch, param, eval_step=(step+1))
            if (step + 1) % save_step == 0:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save({'epoch': epoch, 'model_state': model.module.state_dict(), 'loss': tr_loss},
                               os.path.join(param.model_folder, f'checkpoint_{epoch}_{(step + 1)}.pt'))
                else:
                    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'loss': tr_loss},
                           os.path.join(param.model_folder, f'checkpoint_{epoch}_{(step+1)}.pt'))

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}minute".format(epoch, tr_loss,
                                                                                 (time.time() - train_start) / 60.0))
        if isinstance(model, torch.nn.DataParallel):
            torch.save({'epoch': epoch, 'model_state': model.module.state_dict(), 'loss': tr_loss},
                   os.path.join(param.model_folder, f'checkpoint_{epoch}.pt'))
        else:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'loss': tr_loss},
                   os.path.join(param.model_folder, f'checkpoint_{epoch}.pt'))

        if test_dataloader:
            evaluate(model, test_dataset, test_dataloader, device, epoch, param, eval_step=str(epoch))


def evaluate(model, test_dataset, predict_dataloader, device, epoch_th, param, eval_step="last"):
    # print("***** Running prediction *****")
    model.eval()
    start = time.time()
    aux_criterion = nn.MSELoss(reduction='mean')
    criterion = nn.LogSoftmax(dim=1)

    losses = AverageMeter()
    losses_ctr = AverageMeter()
    losses_cr = AverageMeter()

    itemid2creative_id2ctr_cr = {}
    pred_idx = 0
    with torch.no_grad():
        for step, batch in enumerate(predict_dataloader):
            for name in batch:
                batch[name] = batch[name].to(device)
            pred_ctr, _, _, _, final_emb = model(batch)
            # print("final_emb", final_emb[0,0,:])
            # print("final_emb", final_emb.shape)
            
            pred_ctr_list = pred_ctr.tolist()
            gt_ctr_list = batch['label_ctr'].tolist()
            final_emb_list = final_emb.tolist()
            valid_list = batch['valid']  # bs, #creatives
            for (one_pred_ctr, one_gt_ctr, one_valid, one_emb) \
                    in zip(pred_ctr_list, gt_ctr_list, valid_list, final_emb_list):
                pred_list = []
                cre_imp, cre_num, creative_list = test_dataset.samples[pred_idx]
                for one_idx, one_cre in enumerate(creative_list):
                    if one_valid[one_idx] != 1:
                        print("[Warning] not has creative")
                        continue  # 没有创意，或者创意的Img load失败
                    creativeid = one_cre['creative_id']
                    itemid = one_cre['item_id']
                    newimghash = one_cre['creative_image_hash']
                    imp, click = one_cre.get('exp_impression_cnt', 1), one_cre.get('exp_click_cnt', 1)
                    base_imp, base_click = one_cre.get('base_impression_cnt', 1), one_cre.get('base_click_cnt', 1)

                    pred_list.append({
                        'item_id': itemid,
                        'creative_id': creativeid,
                        'pred_ctr': one_pred_ctr[one_idx],
                        'gt_ctr': one_gt_ctr[one_idx],
                        'new_imghash': newimghash,
                        'imp': imp,
                        'click': click,
                        'emb': ','.join(map(str, one_emb[one_idx])),
                    })
                pred_idx += 1
                itemid2creative_id2ctr_cr[itemid] = pred_list
            

            if (step + 1) % param.print_freq == 0:
                print(f"evaluate on epoch={epoch_th}, inter-step={step}/{len(predict_dataloader)}")
    # print("pred_ans", pred_ans)
    # print("label_ans", label_ans)

    model.train()
    print("acc_eval start--------")
    acc_eval(itemid2creative_id2ctr_cr, epoch_th, param, eval_step)


if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--sample-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--outpath", type=str,
                       default=r"", help="Data path hash")
    param.add_argument("--txt-folder", type=str,
                       default=r"", help="Title feature folder")
    param.add_argument("--img-folder", type=str,
                       default=r"", help="Image feature folder")
    param.add_argument("--cap-folder", type=str,
                       default=r"", help="Caption feature folder")
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
    param.add_argument("--epoches", type=int,
                       default=1, help="Epoches")
    param.add_argument("--learning-rate", type=float,
                       default=5e-5, help="Learning rate for BERT when training")  #TODO
    param.add_argument("--learning-rate-swin", type=float,
                       default=1e-5, help="Learning rate for BERT when training")
    param.add_argument("--learning-rate-bert", type=float,
                       default=1e-5, help="Learning rate for BERT when training")
    param.add_argument("--clip", type=float,
                       default=0.25, help="Learning rate for CRF")
    param.add_argument("--model-folder", type=str,
                       default="./models", help="Folder for saved models")
    param.add_argument("--print-freq", type=int,
                       default=10, help="Frequency for printing training progress")
    param.add_argument("--eval-freq", type=float, default=1.1, help="Percentage of evaluation")
    param.add_argument("--save-freq", type=float, default=1.1, help="Percentage of evaluation")
    param.add_argument("--loss-type", type=str, default="base",
                       help="Types of loss function (e.g., base, online semi-hard, online hardest, online hardest adv)")

    # model parameters:
    param.add_argument("--swin-model", type=str,
                       default="microsoft/swin-base-patch4-window7-224", help="Swin model name")  # microsoft/swin-tiny-patch4-window7-224
    param.add_argument("--fix-swin", action='store_true')
    param.add_argument("--num-step-swin", type=int, default=100000000, help="")
    param.add_argument("--swin-out-size", type=int, default=1024, help="")   # 768
    param.add_argument("--bert-model", type=str, default="cahya/distilbert-base-indonesian", help="Bert model name")
    param.add_argument("--fix-bert", action='store_true')
    param.add_argument("--fix-cap-bert", action='store_true')
    param.add_argument("--num-step-bert", type=int, default=100000000, help="")
    param.add_argument("--cap-bert-out-size", type=int, default=768, help="")
    param.add_argument("--list-len", type=int, default=10, help="")
    param.add_argument("--fc-hidden-size", type=str, default="256,128,1", help="")
    param.add_argument("--dropout", type=float, default=0.1, help="")
    param.add_argument("--lambda-pointwise", type=float, default=0.5, help="weight for pointwise loss")
    param.add_argument("--lambda-sim", type=float, default=0.0, help="weight for pointwise loss")
    # param.add_argument("--mask-txt-ratio", type=float, default=0.1)
    # param.add_argument("--max-mask-num", type=int, default=3)

    param.add_argument("--weight-pointwise", type=float, default=100, help="multiple metrix for pointwise")
    param.add_argument("--weight-listwise", type=float, default=1000, help="multiple metrix for listwise")
    param.add_argument("--imp-count-train", type=int, default=100, help="impression number threshold for training data")
    param.add_argument("--imp-count-test", type=int, default=100, help="impression number threshold for testing data")
    param.add_argument("--txt-input-dim", type=int, default=768, help="input dimension from title feature")
    param.add_argument("--need-img-emb", action='store_true')
    param.add_argument("--need-txt-emb", action='store_true')
    param.add_argument("--need-cap-emb", action='store_true')
    param.add_argument("--local-file", action='store_true')
    param.add_argument("--norm-weight", action='store_true')
    param.add_argument("--without-transformer", action='store_true')
    param.add_argument("--debug", action='store_true')
    param.add_argument("--public", action='store_true')
    param.add_argument("--cpu", action='store_true')
    param.add_argument("--write-emb", action='store_true')

    param = param.parse_args()
    print("Param", param)
    main(param)
