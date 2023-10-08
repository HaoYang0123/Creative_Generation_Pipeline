## Creative Generation Pipeline

Creative generation pipeline: fine-tuning stable diffusion model and prompt with reward model for CTR target.


## Introduction
We proposed an new automated Creative Generation pipeline for Click-Through rate (CGCT) in [Shopee](https://shopee.co.id/). 


## Requirements and Installation
We recommended the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.8.0
* Details shown in requirements.txt


## Reward model
### Download data
1. The public data set can be downloaded from this [link](https://tianchi.aliyun.com/dataset/93585).
2. The pre-processed public data set (train.json and test.json) can be downloaded from this [link](https://drive.google.com/drive/folders/1YfdcPmVvVitkOMDXT7CvB6z9XHGIiite?usp=sharing).
3. The pre-processed commercial __sampled__ data set (train_commercial_sample.json and test_commercial_sample.json) can also be downloaded from the above link. Note that the whole commercial data set will be shared soon.

### Download model
1. All models including pre-trained BERT and pre-trained Swin models can be downloaded from this [link](https://drive.google.com/drive/folders/1_h7XCcbJvvYSv3H8JWqTzEBDulKjHDAs?usp=sharing).

### Generating embedding of image and title [For commercial data]

```bash
#!/bin/bash

set -x

cd reward_model/extract_embedding

bert_emb_model_path=./models/bert_emb_model.pt
bert_emb_config_path=./models/bert_emb_config.json
swin_emb_model_path=./models/swin_emb_model.pt
swin_emb_config_path=./models/swin_emb_config.json
meta_path=./models/meta2idx.json
inpath=$1  # train_commercial_sample.json/test_commercial_sample.json
save_folder_img=./img_fea_pretrain_commercial_sample
save_folder_txt=./txt_fea_pretrain_commercial_sample

google_doc_name="no_write"  # ctr_cr_uplift_exp

CUDA_VISIBLE_DEVICES=0 python -u extract_title_and_image_fea.py \
    --meta-path $meta_path \
    --save-folder-img ${save_folder_img} \
    --save-folder-txt ${save_folder_txt} \
    --test-sample-path $inpath \
    --load-pretrain \
    --batch-size 32 \
    --workers 30 \
    --write-ratio 0.2 \
    --imp-count-test -100 \
    --list-len 31 \
    --print-freq 10 \
    --write-google-name ${google_doc_name} \
    --need-img-emb \
    --need-txt-emb \
    --fix-swin \
    --fix-bert \
    --lambda-pointwise 0.1 \
    --bert-emb-checkpoint-path ${bert_emb_model_path} \
    --bert-emb-meta-path ${bert_emb_config_path} \
    --swin-emb-checkpoint-path ${swin_emb_model_path} \
    --swin-emb-meta-path ${swin_emb_config_path}

```

### Generating embedding of image and title [For public data]

```bash
#!/bin/bash

set -x

cd reward_model/extract_embedding

swin_emb_model_path=./models/swin_emb_model_public.pt
swin_emb_config_path=./models/swin_emb_config.json
meta_path=./models/meta2idx.json
inpath=$1  # train.json/test.json
save_folder_img=./img_fea_pretrain_public

google_doc_name="no_write"  # ctr_cr_uplift_exp

### step1 extract image features
CUDA_VISIBLE_DEVICES=0 python -u extract_only_image_fea_public.py \
    --meta-path ${meta_path} \
    --save-folder-img ${save_folder_img} \
    --test-sample-path ${inpath} \
    --load-pretrain \
    --batch-size 32 \
    --workers 30 \
    --write-ratio 0.2 \
    --imp-count-test -100 \
    --list-len 14 \
    --print-freq 10 \
    --write-google-name ${google_doc_name} \
    --need-img-emb \
    --fix-swin \
    --lambda-pointwise 0.1 \
    --swin-emb-checkpoint-path ${swin_emb_model_path} \
    --swin-emb-meta-path ${swin_emb_config_path}

### step2 extract title features from image features
python -u get_title_fea.py ./img_fea_pretrain_public ./title_fea_pretrain_public

```
