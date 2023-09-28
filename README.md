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
1. The public data set can be download from this [link]([https://www.kaggle.com/c/kddcup2012-track2](https://tianchi.aliyun.com/dataset/93585)).


### Generating embedding of image and title

```bash
#!/bin/bash

set -x

cd extract_embedding

bert_emb_model_path=$1
bert_emb_config_path=$2
meta_path=$3
swin_emb_model_path=$4
swin_emb_config_path=$5
inpath=$6

google_doc_name="no_write"  # ctr_cr_uplift_exp

CUDA_VISIBLE_DEVICES=0 python -u extract_title_and_image_fea.py \
    --meta-path $meta_path \
    --save-folder-img ./img_fea_pretrain \
    --save-folder-txt ./txt_fea_pretrain \
    --test-sample-path $inpath \
    --load-pretrain \
    --batch-size 8 \
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
