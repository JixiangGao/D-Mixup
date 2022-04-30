# D-Mixup: Dynamic Mixup for Multi-Label Long-Tailed Food Ingredient Recognition

<img src="https://github.com/JixiangGao/D-Mixup/blob/main/pics/framework-overview.png" alt="D-Mixup">

## Overview

**D-Mixup** is simple yet effective approach for multi-label long-tailed food ingredient recognition which includes two parts **Dynamic Mixup Sampler** and **Multi-label Mixer**. The former can dynamically select the images based on the previous recognition performance for Multi-label Mixer as inputs, which enhances tail classes that are poorly recognized. The latter performs multi-label images mixup operation, a straightforward data augmentation principle that constructs virtual images as the linear interpolation of two images from the training set and combines their labels in the union manner. We provide the implement of the **D-Mixup** method in this repo.


## Data Preparation

### Datasets

- [VIREO Food 172](http://vireo.cs.cityu.edu.hk/vireofood172/)
- [UEC Food 100](http://foodcam.mobi/dataset100.html)

### Prepare Data Path

1. Use the `args.dataset_dir` to indicate the dataset path.

2. Set `imgs_dir` and `label_dir` in `commins/VireoFoodDatasets.py` to `images` and `labels` folders. 

## Usage

### Requirement

```
pip install -r requirements.txt
```

### Train

We provide scripts for training. 


Example training scripts:

```
python main.py \
    --mode train \
    --use_cuda True \
    --gpu_device 0 \
    --lr 0.0001 \
    --lr_gamma 0.8 \
    --saved_model_dir /path/to/your/save/dir \
    --print_freq 500 \
    --save_freq 1 \
    --load_model_num 100 \
    --tail_threshold 20 \
    --head_threshold 1000  \
    --epoch 200 \
    --use_writer \
    --mix_style mixup-all-all \
    --labels_mixup_style 1 \
    --eval_mode val \
    --dynamic_mode square \
    --not_dropout \
    --batch_size 64 \
    --backbone_net Resnet50_OneFC_Sigmoid \
    --dataset_dir /path/to/VireoFood172/ \
    --label_num 346 \
    --step_size 5000 \
    --input_size 224 \
```

Note: Set `args.backbone_net`, `args.labels_mixup_style` can apply multiple backbone networks and mixup strategies.

Options:
```
args.backbone_net: ['VGG19', 'Resnet50','Resnet101', 'DenseNet121', 'EfficientNetB7', 'SENet' ...]
args.labels_mixup_style : ['1', 'lambda']
```


### Test

Set `args.mode` to `validation` or `test`.


## Main results

<img src="https://github.com/JixiangGao/D-Mixup/blob/main/pics/main-results.png" alt="Main results">


