# rl4ko-lms
## Overview
Training Korean Text-generation Model using Reinforcement Learning

<img width="611" alt="overview" src="https://user-images.githubusercontent.com/15865928/207782306-3035ebee-e2fe-4f42-b9a3-4b5f1dddff8b.png">

## Data preparation
1. sign up and download from
[AI Hub 문서 요약 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)

2. place to ./data/ko_summarization/


## 1. Baseline training

modify config in categorical-finetune/train_kobart.py
```
cfg = {
    'batch_size': 4,
    'accumulate_grad_batches' : 8,
    'epochs': 3,
    'lr': 2e-5,
    "warmup_ratio": 0.2,
    'checkpoint_path': CHPT_PATH,
    "seed": 3444,
    'num_to_rouge': 50,
}
```

#### run categorical-finetune

```
python categorical-finetune/train_kobart.py
```
  
## 2. RL training

modify config in rl-finetune/train_kobart.py 

```
cfg = {
    'batch_size': 1, # rl supports only one batch
    'accumulate_grad_batches' : 8,
    'epochs': 2,
    'lr': 1e-7,
    "warmup_ratio": 0.1,
    'checkpoint_path': CHPT_PATH,
    "seed": 3444,
    'K_epochs': 30,
    'num_to_rouge': 100,
    'pretrained_path' : './checkpoints/kobart-2022-11-21-19-22-15.pt',
    'calc_values' : True,
    'use_beamsearch' : True,
}
```

#### run rl-finetune

```
python rl-finetune/train_kobart.py
```
