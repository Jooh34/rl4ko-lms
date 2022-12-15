# rl4ko-lms

## 1. baseline training

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

####run categorical-finetune

```
python categorical-finetune/train_kobart.py
```
  
## 2. rl training

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

'''
python rl-finetune/train_kobart.py
'''
