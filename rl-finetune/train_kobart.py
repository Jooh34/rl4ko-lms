import os
import sys
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
from tqdm import tqdm
from datetime import datetime
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import wandb
import numpy as np
import random
import copy
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))
from common.dataset import KoSummarizationDataset
from ppo import PPO

from common.dataset import KoSummarizationDataset
from common.rouge import RougeScorer, MyScore

time_format = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
CHPT_PATH = './checkpoints/kobart-summarization/ppo-{}.pt'.format(time_format)

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
    'pretrained_path' : None,
    # 'pretrained_path' : './checkpoints/kobart-2022-11-21-19-22-15.pt',
    'calc_values' : True,
    'use_beamsearch' : True,
}
wandb.init(project='rl4kolms-kobart-ppo', config=cfg)

torch.manual_seed(cfg['seed'])
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
pad_token_id = tokenizer.pad_token_id
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2').to(device)
old_model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2').to(device)
model.train()
if cfg['pretrained_path']:
    model.load_state_dict(torch.load(cfg['pretrained_path']))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

train_dataset = KoSummarizationDataset(tokenizer, 512, "train", 1500)
val_dataset = KoSummarizationDataset(tokenizer, 512, "val", 100)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg['batch_size'], shuffle=False)
test_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg['batch_size'], shuffle=False)

optimizer = AdamW(optimizer_grouped_parameters,
                  lr = cfg['lr'], # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

total_steps = (len(train_dataloader.dataset) * cfg['K_epochs']) * cfg['epochs']

# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)
scheduler = None

columns=["epoch", "generated1", "generated2", "generated3"]
table_datum=[]

len_dl = len(train_dataloader)
len_ds = len(train_dataset)
ppo = PPO(cfg, model, old_model, tokenizer, optimizer, scheduler, cfg['K_epochs'])

rough_scorer = RougeScorer()

def evaluate_step(model, epoch, len_ds):
    model.eval()
    val_loss = 0.0
    rouge_score = {'rouge1':MyScore(0, 0, 0), 'rouge2':MyScore(0, 0, 0), 'rougeL':MyScore(0, 0, 0)}

    evaluation_data=[epoch, 'generated1', 'generated2', 'generated3']
    for (i, batch) in enumerate(tqdm(iter(test_dataloader))):
        B = cfg['batch_size']
        input_ids, label_ids, dec_input_ids = batch
        input_ids=input_ids.type(torch.LongTensor).to(device)
        temp_label_ids = copy.deepcopy(label_ids)
        label_ids=label_ids.type(torch.LongTensor).to(device)
        dec_input_ids=dec_input_ids.type(torch.LongTensor).to(device)
        
        attention_mask = input_ids.ne(pad_token_id).float()
        decoder_attention_mask = dec_input_ids.ne(pad_token_id).float()

        label_attention_mask = label_ids.ne(pad_token_id).float()
        label_ids[label_attention_mask == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=dec_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=label_ids,
            return_dict=True)

        val_loss += outputs.loss.item()*B

        # calculate rough score
        if i < cfg['num_to_rouge']:
            gen_ids = model.generate(input_ids,
                            max_length=512,
                            num_beams=5,
                            eos_token_id=tokenizer.eos_token_id
                            )
            
            generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            temp_label_ids = np.where(temp_label_ids < 0, 0, temp_label_ids)
            label = tokenizer.decode(temp_label_ids[0], skip_special_tokens=True)

            score = rough_scorer.score(generated, label)
            rouge_score['rouge1'].fmeasure += score['rouge1'].fmeasure * B
            rouge_score['rouge2'].fmeasure += score['rouge2'].fmeasure * B
            rouge_score['rougeL'].fmeasure += score['rougeL'].fmeasure * B
            ##
            if i < 3:
                if epoch == 0:
                    input_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    table_datum.append([i+100, input_sentence, label, ''])
                else:
                    input_sentence = ""
                    label =""
                
                evaluation_data[i+1] = generated

    table_datum.append(evaluation_data)
    result_table = wandb.Table(data=table_datum, columns=columns)
    wandb.log({"table" : result_table})
    
    wandb.log({
        "rouge1_fmeasure" : rouge_score['rouge1'].fmeasure / (cfg['num_to_rouge']*cfg['batch_size']),
        "rouge2_fmeasure" : rouge_score['rouge2'].fmeasure / (cfg['num_to_rouge']*cfg['batch_size']),
        "rougeL_fmeasure" : rouge_score['rougeL'].fmeasure / (cfg['num_to_rouge']*cfg['batch_size'])
    })

    wandb.log({"val_loss": val_loss/len_ds, "epoch": epoch})

#evaluate pre-trained model
evaluate_step(model, -1, len_ds)
for epoch in range(cfg['epochs']):
    model.train()
    total_reward = 0.0
    for (i, batch) in enumerate(tqdm(iter(train_dataloader))):
        input_ids, label_ids, dec_input_ids = batch
        input_ids=input_ids.type(torch.LongTensor).to(device)
        temp_label_ids = copy.deepcopy(label_ids)
        label_ids=label_ids.type(torch.LongTensor).to(device)
        
        dec_input_ids=dec_input_ids.type(torch.LongTensor).to(device)
        
        attention_mask = input_ids.ne(pad_token_id).float()
        decoder_attention_mask = dec_input_ids.ne(pad_token_id).float()

        label_attention_mask = label_ids.ne(-100).float()
        label_ids[label_attention_mask == 0] = -100

        ppo.save_one_episode(input_ids, attention_mask, label_ids)

        reward = ppo.buffer.rewards[0]
        total_reward += reward
        wandb.log({"reward": reward})
        ppo.update()

    wandb.log({"total_train_reward" : total_reward / (len_dl)})
    torch.save(model.state_dict(), CHPT_PATH)
    evaluate_step(model, epoch, len_ds)
    # print('[Epoch: {:>4}] test loss = {:>.9}'.format(epoch + 1, loss))