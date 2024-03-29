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
from nltk.translate.bleu_score import sentence_bleu

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))
from common.dataset import KoSummarizationDataset
from common.rouge import RougeScorer, MyScore

time_format = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
CHPT_PATH = './checkpoints/kobart-{}.pt'.format(time_format)

wandb.init(project='rl4kolms-kobart')
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
wandb.config = cfg

torch.manual_seed(cfg['seed'])
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
pad_token_id = tokenizer.pad_token_id
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2').to(device)
model.train()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
#optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg['lr'], weight_decay=0.01)

train_dataset = KoSummarizationDataset(tokenizer, 512, "train")
test_dataset = KoSummarizationDataset(tokenizer, 512, "val", 100)

# train_size = int(0.8 * len(kosum_dataset))
# test_size = len(kosum_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(
#     kosum_dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg['batch_size'], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False)

optimizer = AdamW(optimizer_grouped_parameters,
                  lr = cfg['lr'], # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

t_total = (len(train_dataloader) / cfg["accumulate_grad_batches"]) * cfg['epochs']
warmup_step = int(t_total * cfg["warmup_ratio"])
print(t_total, warmup_step)

# num_warmup_steps = (len(train_dataloader.dataset) // ((cfg['batch_size']*cfg['accumulate_grad_batches']))) * 1
# total_steps = (len(train_dataloader.dataset) // ((cfg['batch_size']*cfg['accumulate_grad_batches']))) * cfg['epochs']

scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_step,
                                            num_training_steps = t_total)


columns=["epoch", "generated1", "generated2", "generated3"]
table_datum=[]

rough_scorer = RougeScorer()
len_dl = len(train_dataloader)

for epoch in range(cfg['epochs']):
    model.train()
    for (i, batch) in enumerate(tqdm(iter(train_dataloader))):
        input_ids, label_ids, dec_input_ids = batch
        input_ids=input_ids.type(torch.LongTensor).to(device)
        label_ids=label_ids.type(torch.LongTensor).to(device)
        dec_input_ids=dec_input_ids.type(torch.LongTensor).to(device)
        
        attention_mask = input_ids.ne(pad_token_id).float()
        decoder_attention_mask = dec_input_ids.ne(pad_token_id).float()

        label_attention_mask = label_ids.ne(-100).float()
        label_ids[label_attention_mask == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=dec_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=label_ids,
            return_dict=True)

        loss = outputs.loss / cfg['accumulate_grad_batches']
        loss.backward()

        if (i+1) % cfg['accumulate_grad_batches'] == 0 or i == len_dl-1:   # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            wandb.log({"loss_train": outputs.loss, "lr": optimizer.param_groups[0]["lr"]})
            # print('[Epoch: {:>4}] train loss = {:>.9}'.format(epoch + 1, loss))


    torch.save(model.state_dict(), CHPT_PATH)
    model.eval()
    val_loss = 0.0
    rouge_score = {'rouge1':MyScore(0, 0, 0), 'rouge2':MyScore(0, 0, 0), 'rougeL':MyScore(0, 0, 0), 'bleu':0}

    evaluation_data=[epoch, 'generated1', 'generated2', 'generated3']
    for (i, batch) in enumerate(tqdm(iter(test_dataloader))):
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

        val_loss += outputs.loss.item()*len(batch)

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
            bleu = sentence_bleu([label.split()], generated.split(), weights=(0.33,0.33,0.33,0))
            rouge_score['bleu'] += bleu * len(batch)
            rouge_score['rouge1'].fmeasure += score['rouge1'].fmeasure * len(batch)
            rouge_score['rouge2'].fmeasure += score['rouge2'].fmeasure * len(batch)
            rouge_score['rougeL'].fmeasure += score['rougeL'].fmeasure * len(batch)
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
        "bleu" : rouge_score['bleu'] / (cfg['num_to_rouge']*cfg['batch_size']),
        "rouge1_fmeasure" : rouge_score['rouge1'].fmeasure / (cfg['num_to_rouge']*cfg['batch_size']),
        "rouge2_fmeasure" : rouge_score['rouge2'].fmeasure / (cfg['num_to_rouge']*cfg['batch_size']),
        "rougeL_fmeasure" : rouge_score['rougeL'].fmeasure / (cfg['num_to_rouge']*cfg['batch_size'])
    })

    wandb.log({"val_loss": val_loss/len(test_dataloader.sampler), "epoch": epoch})
    print('[Epoch: {:>4}] test loss = {:>.9}'.format(epoch + 1, loss))
