import os
import sys
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
from tqdm import tqdm
from datetime import datetime
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))
from common.dataset import KoSummarizationDataset

time_format = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
CHPT_PATH = './checkpoints/kobart-{}.pt'.format(time_format)

cfg = {
    'batch_size': 4,
    'accumulation_step' : 16,
    'epochs': 20,
    'lr': 2e-5,
}
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
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg['lr'], weight_decay=0.01)

kosum_dataset = KoSummarizationDataset(tokenizer, 512, "train")

train_size = int(0.8 * len(kosum_dataset))
test_size = len(kosum_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    kosum_dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg['batch_size'], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg['batch_size'], shuffle=True)

optimizer = AdamW(optimizer_grouped_parameters,
                  lr = cfg['lr'], # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

total_steps = len(train_dataloader.dataset) * cfg['epochs']

scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

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

        loss = outputs.loss
        loss.backward()

        if (i+1) % cfg['accumulation_step'] == 0:             # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad()
            print('[Epoch: {:>4}] train loss = {:>.9}'.format(epoch + 1, loss))

        optimizer.zero_grad()
        scheduler.step()


    torch.save(model.state_dict(), CHPT_PATH)
    model.eval()
    for batch in tqdm(iter(test_dataloader)):
        input_ids, label_ids, dec_input_ids = batch
        input_ids=input_ids.type(torch.LongTensor).to(device)
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

        loss = outputs.loss
    
    print('[Epoch: {:>4}] test loss = {:>.9}'.format(epoch + 1, loss))
