import os
import sys
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
from tqdm import tqdm
from datetime import datetime
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import wandb
import random
import numpy as np
import copy

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))
from common.dataset import KoSummarizationDataset

time_format = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
CHPT_PATH = './checkpoints/skt-{}.pt'.format(time_format)

wandb.init(project='rl4kolms')
cfg = {
    'batch_size': 4,
    'accumulate_grad_batches' : 8,
    'epochs': 10,
    'lr': 5e-5,
    "warmup_ratio": 0.1,
    'checkpoint_path': CHPT_PATH,
    "seed": 3444
}
wandb.config = cfg

torch.manual_seed(cfg['seed'])
random.seed(cfg['seed'])
np.random.seed(cfg['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
pad_token_id = tokenizer.pad_token_id

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)
model.train()

kosum_dataset = KoSummarizationDataset(tokenizer, 512, "train")

train_size = int(0.8 * len(kosum_dataset))
test_size = len(kosum_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    kosum_dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg['batch_size'], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=int(cfg['batch_size']/2), shuffle=False)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [{
    'params': [
        p for n, p in param_optimizer
        if not any(nd in n for nd in no_decay)
    ],
    'weight_decay':
    0.01
}, {
    'params':
    [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    'weight_decay':
    0.0
}]
optimizer = AdamW(optimizer_grouped_parameters,
                    lr=cfg['lr'],
                    correct_bias=False)

# warm up lr
data_len = len(train_dataset)

num_train_steps = int(data_len /
                        (cfg['batch_size'] *
                        cfg['accumulate_grad_batches'] *
                        cfg['epochs']
                        ))
# logging.info(f'num_train_steps : {num_train_steps}')
num_warmup_steps = int(num_train_steps * cfg['warmup_ratio'])
# logging.info(f'num_warmup_steps : {num_warmup_steps}')
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_train_steps)
# lr_scheduler = {
#     'scheduler': scheduler,
#     'monitor': 'loss',
#     'interval': 'step',
#     'frequency': 1
# }

len_ds = len(train_dataset)
len_dl = len(train_dataloader)

columns=["epoch", "input", "label", "generated1"]
table_datum=[]

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
            # decoder_input_ids=dec_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            labels=label_ids,
            return_dict=True)

        loss = outputs.loss
        loss.backward()

        if (i+1) % (cfg['batch_size'] * cfg['accumulate_grad_batches']) == 0 or i == len_dl-1:   # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            wandb.log({"loss_train": loss})
            
    print('[Epoch: {:>4}] train loss = {:>.9}'.format(epoch + 1, loss))

    torch.save(model.state_dict(), CHPT_PATH)
    model.eval()
    val_loss = 0.0
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
            # decoder_input_ids=dec_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            labels=label_ids,
            return_dict=True)

        val_loss += outputs.loss.item()*len(batch)
        if i == 0:
            gen_ids = model.generate(input_ids,
                           max_length=512,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)

            generated1 = tokenizer.decode(gen_ids[0])
            if epoch == 0:
                input_sentence = tokenizer.decode(input_ids[0])
                temp_label_ids = np.where(temp_label_ids < 0, 0, temp_label_ids)
                label = tokenizer.decode(temp_label_ids[0])
            else:
                input_sentence = ""
                label =""

            my_data=[epoch, input_sentence, label, generated1]
            table_datum.append(my_data)
            result_table = wandb.Table(data=table_datum, columns=columns)
            wandb.log({"table" : result_table})
    
    wandb.log({"val_loss": val_loss/len(test_dataloader.sampler), "epoch": epoch})
    print('[Epoch: {:>4}] val loss = {:>.9}'.format(epoch + 1, val_loss/len(test_dataloader.sampler)))
