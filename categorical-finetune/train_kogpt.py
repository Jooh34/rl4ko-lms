import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))

from common.dataset import KoSummarizationDataset

cfg = {
    'batch_size': 1,
    'accumulation_step' : 32,
    'epochs': 20,
    'lr': 2e-5,
}
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(
    # or float32 version: revision=KoGPT6B-ryan1.5b
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)

model = AutoModelForCausalLM.from_pretrained(
    # or float32 version: revision=KoGPT6B-ryan1.5b
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype='auto', low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)

pad_token_id = tokenizer.pad_token_id
model.train()

kosum_dataset = KoSummarizationDataset(tokenizer, 512, "train")

train_size = int(0.8 * len(kosum_dataset))
test_size = len(kosum_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    kosum_dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg['batch_size'], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg['batch_size'], shuffle=True)
    
optimizer = AdamW(model.parameters(),
                  lr = cfg['lr'], # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

total_steps = len(train_dataloader.dataset) * cfg['epochs']

scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# wandb.init(project='project-name')
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