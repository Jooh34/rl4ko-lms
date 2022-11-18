import os
import sys
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
from tqdm import tqdm
from datetime import datetime

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))
from common.dataset import KoSummarizationDataset

CHPT_PATH = './checkpoints/kobart-{}.pt'.format("2022-11-16-19-33-35")

device = 'cuda'

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
pad_token_id = tokenizer.pad_token_id
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2').to(device)
model.eval()
model.load_state_dict(torch.load(CHPT_PATH))

kosum_dataset = KoSummarizationDataset(tokenizer, 512, "test")

dataloader = torch.utils.data.DataLoader(
    kosum_dataset, batch_size=1, shuffle=False)

for (i, batch) in enumerate(iter(dataloader)):
    input_ids, _, label_ids = batch
    input_ids=input_ids.type(torch.LongTensor).to(device)
    label_ids=label_ids.type(torch.LongTensor).to(device)
    
    generated_ids = model.generate(input_ids)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    label = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    print(output)
    print(label)
