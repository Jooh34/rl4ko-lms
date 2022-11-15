import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))

from common.dataset import KoSummarizationDataset


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

_ = model.eval()

kosum_dataset = KoSummarizationDataset()


prompt = '안녕하세요 저는'
with torch.no_grad():
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(
        device='cuda', non_blocking=True)
    gen_tokens = model.generate(
        tokens, do_sample=True, temperature=0.8, max_length=64)
    generated = tokenizer.batch_decode(gen_tokens)[0]


print(generated)
