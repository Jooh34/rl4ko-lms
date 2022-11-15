import os, sys, json
import torch
from torch.utils.data import Dataset
import numpy as np

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '../'))

DATA_MAX = None
# DATA_MAX = 100

class KoSummarizationDataset(Dataset):
    def __init__(self, tokenizer, max_len, ignore_index=-100):
        file_path = "./data/ko_summarization/news_train_original.json"
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.tokenizer = tokenizer
        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index
        self.max_len = max_len

        with open(file_path, 'r', encoding='UTF-8') as file_0:
            news = json.load(file_0)
            data = news['documents']
            for (ix, d) in enumerate(data):
                sentences = ""
                sentences += (d['title'] + " ")

                for t_list in d['text']:
                    for t in t_list:
                        sentences += (t['sentence'] + " ")
                
                #self.x_data.append(sentences)
                label = d['abstractive'][0]

                input_ids = self.tokenizer.encode(sentences)
                # input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
                input_ids = self.add_padding_data(input_ids)

                label_ids = self.tokenizer.encode(label)
                label_ids.append(self.tokenizer.eos_token_id)

                dec_input_ids = [self.tokenizer.eos_token_id]
                dec_input_ids += label_ids[:-1]
                dec_input_ids = self.add_padding_data(dec_input_ids)
                label_ids = self.add_ignored_data(label_ids)

                # label_attention_mask = label_ids.ne(self.pad_token_id).float()
                # label_ids[label_attention_mask == 0] = -100

                self.x_data.append(np.array(input_ids, dtype=np.int_))
                self.y_data.append(np.array(label_ids, dtype=np.int_))
                self.z_data.append(np.array(dec_input_ids, dtype=np.int_))

                if DATA_MAX and DATA_MAX < ix:
                    break

        print(f'KoSummarization : {len(self.y_data)} data initialized.')
    
    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.z_data[idx]