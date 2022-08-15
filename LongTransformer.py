import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import  TrainingArguments, Trainer, AutoTokenizer, LongT5ForConditionalGeneration, AutoConfig
import gc
import os
torch.manual_seed(42)
import time


tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-large")
model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-large").cuda()

class CustomDataset(Dataset):
    def __init__(self,csvpath, tokenizer):
        df = pd.read_csv(csvpath)
        inputs = df['inputs'].tolist()
        targets = df['targets'].tolist()
        inputs_max_length = 512*3           # Input = 1500 words, output = 500 words  ---> Someone pays for my electricity......
        targets_max_length = 512
        self.encoder_ids = []
        self.labels = []
        self.encoder_attention_masks = []
        self.decoder_attention_masks = []
        for i in range(len(inputs)):
            e = inputs[i]
            d = targets[i]
            encoder_dict = tokenizer(f'predict: <{e}> </s>', truncation=True, max_length=inputs_max_length, padding="longest")
            labels_dict = tokenizer(f'<{d}> </s>', truncation=True, max_length=targets_max_length, padding="longest")
            self.encoder_ids.append(torch.tensor(encoder_dict['input_ids']))
            self.encoder_attention_masks.append(torch.tensor(encoder_dict['attention_mask']))
            labels = torch.tensor(labels_dict['input_ids'])
            labels[labels == 0] = -100
            self.labels.append(labels)
            self.decoder_attention_masks.append(torch.tensor(labels_dict['attention_mask']))
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        return self.encoder_ids[idx],self.encoder_attention_masks[idx], self.labels[idx], self.decoder_attention_masks[idx]

dataset = CustomDataset(csvpath='/media/delta/S/training_data.csv',tokenizer=tokenizer)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])


gc.collect()
torch.cuda.empty_cache()



training_args = TrainingArguments(output_dir='/media/delta/S/results', 
                                  num_train_epochs=10, 
                                  logging_steps=10,
                                  save_steps=10000,
                                  per_device_train_batch_size=1, 
                                  per_device_eval_batch_size=1,
                                  gradient_accumulation_steps=1,
                                  gradient_checkpointing=False,
                                  fp16=False,
                                  optim="adafactor",
                                  warmup_steps=1, 
                                  weight_decay=0.05, 
                                  logging_dir='/home/delta/Downloads/logs', 
                                  report_to = 'tensorboard',
                                  )
                                

Trainer(model=model,  args=training_args, train_dataset=train_dataset, 
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'decoder_attention_mask': torch.stack([f[3] for f in data]),
                                                              'decoder_input_ids': torch.stack([f[2] for f in data]),
                                                              'labels': torch.stack([f[2] for f in data])}).train()

model.save_pretrained('/media/delta/S/model')
tokenizer.save_pretrained('/media/delta/S/tokenizer')
