import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import  TrainingArguments, Trainer, AutoTokenizer, LongT5ForConditionalGeneration, AutoConfig
import gc
import os
torch.manual_seed(42)
import time

#Well it does work. Let the model seen work example once (one epoch).

#BUT, the model took the shortcut (blaming the dataset) .. I would do the same

#Lets train for four epoches and see what happens.

#I'm Free but please pay for my electricity or my landlord wont be happy...Takes 5.5 hours to train for 4 epoches...

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base") # good enough
model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base").cuda()

class CustomDataset(Dataset):
    def __init__(self,csvpath, tokenizer):
        df = pd.read_csv(csvpath)
        inputs = df['inputs'].tolist()
        targets = df['targets'].tolist()
        inputs_max_length = 512*2          # Good enough
        targets_max_length = 512
        self.encoder_ids = []
        self.labels = []
        self.encoder_attention_masks = []
        self.decoder_attention_masks = []
        for i in range(len(inputs)):
            e = inputs[i]
            prefix = e.split('\n')[0]
            e = e[len(prefix):]
            d = targets[i]
            encoder_dict = tokenizer(f'<{prefix}>{e}</s>', truncation=True, max_length=inputs_max_length, padding="max_length")
            labels_dict = tokenizer(f'<>{d}</s>', truncation=True, max_length=targets_max_length, padding="max_length")
            self.encoder_ids.append(torch.tensor(encoder_dict['input_ids']))
            self.encoder_attention_masks.append(torch.tensor(encoder_dict['attention_mask']))
            labels = torch.tensor(labels_dict['input_ids'])
            labels[labels == tokenizer.pad_token_id] = -100
            if len(labels) <= 10:
                print(labels)
            self.labels.append(labels)
            self.decoder_attention_masks.append(torch.tensor(labels_dict['attention_mask']))
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        return self.encoder_ids[idx],self.labels[idx]
dataset = CustomDataset(csvpath='/media/delta/S/training_data.csv',tokenizer=tokenizer)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

gc.collect()
torch.cuda.empty_cache()


training_args = TrainingArguments(output_dir='/media/delta/S/results', 
                                  num_train_epochs=4, 
                                  logging_steps=10,
                                  save_steps=10000,
                                  per_device_train_batch_size=2, 
                                  per_device_eval_batch_size=1,
                                  gradient_accumulation_steps=3,
                                  gradient_checkpointing=True,
                                  fp16=False,  #doesnt work for this model
                                  optim="adafactor", #change to adamW if you have have enough memory
                                  warmup_steps=1, 
                                  weight_decay=0.05, 
                                  logging_dir='/home/delta/Downloads/logs', 
                                  report_to = 'tensorboard',
                                  )
                                

Trainer(model=model,  args=training_args, train_dataset=train_dataset, 
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                            'labels': torch.stack([f[1] for f in data])
                                                            }).train()

model.save_pretrained('/media/delta/S/model')
tokenizer.save_pretrained('/media/delta/S/tokenizer')
