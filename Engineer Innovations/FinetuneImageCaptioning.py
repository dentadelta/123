import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split
from transformers import  TrainingArguments, Trainer, ViTFeatureExtractor, BertTokenizer, VisionEncoderDecoderModel
import torch
import gc
import os
torch.manual_seed(42)
from pathlib import Path

# I'm on Linux so you need to convert back to Windows

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = '/media/delta/S/Photos/Photo_Data'
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224-in21k", "bert-base-uncased").to(device)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

list_of_csv = glob.glob(f'{path}/*.csv')  # to change

DF = []
for f in list_of_csv:
    df = pd.read_csv(f)
    DF.append(df)
ds = pd.concat(DF)

class CustomDataset(Dataset):
    def __init__(self,ds, tokenizer,feature_extractor):
        self.Pixel_Values = []
        self.Labels = []
        for i,r in ds.iterrows():
            image_path = r['IMAGEPATH']             #A table in csv format with 2 columns IMAGEPATH and CAPTION
            labels = r['CAPTION']
            labels = str(labels)
            if len(image_path) >=10 and len(labels)>=10:
                image_path = image_path.split('\\')
                image_path = image_path[-3:]
                image_path = Path(os.getcwd(),image_path[0],image_path[1],image_path[2])
                image = Image.open(str(image_path)).convert("RGB")
                pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
                self.Pixel_Values.append(pixel_values)
                labels = tokenizer(labels,return_tensors="pt", truncation=True, max_length=128, padding="max_length").input_ids
                labels[labels == tokenizer.pad_token_id] = -100
                self.Labels.append(labels)
        
    def __len__(self):
        return len(self.Pixel_Values)

    def __getitem__(self, idx):
        return {"pixel_values": self.Pixel_Values[idx], "labels": self.Labels[idx]}

dataset = CustomDataset(ds,tokenizer,feature_extractor)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])


gc.collect()
torch.cuda.empty_cache()

training_args = TrainingArguments(output_dir=str(Path(os.getcwd(),'results')), 
                                  num_train_epochs=6, 
                                  logging_steps=300,
                                  save_steps=14770,
                                  per_device_train_batch_size=16, 
                                  per_device_eval_batch_size=16,
                                  gradient_accumulation_steps=1,
                                  gradient_checkpointing=False,
                                  fp16=False,  #doesnt work for this model
                                  optim="adamw_torch", #change to adamw_torch if you have have enough memory['adamw_hf', 'adamw_torch', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'sgd', 'adagrad']
                                  warmup_steps=1, 
                                  weight_decay=0.05, 
                                  logging_dir='/home/delta/Downloads/logs',  # loss graph
                                  report_to = 'tensorboard',
                                  )

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"][0] for example in examples])   #0 to change from [1,3,224,224] to  [3,224,224]  torch stack will add it back depends on the batch size,
    labels = torch.stack([example["labels"][0] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


Trainer(model=model,  args=training_args, train_dataset=train_dataset, 
        eval_dataset=val_dataset, data_collator=collate_fn).train()
        
model.save_pretrained('/media/delta/S/model_caption')
tokenizer.save_pretrained('/media/delta/S/tokenizer_caption')
feature_extractor.save_pretrained('/media/delta/S/feature_extractor_caption')
