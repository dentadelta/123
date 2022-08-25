import pandas as pd
import glob
from pathlib import Path
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor, ViTFeatureExtractor, BertTokenizer
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torch
import gc
from pathlib import Path
import wandb
import click
from PIL import Image
import random

#https://arxiv.org/pdf/2111.07991.pdf
#huggingface always make it looks easy, but never works on new models if trained at scale
#outperform clip after finetune

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if Path('duo_constrated_tokenizer').exists() and Path('duo_constrated_model').exists() and Path('duo_constrated_feature_extractor').exists():
    tokenizer = BertTokenizer.from_pretrained('duo_constrated_tokenizer')
    feature_extractor = ViTFeatureExtractor.from_pretrained('duo_constrated_feature_extractor')
    processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
    model = VisionTextDualEncoderModel.from_pretrained('duo_constrated_model')
else:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
    model = VisionTextDualEncoderModel.from_vision_text_pretrained("google/vit-base-patch16-224", "bert-base-uncased")

model.to(device)

class CustomDataset(Dataset):
    def __init__(self,ds, processor):
        images = ds['path to image'].values.tolist()
        text = ds['image caption'].values.tolist()
        self.input_ids = []
        self.attention_mask = []
        self.pixel_values = []
        for i,t in zip(images,text):
            if len(i) >=10 and len(t)>=10:
                image = Image.open(i).convert('RGB')
                inp = processor(text=[t], images=[image], return_tensors="pt", truncation=True, max_length=128, padding="max_length")
                input_ids = inp["input_ids"]
                input_ids = input_ids.unsqueeze(0)
                attention_mask = inp["attention_mask"].squeeze(0)
                pixel_values = inp["pixel_values"].squeeze(0)
                self.input_ids.append(input_ids)
                self.attention_mask.append(attention_mask)
                self.pixel_values.append(pixel_values)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "pixel_values": self.pixel_values[idx]}


def collate_fn(examples):
    input_ids = torch.stack([example["input_ids"] for example in examples])  
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}

@click.group()
def cli():
    pass

@cli.command()
def finetune():
    training_data = click.prompt('Enter training data path')
    number_epochs = click.prompt('Enter number of epochs')
    number_epochs = int(number_epochs)
    training_data = training_data.replace('"','').replace("'",'').lstrip().rstrip()
    df = pd.read_excel(training_data)
    wandb.init(project="betterthanclip")
    dataset = CustomDataset(df,processor)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    gc.collect()
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    scaler = torch.cuda.amp.GradScaler()

    for _ in range(number_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            attention_mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].view(-1, attention_mask.shape[-1]).to(device)
            pixel_values = batch["pixel_values"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_loss=True)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        wandb.log({"Training Loss": total_loss / len(train_dataloader)})
        print(f"Training Loss: {total_loss / len(train_dataloader)}")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for _, batch in enumerate(val_dataloader):
                attention_mask = batch["attention_mask"].to(device)
                input_ids = batch["input_ids"].view(-1, attention_mask.shape[-1]).to(device)
                pixel_values = batch["pixel_values"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_loss=True)
                loss = outputs.loss
                total_loss += loss.item()
                print(f"{input_ids.shape} {attention_mask.shape} {pixel_values.shape}")

            wandb.log({"Validation Loss": total_loss / len(val_dataloader)})
            print("Validation Loss: {}".format(total_loss / len(val_dataloader)))

    model.save_pretrained('duo_constrated_model')
    tokenizer.save_pretrained('duo_constrated_tokenizer')
    feature_extractor.save_pretrained('duo_constrated_feature_extractor')

if __name__ == '__main__':
    cli.add_command(finetune)
    cli()
# Refer to Duo_Vision_Encoder_Inference.ipynb for inference
