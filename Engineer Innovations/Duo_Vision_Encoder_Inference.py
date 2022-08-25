#!/usr/bin/env python
# coding: utf-8

# In[103]:


from transformers import VisionTextDualEncoderModel,VisionTextDualEncoderProcessor,ViTFeatureExtractor,BertTokenizer
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from pathlib import Path
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np


# In[4]:


get_ipython().run_cell_magic('capture', '', 'device = \'cuda\' if torch.cuda.is_available() else \'cpu\'\nif Path(\'duo_constrated_tokenizer\').exists() and Path(\'duo_constrated_model\').exists() and Path(\'duo_constrated_feature_extractor\').exists():\n    tokenizer = BertTokenizer.from_pretrained(\'duo_constrated_tokenizer\')\n    feature_extractor = ViTFeatureExtractor.from_pretrained(\'duo_constrated_feature_extractor\')\n    processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)\n    model = VisionTextDualEncoderModel.from_pretrained(\'duo_constrated_model\')\nelse:\n    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")\n    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")\n    processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)\n    model = VisionTextDualEncoderModel.from_vision_text_pretrained("google/vit-base-patch16-224", "bert-base-uncased")\n\nmodel.to(device)\nmodel.eval()\n')


# In[5]:


training_data = pd.read_excel(r'/media/delta/S/trainig_data.xlsx')


# In[7]:


class ImageData(Dataset):
    def __init__(self,ds, feature_extractor):
        self.images = ds['path to image'].values.tolist()
        self.captions = ds['image caption'].values.tolist()
        self.pixel_values = []
        for i in self.images:
            image = Image.open(i).convert('RGB')
            image = feature_extractor(image, return_tensors='pt').pixel_values.to(device)
            self.pixel_values.append(image)

    def __len__(self):
        return len(self.pixel_values)

    def __getitem__(self, idx):
        return  self.pixel_values[idx], self.images[idx], self.captions[idx]


# In[8]:


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    images = [example[1]for example in examples]
    captions = [example[2]for example in examples]
    return {"pixel_values": pixel_values,'image_path':images, 'captions':captions}


# In[9]:


asset_data = DataLoader(ImageData(training_data,feature_extractor),
                        batch_size=64, shuffle=False, collate_fn=collate_fn)


# In[10]:


gc.collect()
torch.cuda.empty_cache()


# In[21]:


captions = 'guardrail'   #Search an image using your own caption
scores = []
image_paths = []
caption = []
for _, batch in enumerate(asset_data):
    pixel_values = batch['pixel_values'].view(-1,3,224,224)
    image_path = batch['image_path']
    text = batch['captions']
    inputs = tokenizer(captions,padding='max_length',return_tensors='pt',max_length=128)
    inputs['pixel_values'] = pixel_values
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    logits_per_image = outputs.logits_per_image.view(-1)
    ids = logits_per_image.argmax().unsqueeze(0).item()
    probability = logits_per_image[ids].item()
    image_path = image_path[ids]
    text = text[ids]
    image_paths.append(image_path)
    scores.append(probability)
    caption.append(text)


# In[52]:


ids = torch.topk(torch.tensor(scores),len(scores)).indices
scores = [scores[i] for i in ids]
image_paths  = [image_paths[i] for i in ids]


# In[ ]:


writer = SummaryWriter()
IMG = [Image.open(i) for i in image_paths]
font = ImageFont.load_default()
IMG = [ImageOps.fit(image,(512,512)) for image in IMG]
IMG = [np.array(image) for image in IMG]
IMG = np.array(IMG)
writer.add_images('image_score_ranked_by_probability',IMG,dataformats='NHWC')
writer.close()


# In[ ]:


#go to terminal,cd to the same folder: tensorboard --logdir runs

