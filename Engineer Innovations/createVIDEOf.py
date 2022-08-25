import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from ast import literal_eval
import numpy as np
from IPython.display import Video
import cv2
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings('ignore')
import glob
from pathlib import Path
import click

processor = OwlViTProcessor.from_pretrained("processor")                        #"google/owlvit-base-patch32"
model = OwlViTForObjectDetection.from_pretrained("model").to(device)		#"google/owlvit-base-patch32"

df = pd.DataFrame(columns=['image_path','box','score','label'])
df['box'] = df['box'].astype(object)
@click.command()
@click.argument('image_path')
@click.argument('captions')
@click.argument('start_distance')
@click.argument('end_distance')
def createVideo(image_path,captions,start_distance,end_distance):
    try:
        start_distance = float(start_distance)
        end_distance = float(end_distance)
        k = 0
        captions = captions.split(',')
        captions = [c.lstrip() for c in captions]
        if type(captions) == str:
            captions = [captions]
        list_of_image1 = glob.glob(f'{image_path}/*.png')
        list_of_image2 = glob.glob(f'{image_path}/*.jpg')
        list_of_images = list_of_image1 + list_of_image2
        list_of_images = [Path(i) for i in list_of_images]
        list_of_images.sort(key=lambda x: x.name)
        list_of_images = list(filter(lambda x: float(x.name.split('_')[-2]) >= start_distance and float(x.name.split('_')[-2]) <= end_distance,list_of_images))   # ..distance_1.0_km.png
        images = [Image.open(l).convert('RGB') for l in list_of_images]
        texts = captions
        color_dict = {}
        for i in range(len(texts)):
            color_dict[i] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for i in range(len(images)):
            image = images[i]
            inputs = processor(text=texts, images=image, return_tensors="pt")
            outputs = model(**inputs.to(device)) 
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
            for box, score, label in zip(boxes, scores, labels):
                box = [int(i) for i in box.tolist()]
                df.at[k,'image_path'] = list_of_images[i]
                df.at[k,'box'] = box
                df.at[k,'score'] = score.item()
                df.at[k,'label'] = captions[label]
                k += 1
        df.to_csv('dvr.csv',index=False)
        print('Memory used', round(int(torch.cuda.memory_allocated())/(1024*1024*1024),3),'GB')
        print('done')
    except Exception as e:
        print(e)
if __name__ == '__main__':
    createVideo()
