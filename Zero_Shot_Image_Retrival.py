import clip
import torch
import torchvision.transforms as T
from PIL import Image
import click
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import glob
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px')
model.to(device)

@click.command()
def zero_shot_image_retrival():
    image_path = click.prompt('Enter image path, maximum 1000 images')
    captions = click.prompt('Please enter a caption to retrieve relevant images')
    how_many = click.prompt('How many images do you want to retrieve?', type=int)
    image_path = image_path.replace("'", '')
    list_of_image_1 = glob.glob(str(Path(image_path,'*.png')))     
    list_of_image_2 = glob.glob(str(Path(image_path,'*.jpg')))     
    list_of_images = list_of_image_1 + list_of_image_2
    original_images = [Image.open(image) for image in list_of_images]
    images = [preprocess(image).unsqueeze(0) for image in original_images]
    for i in range(min(1000,len(images))):   
        with torch.no_grad():
            images[i]= model.encode_image(images[i].reshape(1,3,336,336).to(device)).float()
            torch.cuda.empty_cache()
    images = torch.stack(images)
    images = images / images.norm(dim = -1, keepdim = True)
    text_vec = model.encode_text(clip.tokenize(captions).to(device)).float()
    text_vec = text_vec / text_vec.norm(dim = 0, keepdim = True)
    images = images.view(images.shape[0], -1)
    text_vec = text_vec @ images.T
    ids = torch.topk(text_vec, how_many).indices
    ids = ids.cpu().numpy()[0]
    retrieved_images = [original_images[i] for i in ids]
    for image in retrieved_images:
        image.show()
    
if __name__ == '__main__':
    zero_shot_image_retrival()

# Usage:
# python Zero_Shot_Image_Retrival.py
# Enter image path, maximum 1000 images: './images'
# Please enter a caption to retrieve relevant images: 'photo of an intersection'
# How many images do you want to retrieve? 1
 
