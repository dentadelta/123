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
import pandas as pd
import gradio as gr
from PIL import Image

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_MODEL_FOLDER = '/media/delta/S/clipmodel_large.pth'


if Path(SAVED_MODEL_FOLDER).exists():
    model = clip.model.build_model(torch.load(SAVED_MODEL_FOLDER)).to(device)
    preprocess = clip.clip._transform(model.visual.input_resolution)

else:
    model, preprocess = clip.load('ViT-L/14@336px')
    model.to(device)
    

for p in model.parameters():
    p.require_grads = False

@click.group()
def cli():
    pass

@click.command()
def saveimageastensor():
    image_path = click.prompt('Enter image path')
    image_path = image_path.replace("'", '')
    image_path = image_path.replace('"', '')
    list_of_image_1 = glob.glob(str(Path(image_path,'*.png')))
    list_of_image_2 = glob.glob(str(Path(image_path,'*.jpg')))
    list_of_images = list_of_image_1 + list_of_image_2
    original_images = [Image.open(image) for image in list_of_images]
    images = [preprocess(image).unsqueeze(0) for image in original_images]
    images = [model.encode_image(image.to(device)).float() for image in images]
    images = torch.stack(images)
    images = images / images.norm(dim = -1, keepdim = True)

    torch.save(images, str(Path(image_path,'encoded_images.pt')))
    with open(str(Path(image_path,'original_image.txt')), 'w') as f:
        for image in list_of_images:
            f.write(str(Path(image).name) + '\n')
    print('Images saved')

@click.command()
def postusage():
    tensor_path = click.prompt('Enter tensor path')
    def retrieve_best_images(captions,tensor_path=tensor_path):
        text_vec = model.encode_text(clip.tokenize(captions).to(device)).float()
        tensor_path = tensor_path.replace("'", '')
        tensor_path = tensor_path.replace('"', '')
        images = torch.load(str(Path(tensor_path)), map_location=device)
        text_vec = text_vec / text_vec.norm(dim = 0, keepdim = True)
        images = images.view(images.shape[0], -1)
        text_vec = text_vec @ images.T
        ids = torch.topk(text_vec, 1).indices
        ids = ids.cpu().numpy()[0][0]
        folder_path = Path(tensor_path).parent
        with open(str(Path(folder_path,'original_image.txt')), 'r') as f:
            original_image_path = f.readlines()
        retrieved_image_path = original_image_path[ids]
        retrieved_image_path = retrieved_image_path.replace('\n', '')
        retrieved_image_path = Path(folder_path, retrieved_image_path)
        return Image.open(retrieved_image_path)

    gr.Interface(retrieve_best_images, 'text', 'image').launch()


if __name__ == '__main__':
    cli.add_command(saveimageastensor)
    cli.add_command(postusage)
    cli()
