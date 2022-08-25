import torch
from imagen_pytorch import Unet, Imagen
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)


unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 224),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()


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

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    images = [example[1]for example in examples]
    captions = [example[2]for example in examples]
    return {"pixel_values": pixel_values,'image_path':images, 'captions':captions}

training_data = pd.read_excel(r'/media/delta/S/trainig_data.xlsx')
training_data = training_data.dropna()
training_data = training_data.reset_index(drop=True)
training_data = training_data[:10]

dataset = ImageData(training_data, imagen.feature_extractor)