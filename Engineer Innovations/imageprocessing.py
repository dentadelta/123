from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from typing import Tuple, List
from transformers import LayoutLMv3FeatureExtractor
from pathlib import Path
import json
from datetime import datetime

def copyandpastepdf(data,sourceimage):
    blankImage = Image.new("RGB", (1660, 2340), (255, 255, 255))
    data = data.split('\n')
    data = [d.split(' ') for d in data]
    cropImages = []
    cropLocation = []
    name = []
    counter = 0
    for b in data: 
        try:
            b = [int(a) for a in b]
            cropped = sourceimage.crop(tuple(b))
            cropImages.append(cropped)
            cropLocation.append(b)
            blankImage.paste(cropped,(b[0],b[1]))
            extracted_words = ' '.join(features(cropped)['words'][0])
            name.append(extracted_words)
            cropped.save(f'croppedData/{extracted_words}x{counter}.png')
            counter += 1
        except: pass  
    crop_dict = {}
    counter = 0
    for i,l,n in zip(cropImages,cropLocation,name):
        crop_dict[n] = {
            'box':l,
            'savefolder':f'croppedData/{n}x{counter}.png'
            }
        counter += 1
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    path = Path(f'croppedData/croppedData{dt_string}.json')
    with open(str(path),'w') as file:
        json.dump(crop_dict,file,indent=2)
        file.close()  
    print(str(path))
    return crop_dict,blankImage

features = LayoutLMv3FeatureExtractor()

class fontData(BaseModel):
    textType: str
    font_path: str
    font_size: int
    color: Tuple[int, int, int]


class textData(BaseModel):
    text: str
    box: Tuple[int, int,int,int]
    font_path: str
    font_size: int
    color: Tuple[int, int, int]


class insert_text_request(BaseModel):
    text: List[str]
    location: List[Tuple[int, int]]
    font_path: List[str]
    font_size: List[int]
    color: List[Tuple[int, int, int]]


def create_text_data(fontdata:fontData, box: Tuple[int,int,int,int], text:str):
    return textData(text=text, box=box, font_path=fontdata.font_path, font_size=fontdata.font_size, color=fontdata.color)

def insert_text(request: insert_text_request, image):
    draw = ImageDraw.Draw(image)
    for i in range(len(request.text)):
        font = ImageFont.truetype(request.font_path[i], request.font_size[i])
        draw.text(request.location[i], request.text[i], font=font, fill=request.color[i])
    return image

def create_image_from_jsonData(jsondata,image):
    request = insert_text_request(**jsondata)
    image = insert_text(request,image=image)
    return image

def get_text_location_from_font_text(font_path, font_size, text):
    font = ImageFont.truetype(font_path, font_size)
    width, height = font.getsize(text)
    return width, height


def split_draw_text_into_lines(text, font_path, font_size, width_box):
    lines = []
    line = ''
    text = text.split(' ')
    mintext = min([get_text_location_from_font_text(font_path, font_size,t)[0] for t in text])
    if width_box < mintext:
        print('Text is too long')
        return ['']
    for i in range(len(text)):
        width_text = get_text_location_from_font_text(font_path, font_size, line + text[i])[0]
        if width_text > width_box:
            lines.append(line)
            line = ''
        line += text[i] + ' '
    lines.append(line)
    return lines

def create_json_data_text(data: textData):
    width_box, height_box = data.box[2] - data.box[0], data.box[3] - data.box[1] 
    if width_box <= 0 or height_box <= 0:
        print('Box is not valid')
        return None
    texts = split_draw_text_into_lines(data.text, data.font_path, data.font_size, width_box)
    locations = []
    current_height = data.box[1]
    for i in range(len(texts)):
        _ , text_height = get_text_location_from_font_text(data.font_path, data.font_size, texts[i])
        if i != 0 :
            current_height += text_height
        else:
            current_height = 0
        if current_height < height_box:
            locations.append((data.box[0], current_height + data.box[1]))
        else:
            print(current_height,height_box)
            print('Check dimesions of box')
            return None
    font_path =  [data.font_path for i in range(len(texts))]
    font_size = [data.font_size for i in range(len(texts))]
    color = [data.color for i in range(len(texts))]
    return {'text': texts, 'location': locations, 'font_path': font_path, 'font_size': font_size, 'color': color}

def write_text_in_box(data: List[textData], image: Image):

    for i in range(len(data)):
        json_data = create_json_data_text(data[i])
        if json_data is not None:
            image = create_image_from_jsonData(json_data,image)
    return image

if __name__ == '__main__':

    image = Image.new("RGB", (1660, 2340), (255, 255, 255))
    textdata = textData(
        text='hello world from another country from another city',
        box=(0,0,500,500),
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        font_size=50,
        color=(0, 0, 0),
    )

    textdata2 = textData(
        text='Another Boxes Another Information',
        box=(500,500,501,900),
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        font_size=50,
        color=(0, 0, 0),
    )
    image = write_text_in_box([textdata,textdata2],image)
    draw = ImageDraw.Draw(image)
    draw.rectangle(textdata.box, outline='red')
    draw.rectangle(textdata2.box, outline='blue')
    image.save('test.png')
    
