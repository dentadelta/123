# Image Captioning

This image captioning was finetuned on XXX database. Using this model, the user can quickly generate a caption for an image. The model was trained on YYY database, so the user can test out the model by downloading asset data from other database from other places

## Installation

In order to run the image captioning model, the following python libraries will need to be downloaded and installed on the user computers

If CUDA graphiccard is not available:

```bash
pip install --proxy={username}@{pac} gradio, pip3 install torch torchvision torchaudio transformers notebook
```

where 
   {username} = company login username and 
   {pac}      = company proxy


## Usage
Open a terminal from the Window Search bar, and type the followings:


```bash
cd "{folder location}"
```
Where "{folder location} is this folder location". For example: cd "C:\\aaa\bbb\ccc\thisfolder". Make sure to start with a double quote and end with a double quote.

```bash
python imagecaptioning.py
```

Wait for a few minutes, until a link is printed out from the terminal. Copy and paste that link to to the web browser to use the application. Don't worry, it is a localhost weblink that runs on your work computer (not through the internet)

## Next
Making walk through videos.
