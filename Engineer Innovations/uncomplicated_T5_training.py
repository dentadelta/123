import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.utils import shuffle
import re
import time


def main():
    d = torch.device('cuda') #cuda
    print('loading model')
    evaluate_model = T5ForConditionalGeneration.from_pretrained('t5-base') 
    evaluate_model = evaluate_model.to(d)
    print('model loaded')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    print('tokenizer loaded')
    time.sleep(10)
    evaluate_model.train()	
    optimizer = torch.optim.Adam(evaluate_model.parameters(), lr=6e-5)
    print('optimizer loaded')
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    df = pd.read_csv('/home/dentadelta/Downloads/train_dataset.csv') #Where the file is saved
    print('data loaded')
    global_iterator = 0
    for epoch in range(5):
        if epoch != 0:
            scheduler.step()
        df = shuffle(df)
        iter_loss = 0
        j = 0
        for i,r in df.iterrows():
            prefix = r.prefix.lower()
            input_text = r.input_text.lstrip()
            input_text = re.sub(r'  ',' ',input_text)
            input_text = input_text.lower()
            input_text = '<{}>: <{}> </s>'.format(prefix ,input_text)
            input_encodings = tokenizer.encode_plus(input_text, pad_to_max_length=True, max_length=500)
            input_ids = input_encodings['input_ids']
            input_ids = torch.LongTensor(input_ids).view(1,-1)	
            attention_mask = input_encodings['attention_mask']
            attention_mask = torch.LongTensor(attention_mask).view(1,-1)	
            target_text = f'<{r.target_text.lower()}> </s>'
            target_encoding = tokenizer.encode_plus(target_text, pad_to_max_length=True, max_length=110)	
            lm_labels = target_encoding['input_ids']
            lm_labels = torch.LongTensor(lm_labels).view(1,-1)
            lm_labels[lm_labels[:, :] == 0] = -100	
            decoder_attention_mask = target_encoding['attention_mask']
            decoder_attention_mask = torch.LongTensor(decoder_attention_mask).view(1,-1)	
            input_ids = input_ids.to(d)
            attention_mask = attention_mask.to(d)
            lm_labels = lm_labels.to(d)
            decoder_attention_mask = decoder_attention_mask.to(d)	
            inputs = {'input_ids':input_ids,'attention_mask':attention_mask,'lm_labels': lm_labels,'decoder_attention_mask':decoder_attention_mask}
            outputs = evaluate_model(**inputs)
            del input_ids, attention_mask,lm_labels,decoder_attention_mask, inputs
            loss = outputs[0]
            del outputs
            loss_val = loss.item()
            loss_val = round(loss_val,3)
            iter_loss += loss_val
            if global_iterator % 100 == 0:
                average_loss = iter_loss/100
                iter_loss = 0
                print('average_loss: ',average_loss, 'epoch', epoch, 'iter',j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_iterator += 1
            j += 1

    evaluate_model.save_pretrained('/home/dentadelta/Downloads/') #Where the model is saved

if __name__ == '__main__':
    main()        
