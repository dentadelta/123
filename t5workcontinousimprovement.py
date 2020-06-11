import pyarrow as pa
import pandas as pd
import os
import torch
from transformers import (T5Config, T5Tokenizer, T5ForConditionalGeneration, TextDataset, DataCollator, Trainer, TrainingArguments)
import ipywidgets as widgets
import random
from typing import Dict, List
import nlp
from dataclasses import dataclass
from tqdm.auto import tqdm
import re
import pathlib
import numpy as np
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import textwrap
import argparse
@dataclass
class T2TDataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'lm_labels': lm_labels, 
            'decoder_attention_mask': decoder_attention_mask
        }

class Custom_T5_Training():
  def __init__(self, dataset_path,working_folder, maximum_input_length, maximum_output_length, epochs=1, logging_step=1000, model_name = 't5-base'):
    self.fields = [('input_text', pa.string()),('target_text', pa.string()),('prefix', pa.string())]
    self.dataset_path = dataset_path
    self.working_folder = working_folder
    self.model_name = model_name
    self.maximum_input_length = maximum_input_length
    self.maximum_output_length = maximum_output_length
    self.epochs = epochs
    self.create_dataset()
    self.load_model()
    self.data_collator = T2TDataCollator()
    self.training_args = TrainingArguments(
                        output_dir= self.working_folder,
                        overwrite_output_dir=True,
                        do_train=True,
                        do_eval =False,
                        num_train_epochs=self.epochs,   
                        per_device_train_batch_size=                1, 
                        logging_steps=                              logging_step,   
                        save_steps=                                 -1,
                        )

    self.progress = widgets.FloatProgress(value=0.1, min=0.0, max=1.0, bar_style = 'info')


  def create_dataset(self):
    self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
    df = pd.read_csv(self.dataset_path)
    train_dataset = self.createdatafromcsv(df)
    self.train_dataset = train_dataset
    self.tokenizer.save_pretrained(self.working_folder)

  def add_eos_to_examples(self,example):
    example['input_text'] = '<{}>: <{}> </s>'.format(example['prefix'] , example['input_text'] )
    example['target_text'] = '"<{}> </s>"'.format(example['target_text'])
    return example

  def convert_to_features(self,example_batch):
    input_encodings = self.tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=self.maximum_input_length)     ########## Specify the maximum input lengths (context + question)
    target_encodings = self.tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=self.maximum_output_length)     ########## Specify the maximum output length
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }
    return encodings

  def load_model(self):
    file = pathlib.Path('{}/pytorch_model.bin'.format(self.working_folder))
    if file.exists():
      self.model = T5ForConditionalGeneration.from_pretrained(self.working_folder)
    
    else:
      config = T5Config.from_pretrained(self.model_name)
      self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, config=config)
      self.model.save_pretrained(self.working_folder)

  def train_model(self):
    trainer = Trainer(
                        model= self.model,
                        args=self.training_args,
                        data_collator=self.data_collator,
                        train_dataset=self.train_dataset,
                        prediction_loss_only=True
                      )
    self.progress.value = 0.4
    p_start, p_end = 0.4, 1.
    def progressify(f):
      def inner(*args, **kwargs):
        if trainer.epoch is not None:
          self.progress.value = p_start + trainer.epoch / self.epochs * (p_end - p_start)
          return f(*args, **kwargs)
      return inner
    try:
      trainer._training_step = progressify(trainer._training_step)
      trainer.train()
    
    except KeyboardInterrupt:
      print('Keyboard interrupted, but dont worry because...')
    finally:
      trainer.save_model(self.working_folder)
      print('the model has been saved')

  def createdatafromcsv(self, da):
      d = pd.DataFrame(columns=['prefix','input_text','target_text'])
      j = 0
      for i,r in da.iterrows():
        input_text = r['input_text']
        prefix = r['prefix']
        d.loc[j,'prefix'] = prefix
        d.loc[j,'input_text'] = input_text
        try:
          d.loc[j,'target_text'] = r['target_text'].lower()
        except:
          d.loc[j,'target_text'] = ''
        j += 1

      work_dataset = nlp.arrow_dataset.Dataset(pa.Table.from_pandas(d,pa.schema(self.fields)))
      work_dataset = work_dataset.map(self.add_eos_to_examples)
      work_dataset = work_dataset.map(self.convert_to_features, batched=True)
      columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
      work_dataset.set_format(type='torch', columns=columns)
      return work_dataset

  def Knowledge_Update(self):
    number_of_rows = len(self.train_dataset)
    dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=number_of_rows)
    evaluate_model = T5ForConditionalGeneration.from_pretrained(self.working_folder).to(device)
    answers = [] 
    for batch in tqdm(dataloader):
        outs = evaluate_model.generate(input_ids=batch['input_ids'].to(device), 
                        attention_mask=batch['attention_mask'].to(device),
                        early_stopping=True)
        outs = [self.tokenizer.decode(ids) for ids in outs]
        answers.extend(outs)

    predictions = []
    input_texts = []
    for ref, pred in zip(self.train_dataset, answers):
      pred = pred[4:-1]
      predictions.append(pred)
        
      input_ = self.tokenizer.decode(ref['input_ids'])
      input_ = ''.join(input_)
      input_ = re.sub('[!@#$*-]', '', input_)
      input_ = input_.lstrip().title()

      start_index = input_.index('>:')
      prefix = input_[:start_index]
      input_ = input_[start_index + 6:-1]

      input_texts.append(input_)

    results = {'prefix': prefix[2:], 'input_text': input_texts, 'target_text': predictions}
    ds = pd.DataFrame(results)
    return ds

def Step1(My_T5):
    df = My_T5.Knowledge_Update()
    text:[str] = []
    for i,r in df.iterrows():
      prefix = r['prefix']
      input_text = r['input_text'].lstrip()
      target_text = r['target_text'].lstrip()
      concat_text = f'prefix:{prefix}\ninput_text:{input_text}\ntarget_text:{target_text}\n'
      text.append(concat_text)
      text_:str = '\n'.join(text)
    with open('UpdateKnowledge.txt','w') as file:
      file.write(text_)
      file.close()

def Step2(My_T5,IMPROVEMENT):
    with open('UpdateKnowledge.txt') as file:
      f = file.readlines()
      j = 0
      d = pd.read_csv(My_T5.dataset_path)
      d['target_text'] = ''
      for i in f:
        if len(i) == 1:
          j += 1
        if i[:7] == 'target_':
          target_t = i[12:]
          d.loc[j,'target_text'] = target_t.rstrip('\n')
      d.to_csv(My_T5.dataset_path, index= None)
      file = pathlib.Path(f'{IMPROVEMENT}/accumulatedknowledge.csv')
      if file.exists():
        dc = pd.read_csv(f'{IMPROVEMENT}/accumulatedknowledge.csv')
        dc = pd.concat([dc, d])
      else:
        dc = d
      dc.to_csv(f'{IMPROVEMENT}/accumulatedknowledge.csv', index= None)
      print('New knowledge saved')

def Step3(My_T5):
    My_T5.train_model()


def Step4(My_T5):
    dd =My_T5.Knowledge_Update()
    text = []
    for i,r in dd.iterrows():
      prefix = r['prefix']
      input_text = r['input_text'].lstrip()
      target_text = r['target_text'].lstrip()
      concat_text = f'prefix:{prefix}\ninput_text:{input_text}\ntarget_text:{target_text}\n'
      text.append(concat_text)
    text = '\n'.join(text)
    with open('ConfirmingKnowledgeUpdate.txt','w') as file:
      file.write(text)
      file.close()
    
def Step5(My_T5):
    da = pd.DataFrame(columns=['prefix','input_text', 'target_text'])
    with open('test_file.txt') as file:
      j = 0
      textline = file.readlines()
      for f in textline:
        print(f)
        if f.find('prefix') > -1:
          da.at[j,'prefix'] = f[len('prefix:'):]
        if f.find('input_text') > -1:
          da.at[j,'input_text'] = f[len('input_text:'):]        
          j += 1
      da.at[j,'target_text'] = '.'
    da = da.dropna()
    print(da)

   



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traininglibrary', metavar='T', type=str, nargs=1,help='specify improvement file path')
    parser.add_argument('workingfolder', metavar='W', type=str, nargs=1,help='specify training file path')   
    parser.add_argument('filepath', metavar='F', type=str, nargs=1,help='specify new file path')
   
    parser.add_argument('steps', metavar='S',type=int,nargs=1, help="Specify training step")
    parser.add_argument('epochs', metavar='E',type=int,nargs=1, help="Specify training epoch")
    parser.add_argument('command', metavar='N', type=str, nargs=1,help='Send in your command')

    args = parser.parse_args()
    
    IMPROVEMENT = args.traininglibrary[0]
    WORKING_FOLDER = args.workingfolder[0]
    DATAPATH = args.filepath[0]
    command = args.command[0]
    epochs = args.epochs[0]
    steps = args.steps[0]

    My_T5 = Custom_T5_Training(
          dataset_path= f'{IMPROVEMENT}/accumulatedknowledge.csv' if command == 'Training' else DATAPATH,
          working_folder= WORKING_FOLDER,
          maximum_input_length=250,
          maximum_output_length= 100,
          model_name= 't5-base',
          logging_step = steps,
          epochs = epochs)

    print(command)
        

    if command == 'KnowledgeUpdate':
        Step1(My_T5)
        with open('test_file.txt','w') as file:
          file.write('prefix:\ninput_text:')
          file.close()
    
    if command == 'KnowledgeUpgrade':
        Step2(My_T5,IMPROVEMENT)
    
    if command == 'Training':
        print(My_T5.dataset_path)
        Step3(My_T5)
        
    if command == 'Confirmation':
        Step4(My_T5)
    
    if command == 'Test':
        Step5(My_T5)
        
if __name__ == "__main__":
    main()





