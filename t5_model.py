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
  def __init__(self, dataset_path,working_folder, maximum_input_length, maximum_output_length, model_name = 't5-base'):
    self.dataset_path = dataset_path
    self.working_folder = working_folder
    self.model_name = model_name
    self.maximum_input_length = maximum_input_length
    self.maximum_output_length = maximum_output_length
    self.load_data()
    self.load_model()

  def load_data(self):
    file = pathlib.Path('{}/train_data.pt'.format(self.working_folder))
    if file.exists():
      self.train_dataset = torch.load('{}/train_data.pt'.format(self.working_folder))
      self.valid_dataset = torch.load('{}/valid_data.pt'.format(self.working_folder))
      self.test_dataset =  torch.load('{}/test_data.pt'.format(self.working_folder))

      self.tokenizer = T5Tokenizer.from_pretrained(self.working_folder)
    else:
      self.create_dataset()

  def load_model(self):
    file = pathlib.Path('{}/pytorch_model.bin'.format(self.working_folder))
    if file.exists():
      self.model = T5ForConditionalGeneration.from_pretrained(self.working_folder)
    
    else:
      config = T5Config.from_pretrained(self.model_name)
      self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, config=config)

  def train_model(self, epochs=1):
    data_collator = T2TDataCollator()
    progress = widgets.FloatProgress(value=0.1, min=0.0, max=1.0, bar_style = 'info')
    training_args = TrainingArguments(
                        output_dir= self.working_folder,
                        overwrite_output_dir=True,
                        do_train=True,
                        do_eval =True,
                        num_train_epochs=epochs,   
                        per_device_train_batch_size=                2, 
                        logging_steps=                              1000,   
                        save_steps=                                 -1
                        )
    trainer = Trainer(
                        model= self.model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=self.train_dataset,
                        eval_dataset =self.test_dataset,
                        prediction_loss_only=True)
      

    progress.value = 0.4
    p_start, p_end = 0.4, 1.
    
    def progressify(f):
      def inner(*args, **kwargs):
        if trainer.epoch is not None:
          progress.value = p_start + trainer.epoch / epochs * (p_end - p_start)
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

  def create_dataset(self):
    self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
    df = pd.read_csv(self.dataset_path)
    df_train, df_valid, df_test= np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

    fields = [
          ('input_text', pa.string()),
          ('target_text', pa.string()),
          ('prefix', pa.string())
      ]


    train_dataset = nlp.arrow_dataset.Dataset(pa.Table.from_pandas(df_train,pa.schema(fields)))
    valid_dataset = nlp.arrow_dataset.Dataset(pa.Table.from_pandas(df_valid,pa.schema(fields)))
    test_dataset  = nlp.arrow_dataset.Dataset(pa.Table.from_pandas(df_test,pa.schema(fields)))

    train_dataset = train_dataset.map(self.add_eos_to_examples)
    train_dataset = train_dataset.map(self.convert_to_features, batched=True)

    valid_dataset = valid_dataset.map(self.add_eos_to_examples, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(self.convert_to_features, batched=True, load_from_cache_file=False)

    test_dataset = test_dataset.map(self.add_eos_to_examples, load_from_cache_file=False)
    test_dataset = test_dataset.map(self.convert_to_features, batched=True, load_from_cache_file=False)

    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)

    torch.save(train_dataset, '{}/train_data.pt'.format(self.working_folder))
    torch.save(valid_dataset, '{}/valid_data.pt'.format(self.working_folder))
    torch.save(test_dataset, '{}/test_data.pt'.format(self.working_folder))

    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset
    self.test_dataset = test_dataset
    self.tokenizer.save_pretrained(self.working_folder)


  def validate_model(self,length=100):
    dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=32)
    evaluate_model = T5ForConditionalGeneration.from_pretrained(self.working_folder).to(device)
    try:
      answers = []

      for batch in tqdm(dataloader):
        outs = evaluate_model.generate(input_ids=batch['input_ids'].to(device), 
                        attention_mask=batch['attention_mask'].to(device),
                        early_stopping=True)
        outs = [self.tokenizer.decode(ids) for ids in outs]
        answers.extend(outs)
    except KeyboardInterrupt:
      print('proceeds to evaluation')

    finally:
      predictions = []
      references = []
      input_texts = []
      for ref, pred in zip(self.valid_dataset, answers):
        pred = pred[4:-1]
        predictions.append(pred)
        
        input_ = self.tokenizer.decode(ref['input_ids'])
        input_ = ''.join(input_)[0:-3]
        input_ = re.sub('[!@#$*-]', '', input_)
        input_ = input_.lstrip().title()

        start_index = input_.index('>:')
        prefix = input_[:start_index]
        input_ = input_[start_index + 6:]

        input_texts.append(input_)

        output_ = self.tokenizer.decode(ref['target_ids'])
        output_ = ''.join(output_)[4:-3]
        references.append(output_)

      for _ in range(min(5, len(answers))):
        i = random.randint(0, len(predictions))
        print('Input:             {}\nPredicted Answer:  {}\nReal Answer:       {}\n'.format(input_texts[i],predictions[i], references[i]))

"""# Run Your Own Model Here"""

My_T5 = Custom_T5_Training(
    dataset_path= 'https://www.dropbox.com/s/6w5z4qvt8vytngm/training_data.csv?dl=1', # Try to put your data on your personal cloud database (as an url) so that you can keep training the model using the latest available data
    working_folder= '/content/',   # Change this to you google drive so that you dont have to retrain your own model from scratch in the future
    maximum_input_length=50,
    maximum_output_length= 16,
    model_name= 't5-small')   # Try t5-base or t5-large model if your computer can handle it

"""Uncomment and run the below function if you updated your dataset:"""

#My_T5.create_dataset()

My_T5.train_model()   # you can interrup training anytime and the model will be saved

My_T5.validate_model()  # Validating on dataset the model has never seen before

"""I removed the 'context" from the dataset so that I can use the dataset as a sequence to sequence model, not as a question to answer model. This explains why the predicted answers are not accurate. 

Nevertheless T5 is the state of the art NLP model. 

If you need to train a question to answer model, you needs to use a Long Form pretrained model (not a T5 model)
"""

