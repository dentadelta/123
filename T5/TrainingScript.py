import pyarrow as pa
import pandas as pd
import os
import torch
from transformers import (T5Config, T5Tokenizer, T5ForConditionalGeneration, TextDataset, DataCollator, Trainer, TrainingArguments,HfArgumentParser)
import random
from typing import Dict, List
import nlp
from dataclasses import dataclass, field
from tqdm.auto import tqdm, trange
import re
import pathlib
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from typing import Optional,NamedTuple
import json
import logging
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
logger = logging.getLogger(__name__)
from packaging import version
try:
    import wandb
    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    print('wandb not setup')
    _has_wandb = False

def is_wandb_available():
    return _has_wandb

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

class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float

#Subclassed from Huggingface to stop it from spamming the terminal with useless information        
class customTrainer(Trainer):
    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)

      
    def train(self, model_path: Optional[str] = None):
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1)
        else:

            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
        # Train!
        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
 
        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

            except ValueError:
                self.global_step = 0


        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable= self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)


            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable= True)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)


                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)
                        if self.args.evaluate_during_training:
                            self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()
                            
                        if self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()

        return TrainOutput(self.global_step, tr_loss / self.global_step)


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
    input_encodings = self.tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=self.maximum_input_length)     
    target_encodings = self.tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=self.maximum_output_length)     
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
    trainer = customTrainer(
                        model= self.model,
                        args=self.training_args,
                        data_collator=self.data_collator,
                        train_dataset=self.train_dataset,
                        prediction_loss_only=False,
                      )
    try:
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

@dataclass
class CustomT5Argument:
    train_data_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    workingfolder: Optional[str] = field(default=None, metadata={"help": "Location where the model will be saved."})
    training_library: Optional[str] = field(default=None, metadata={"help": "The training library where good examples are saved"})
    input_size: Optional[int] = field(default=150, metadata={"help": "maximum input length, the default value is 150"})
    output_size: Optional[int] = field(default=50, metadata={"help": "maximum output length, the default value is 50"})
    epochs: Optional[int] = field(default=1, metadata={"help": "Number of training epochs"})
    print_loss: Optional[int] = field(default=10, metadata={"help": "Print loss every X steps"})
    command: Optional[str] = field(default='KnowledgeUpdate', metadata={"help": """Specify what you want to do with the model. Possible commands are: KnowledgeUpdate, KnowledgeUpgrade, Training, TrainingFromFile,Confirmation"""})
    wandb_project_name: Optional[str] = field(default='My Project', metadata={"help": "Specify the wandb project name so that you can view your loss at wandb website. If you run the program on the cloud, all you will see is a black terminal. Not fun. Dont worry the website only track your loss, epoch and step"})

def main():
    parser = HfArgumentParser((CustomT5Argument))
    args = parser.parse_args_into_dataclasses()[0]

    os.environ["WANDB_PROJECT"] = args.wandb_project_name
    IMPROVEMENT = args.training_library
    WORKING_FOLDER = args.workingfolder
    DATAPATH = args.train_data_file
    command = args.command
    epochs = args.epochs
    steps = args.print_loss
    input_length = args.input_size
    output_size = args.output_size

    My_T5 = Custom_T5_Training(
          dataset_path= f'{IMPROVEMENT}/accumulatedknowledge.csv' if command == 'Training' else DATAPATH,
          working_folder= WORKING_FOLDER,
          maximum_input_length=input_length,
          maximum_output_length= output_size,
          model_name= 't5-base',
          logging_step = steps,
          epochs = epochs)

    if command == 'KnowledgeUpdate':
        Step1(My_T5)
        with open('test_file.txt','w') as file:
          file.write('prefix:\ninput_text:')
          file.close()
    
    if command == 'KnowledgeUpgrade':
        Step2(My_T5,IMPROVEMENT)
    
    if command == 'Training':
        print('The model will be trained from this file: ',My_T5.dataset_path)
        Step3(My_T5)

    if command == 'TrainingFromFile':
        print('The model will be trained from this file',My_T5.dataset_path)
        Step3(My_T5)    

    if command == 'Confirmation':
        Step4(My_T5)
    
        
if __name__ == "__main__":
    main()
