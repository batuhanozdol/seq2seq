import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from os.path import isfile, join

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from nlp import load_metric, Dataset as DT
import datasets

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelWithLMHead,
    get_linear_schedule_with_warmup
)

def get_last_500_tokens(string):
    # Split the string into a list of tokens
    tokens = string.split()
    # Truncate the list of tokens to the desired number of tokens
    truncated_tokens = tokens[-500:]
    # Join the truncated list of tokens back into a string
    return ' '.join(truncated_tokens)

def preprocess(text):
    text = re.sub(r"([.,!?¿])", r" \1 ", text)
    text = re.sub('\s{2,}', ' ', text)
    text = text.replace("?", "")
    text = text.replace(".", "")
    text = text.strip()
    #text = '<start> ' + text + ' <end>'    

    return get_last_500_tokens(text)

def load_data(): 
    with open('./dataset_batuhan.json', 'r') as f:
        data = json.load(f)

    context = []
    question = []
    answer = []
    
    for key in data:
        for item in data[key]:
            if item == "context":
                for context_item in data[key][item]:
                    context.append(context_item)
            elif item == "soru":
                for question_item in data[key][item]:
                    question.append(question_item)
            elif item == "cevap":
                for answer_item in data[key][item]:
                    answer.append(answer_item)
    
    # print(len(context))
    # print(len(question))
    # print(len(answer))
    
    return context, question, answer

def load_dataset(answer, question):
    train_set = []
    valid_set = []
    test_set = []

    for i in range(0, int(len(question)*70/100)):
        data = {}
        data['question'] = preprocess(question[i])
        data['answer']= preprocess(answer[i])
        train_set.append(data) 
    
    for i in range(int(len(question)*70/100), int(len(question)*90/100)):
        data = {}
        data['question'] = preprocess(question[i])
        data['answer']= preprocess(answer[i])
        test_set.append(data)  
    
    for i in range(int(len(question)*90/100), len(question)):
        data = {}
        data['question'] = preprocess(question[i])
        data['answer']= preprocess(answer[i])
        valid_set.append(data)  
    
    df_tr = pd.DataFrame.from_records(train_set)
    dataset_tr = DT.from_pandas(df_tr)
    
    df_test = pd.DataFrame.from_records(test_set)
    dataset_test = DT.from_pandas(df_test)
    
    df_valid = pd.DataFrame.from_records(valid_set)
    dataset_valid = DT.from_pandas(df_valid)
    
    return dataset_tr, dataset_test, dataset_valid


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams ) #hparams = hparams        
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.rouge_metric = load_metric('rouge') 
        
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
            
            
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
    
    def save_model(self, output_path):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        self.tokenizer.save_vocabulary(output_path)
    
    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    
    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
        
    def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
    )
    
    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
    def _generative_step(self, batch) :
        
        t0 = time.time()
        
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=150, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
            
        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]  
    
        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
#         rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(self.lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target)
        self.rouge_metric.add_batch(preds, target)
        
#         rouge_results = self.rouge_metric.compute() 
#         rouge_dict = self.parse_score(rouge_results)
#         base_metrics.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])
        
        return base_metrics
    

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
    
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log("avg_train_loss", avg_train_loss, logger=True, prog_bar=True)

        #return {"avg_train_loss": avg_train_loss, log = tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)
    
  
    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        
        rouge_results = self.rouge_metric.compute() 
        rouge_dict = self.parse_score(rouge_results)
    
        tensorboard_logs.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])
        
        ## Clear out the lists for next epoch
        self.target_gen= []
        self.prediction_gen=[]
        self.log("avg_val_loss", avg_loss, logger=True, prog_bar=True)
        
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]
  
    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp=False):
    #    if self.trainer.use_tpu:
    #        xm.optimizer_step(optimizer)
    #    else:
    #        optimizer.step()
    #    optimizer.zero_grad()
    #    self.lr_scheduler.step()
  
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict
    
    def train_dataloader(self):   
        n_samples = self.n_obs['train']
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        validation_dataset = get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)
    
    
    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", num_samples=n_samples, args=self.hparams)
        
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

class IctihatDataset(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):  
        if type_path == 'validation':
            self.dataset =  validation_data
        elif type_path == 'train':
            self.dataset =  train_data
        elif type_path == 'test':
            self.dataset =  test_data

        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
  
    def __len__(self):
        return self.dataset.shape[0]
    
    def clean_text(self, text):
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        
        return text
    
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['question']))
        
        input_ = self.clean_text(example_batch['question'])
        target_ = self.clean_text(example_batch['answer'])
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
       
        return source, targets
  
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

def get_dataset(tokenizer, type_path, num_samples, args):
      return IctihatDataset(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                        output_length=args.max_output_length)

def train():
    args_dict = dict(
        output_dir=checkpoint_output_path, # path to save the checkpoints
        model_name_or_path=base_t5_model,
        tokenizer_name_or_path=base_t5_model,
        max_input_length=512,
        max_output_length=200,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=1,
        eval_batch_size=1,
        num_train_epochs=5,
        gradient_accumulation_steps=1,
        n_gpu=1,
        resume_from_checkpoint=None, 
        val_check_interval = 0.05, 
        n_val=1000,
        n_train=-1,
        n_test=-1,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )
            
    args = argparse.Namespace(**args_dict)
    print(args_dict)        

    ## Define Checkpoint function
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,  
        filename='{epoch}-{step:.2f}',
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
        mode="max"
    )

    ## If resuming from checkpoint, add an arg resume_from_checkpoint
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=args.val_check_interval,
        callbacks=[LoggingCallback(), checkpoint_callback],
        #resume_from_checkpoint="/home/ubuntu/summarization/baselines/t5/results/checkpoints/epoch=5-step=75400.00.ckpt"
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.save_model(result_output_path)


def generate_summary(model, batch): 
    #tokenized_text = tokenizer.encode(batch['text'][0], return_tensors="pt", truncation=True, max_length=512).to("cuda")
    tokenized_text = tokenizer.encode(batch['question'][0], return_tensors="pt", truncation=True, max_length=512)

    summary_ids = model.generate(
	    tokenized_text,
            max_length=SUMMARY_LEN, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
        )
    result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return { 'pred_summary': [result],  'answer' : batch['answer']}
    

def test(checkpoint):
    batch_size = 16 #16 for test runs

    model = AutoModelWithLMHead.from_pretrained(checkpoint).to("cuda")
    #model = AutoModelWithLMHead.from_pretrained(checkpoint)

    results = test_data.map(lambda batch: generate_summary(model, batch), batched=True, batch_size=1)
    print(results)

    result = rouge.compute(predictions=results["pred_answer"], references=results["answer"], rouge_types=["rouge2"])["rouge2"].mid
    pd.options.display.max_colwidth = None
    df = pd.DataFrame(results)

    print(df.head())
    with open('./h5-model/export.json', "w", encoding='utf-8') as outfile:
        df.to_json(outfile, orient='records', force_ascii=False, indent=4)

    pd.options.display.max_colwidth = 50
    print(result)


    
rouge = datasets.load_metric("rouge")
base_t5_model = 'google/mt5-base'
tokenizer = T5Tokenizer.from_pretrained(base_t5_model)
data_dir = './'
checkpoint_output_path = './t5-model/checkpoints/'
result_output_path = './t5-model/model'
SECTION_MAX_TOKEN = 250
SUMMARY_LEN = 200
    
context, question, answer = load_data()
    
for i in range(0, len(answer)):
    answer[i] = preprocess(answer[i])
    question[i] = preprocess(question[i])

train_data, test_data, validation_data = load_dataset(answer, question)
logger = logging.getLogger(__name__)

train()
#test(result_output_path)
