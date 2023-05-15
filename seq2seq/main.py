import json
import os
import re
from datasets import Dataset
import datasets
from os.path import isfile, join
import pandas as pd
from transformers import BertTokenizerFast, EncoderDecoderModel, BertTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, GPT2Tokenizer, Trainer, TrainingArguments
from tokenizers.normalizers import NFKC
from tokenizers import normalizers
import unicodedata

sample_size = 5000
encoder_max_length=512
decoder_max_length=200 #128
# batch_size = 16
batch_size=1

data_dir = './'
results_dir = './seq2seq/'
bert_model_name = 'dbmdz/bert-base-turkish-cased' #'dbmdz/bert-base-turkish-cased'
gpt2_model_name = 'gorkemgoknar/gpt2-small-turkish' # 'redrussianarmy/gpt2-turkish-cased'
rouge = datasets.load_metric("rouge")

# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

def unicode_normalization(text):
    return unicodedata.normalize('NFKC', text)

GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
bert_tokenizer.normalizer = normalizers.Sequence([NFKC()])
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token

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

def map_to_length(x):
    	x["question_len"] = len(bert_tokenizer(x["question"]).input_ids)
    	x["question_longer_512"] = int(x["question_len"] > 512)
    	x["answer_len"] = len(bert_tokenizer(x["answer"]).input_ids)
    	x["answer_longer_64"] = int(x["answer_len"] > 64)
    	x["answer_longer_128"] = int(x["answer_len"] > 128)
    	return x

def compute_and_print_stats(x):
    	if len(x["question_len"]) == sample_size:
    		print("Question Mean: {}, %-Question > 512:{}, Answer Mean:{}, %-SAnswer > 64:{}, %-Answer > 128:{}".format(sum(x["question_len"]) / sample_size,
    		sum(x["question_longer_512"]) / sample_size, sum(x["answer_len"]) / sample_size, sum(x["answer_longer_64"]) / sample_size, sum(x["answer_longer_128"]) / sample_size))

def process_data_to_model_inputs(batch, decoder):
        	# tokenize the inputs and labels
        	inputs = bert_tokenizer(batch["question"], padding="max_length", truncation=True, max_length=encoder_max_length)
        	if decoder == 'bert':
        		outputs = bert_tokenizer(batch["answer"], padding="max_length", truncation=True, max_length=decoder_max_length)
        	else:
        		outputs = gpt2_tokenizer(batch["answer"], padding="max_length", truncation=True, max_length=decoder_max_length)
        
        	batch["input_ids"] = inputs.input_ids
        	batch["attention_mask"] = inputs.attention_mask
        	batch["decoder_input_ids"] = outputs.input_ids
        	batch["decoder_attention_mask"] = outputs.attention_mask
        	batch["labels"] = outputs.input_ids.copy()
        	
        	# because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        	# We have to make sure that the PAD token is ignored
        	if decoder == 'bert':
        		batch["labels"] = [[-100 if token == bert_tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
        	else:
        	    	# complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
        		batch["labels"] = [[-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]]
        	return batch

def save_model(model):
	model.save_pretrained("bert2bert")

def compute_metrics_bert(pred):
        	labels_ids = pred.label_ids
        	pred_ids = pred.predictions
        	
        	pred_str = bert_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        	labels_ids[labels_ids == -100] = bert_tokenizer.pad_token_id
        	label_str = bert_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        	
        	return compute_metrics(pred_str, label_str)

def compute_metrics_gpt2(pred):
        	labels_ids = pred.label_ids
        	pred_ids = pred.predictions
        	
        	pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        	labels_ids[labels_ids == -100] = gpt2_tokenizer.eos_token_id
        	label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        	
        	return compute_metrics(pred_str, label_str)
	
def compute_metrics(pred_str, label_str):
        	rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
        
        	return {
        		"rouge2_precision": round(rouge_output.precision, 4),
                	"rouge2_recall": round(rouge_output.recall, 4),
                	"rouge2_fmeasure": round(rouge_output.fmeasure, 4)
        	}

def train_bert2bert(train_data, val_data):
        	data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)
        	output = data_stats.map(compute_and_print_stats, batched=True, batch_size=-1)
        	
        	#delete
        	#train_data = train_data.select(range(64))
        	train_data = train_data.map(lambda batch: process_data_to_model_inputs(batch, 'bert'), batched=True, batch_size=batch_size, remove_columns=["question", "answer"])
        
        	train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
        	#delete	
        	#val_data = val_data.select(range(16)) 
        	val_data = val_data.map(lambda batch: process_data_to_model_inputs(batch, 'bert'), batched=True, batch_size=batch_size, remove_columns=["question", "answer"])
        	val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
        
        	bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(bert_model_name, bert_model_name, tie_encoder_decoder=True)
        	
        	bert2bert.config.decoder_start_token_id = bert_tokenizer.cls_token_id
        	bert2bert.config.eos_token_id = bert_tokenizer.sep_token_id
        	bert2bert.config.pad_token_id = bert_tokenizer.pad_token_id
        	bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size
        	
        	#configure these
        	bert2bert.config.max_length = 200 #142
        	bert2bert.config.min_length = 56
        	bert2bert.config.no_repeat_ngram_size = 3
        	bert2bert.config.early_stopping = True
        	bert2bert.config.length_penalty = 2.0
        	bert2bert.config.num_beams = 4
        
        	training_args = Seq2SeqTrainingArguments(
            		predict_with_generate=True,
            		evaluation_strategy="steps",
            		per_device_train_batch_size=batch_size,
            		per_device_eval_batch_size=batch_size,
            		fp16=True, 
            		output_dir=results_dir + "results/bert2bert",
            		#logging_steps=2,
            		#save_steps=10,
            		#eval_steps=4,
            		logging_steps=1000,
            		save_steps=500,
            		eval_steps=7500,
            		warmup_steps=2000,
            		save_total_limit=5,
            		num_train_epochs=100
        	)
        
        	
        	trainer = Seq2SeqTrainer(model=bert2bert, tokenizer=bert_tokenizer, args=training_args, compute_metrics=compute_metrics_bert, train_dataset=train_data, eval_dataset=val_data)
        	trainer.train()

def train_bert2gpt2(train_data, val_data):
        	data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)
        	output = data_stats.map(compute_and_print_stats, batched=True, batch_size=-1)
        	
        	#delete
        	#train_data = train_data.select(range(64))
        	train_data = train_data.map(lambda batch: process_data_to_model_inputs(batch, 'gpt2'), batched=True, batch_size=batch_size, remove_columns=["question", "answer"])
        
        	train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
        	#delete	
        	#val_data = val_data.select(range(16)) 
        	val_data = val_data.map(lambda batch: process_data_to_model_inputs(batch, 'gpt2'), batched=True, batch_size=batch_size, remove_columns=["question", "answer"])
        	val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
        
        	bert2gpt2 = EncoderDecoderModel.from_encoder_decoder_pretrained(bert_model_name, gpt2_model_name, tie_encoder_decoder=True)
        	bert2gpt2.decoder.config.use_cache = False 	# cache is currently not supported by EncoderDecoder framework
        	# CLS token will work as BOS token
        	bert_tokenizer.bos_token = bert_tokenizer.cls_token
        	# SEP token will work as EOS token
        	bert_tokenizer.eos_token = bert_tokenizer.sep_token
        	
        	# set decoding params
        	bert2gpt2.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
        	bert2gpt2.config.eos_token_id = gpt2_tokenizer.eos_token_id
        	bert2gpt2.config.max_length = 142
        	bert2gpt2.config.min_length = 56
        	bert2gpt2.config.no_repeat_ngram_size = 3
        	bert2gpt2.early_stopping = True
        	bert2gpt2.length_penalty = 2.0
        	bert2gpt2.num_beams = 4
        
        	training_args = TrainingArguments(
            		output_dir= results_dir + "results/bert2gpt",
            		per_device_train_batch_size=batch_size,
            		per_device_eval_batch_size=batch_size,
            		#predict_from_generate=True,
            		#evaluate_during_training=True,
            		do_train=True,
            		do_eval=True,
            		logging_steps=1000,
            		save_steps=1000,
            		eval_steps=1000,
            		overwrite_output_dir=True,
            		warmup_steps=2000,
            		save_total_limit=10,
            		fp16=True,
        		num_train_epochs=100
        	)
        
        	trainer = Trainer(model=bert2gpt2, args=training_args, compute_metrics=compute_metrics_gpt2, 
                           train_dataset=train_data, eval_dataset=val_data)	
        	trainer.train()

def generate_summary(batch, model, decoder):
    inputs = bert_tokenizer(batch["question"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    outputs = model.generate(input_ids, attention_mask=attention_mask)
	
    if decoder == 'bert':
        decoder_tokenizer = bert_tokenizer
    else:
        decoder_tokenizer = gpt2_tokenizer
	
    output_str = decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_answer"] = output_str
    print(outputs)
	
    return batch
        
def test(checkpoint, decoder, test_data):
	batch_size = 16 #16 for test runs
	model = EncoderDecoderModel.from_pretrained(checkpoint).to("cuda")
	results = test_data.map(lambda batch: generate_summary(batch, model, decoder), batched=True, batch_size=batch_size)
	result = rouge.compute(predictions=results["pred_answer"], references=results["answer"], rouge_types=["rouge2"])["rouge2"].mid
	pd.options.display.max_colwidth = None
	df = pd.DataFrame(results)
	print(df.head())
	with open('./seq2seq/export.json', "w", encoding='utf-8') as outfile:
		df.to_json(outfile, orient='records', force_ascii=False, indent=4)
	pd.options.display.max_colwidth = 50
	print(result)
        
        
        
        
	