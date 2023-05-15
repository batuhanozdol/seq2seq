# EncoderDecoder Summarization

This is the code for training two encoder-decoder models used for summarization, bert2bert and bert2gpt2.
Based on the notebook by patrickvonplaten (huggingface) to fine-tune summarization models using bert2bert and bert2gpt2.

## Note
For each model, dataset_batuhan.json dataset has to be placed under the current directory of the working model's .py path.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Use train_bert2bert.py to train bert2bert 
Use train_bert2bert.py to train bert2gpt2

```bash
python train_bert2bert.py
python train_bert2bert.py
```

Use test.py train to generate summaries for jsons under test directory using either models.
python test.py

Specify the the model using the checkpoint path in test script.

```python
from main import test

if __name__ == '__main__':
	test('./seq2seq/checkpoint-3000', 'gpt2', test_data)
```
