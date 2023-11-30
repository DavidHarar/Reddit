# %%

# Globals
# ---------------
root = '/home/david/Desktop/projects/Reddit/'
ORIGIN      = './data/unlabeled/'
DATA_FILE   = ORIGIN+'worldnews_processed_unlabeled_comments_70k.csv'
TXT_FILE    = ORIGIN+'worldnews_processed_unlabeled_comments_70k.txt'

# training params
batch_size = 16
epochs = 10
seed_val = 1234

# Model
HF_BERT_MODEL = 'roberta-base'
MODEL_PATH  = f'./models/{HF_BERT_MODEL}_retrained/'

# Setup
# ---------------

import os
import sys

os.chdir(root)
sys.path.append(root)

import os
import sys
import numpy as np
import pandas as pd
import random

import torch

from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print(len('* Continue Pre-Training *')*'*')
print('* Continue Pre-Training *')
print(len('* Continue Pre-Training *')*'*')


# -------------------

# Load Data
data = pd.read_csv(DATA_FILE,index_col=0)
tokenizer = RobertaTokenizer.from_pretrained(HF_BERT_MODEL)
max_len = 512

train = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=TXT_FILE,
    block_size=512,
)

# Load Model
model = RobertaForMaskedLM.from_pretrained(
      HF_BERT_MODEL,          # Use the 12-layer BERT model, with an uncased vocab.
      num_labels = 2,               # The number of output labels--2 for binary classification.
                                    # You can increase this for multi-class tasks.
      output_attentions = False,    # Whether the model returns attentions weights.
      output_hidden_states = False # Whether the model returns all hidden-states.
      )
model.cuda()

# Create a Data Collector
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    # per_gpu_train_batch_size=3*20,
    per_device_train_batch_size=8,
    save_steps=20_000,
    save_total_limit=5,
    prediction_loss_only=True,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train,
)

# Train
trainer.train()

