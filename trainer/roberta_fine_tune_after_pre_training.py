# Globals
# ---------------

# data locations
root = '/home/david/Desktop/projects/Reddit/'
ORIGIN = './data/labeled_sample_and_splits/'
DATA_FILE = ORIGIN+'labeled_data_2023-11-07.csv'
train_idx_path = ORIGIN+'train_idx_2023-11-07.npy'
val_idx_path   = ORIGIN+'val_idx_2023-11-07.npy'
test_idx_path  = ORIGIN+'test_idx_2023-11-07.npy'
RELABELED_DATA_FILE = ORIGIN+'relabeled_data_2023-11-07.csv'
AUGMENTED_DATA_FILE = ORIGIN+'train_manually_labeled_augmented_2023-11-08.csv'

# training params
batch_size = 16
epochs = 3
seed_val = 1234

# Model
HF_BERT_MODEL = './models/roberta-base_pretrained_80000.pt'
HF_TOKENIZER = "./models/roberta-base_pretrained_tokenizer/"

print(len('* Fine-Tuning *')*'*')
print('* Fine-Tuning *')
print(len('* Fine-Tuning *')*'*')

# ---------------

import os
import sys


os.chdir(root)
sys.path.append(root)

from utils.training_and_reporting import *

import re
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import time
import datetime
import random

import torch
from torch.utils.data import TensorDataset, random_split, SubsetRandomSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn

from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# Load Data
data = pd.read_csv(DATA_FILE,index_col=0)
data['label'] = data['Negative']

train_idx = np.load(train_idx_path)
val_idx   = np.load(val_idx_path)
test_idx  = np.load(test_idx_path)

# relabeled_data = pd.read_csv(RELABELED_DATA_FILE,index_col=0)
augmented_data = pd.read_csv(AUGMENTED_DATA_FILE,index_col=0)
data.iloc[:5]

# Add augmented data
augmented_data = augmented_data[['title','augmented_comment', 'label']].rename({'augmented_comment': 'comment'}, axis = 1)

# Add new relabeled data to the labled data
data.drop(['clean_label'],axis=1,inplace=True)

a = data.shape
data = pd.concat([data,augmented_data], axis = 0, ignore_index = True)
b = data.shape
print(f'Adding relabeled data to data: {a}->{b}')

# update train_idx
original_N  = a[0]
new_N       = b[0]
missing_idx = np.arange(original_N,new_N)
a = len(train_idx)
original_train_idx = train_idx.copy()
train_idx = np.array(train_idx.tolist()+missing_idx.tolist())
b = len(train_idx)
print(f'Training examples: {a}->{b}')

# Add relabeled data
relabeled_data = pd.read_csv(RELABELED_DATA_FILE,index_col=0)

# clean automatic labeling
relabeled_data['clean_label'][relabeled_data['clean_label'].str.contains('unrelated')] = 'unrelated'
relabeled_data['clean_label'][relabeled_data['clean_label'].str.contains('neutral')] = 'neutral'
relabeled_data['clean_label'][relabeled_data['clean_label'].isin(['anti semitic', 'anti israeli'])] = 'negative'
relabeled_data['clean_label'][relabeled_data['clean_label'].str.contains('pro israeli')] = 'positive'
relabeled_data['clean_label'] = (relabeled_data['clean_label'] == 'negative').astype(int)

display(pd.crosstab(relabeled_data['clean_label'],
            relabeled_data['label']))

# Keep only agreed observations
mask = (relabeled_data['label'] == relabeled_data['clean_label'])
a = relabeled_data.shape
relabeled_data = relabeled_data[mask].reset_index(drop=True)
b = relabeled_data.shape
print(f'Removing inconsistencies from relabeled data: {a}->{b}')

# Add new relabeled data to the labled data
relabeled_data.drop(['clean_label'],axis=1,inplace=True)

a = data.shape
data = pd.concat([data,relabeled_data], axis = 0, ignore_index = True)
b = data.shape
print(f'Adding relabeled data to data: {a}->{b}')

# update train_idx
original_N  = a[0]
new_N       = b[0]
missing_idx = np.arange(original_N,new_N)
a = len(train_idx)
original_train_idx = train_idx.copy()
train_idx = np.array(train_idx.tolist()+missing_idx.tolist())
b = len(train_idx)
print(f'Training examples: {a}->{b}')


# Shuffle
np.random.shuffle(train_idx)

# Check If there's a GPU
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Data Prep
sentences = np.array([f'{x} ; {y}' for (x,y) in zip(data['title'].tolist(), zip(data['comment'].tolist()))])
labels    = data[['label']].values

tokenizer = RobertaTokenizer.from_pretrained(HF_TOKENIZER)
max_len = 512

# Tokenize
input_ids = []
attention_masks = []

for sent in tqdm(sentences, desc='Processing sentences..'):
    encoded_dict = tokenizer.encode_plus(
                        sent,                           # Sentence to encode.
                        add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',          # Return pytorch tensors.
                   )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
labels = labels.long()

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])


# Split, stratify
N = len(labels)

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

dataset = TensorDataset(input_ids, attention_masks, labels)


train_dataloader = DataLoader(
            dataset,                  # The training samples.
            sampler = train_sampler,  # Select batches randomly
            batch_size = batch_size   # Trains with this batch size.
        )
validation_dataloader = DataLoader(
            dataset,                  # The validation samples.
            sampler = valid_sampler,  # Pull out batches sequentially.
            batch_size = batch_size   # Evaluate with this batch size.
        )

# define model
model = RobertaForSequenceClassification.from_pretrained(
    HF_BERT_MODEL,          # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2,               # The number of output labels--2 for binary classification.
                                  # You can increase this for multi-class tasks.
    output_attentions = False,    # Whether the model returns attentions weights.
    output_hidden_states = False # Whether the model returns all hidden-states.
    )

# Tell pytorch to run this model on the GPU.
model.cuda()

# training scheduler
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,            # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8            # args.adam_epsilon  - default is 1e-8.
                )

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# Training
# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()


# store values
labels_total = []
probs_total  = []


# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        result = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)

        loss = result.loss
        logits = result.logits

        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        scheduler.step()

        # store
        probs = torch.softmax(logits, dim=1)

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    # calc KPIs
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # sys.exit(0)

        with torch.no_grad():

            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        loss = result.loss
        logits = result.logits

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().tolist()
        b_labels = b_labels.detach().cpu().numpy().tolist()
        labels_total += b_labels
        probs_total  += probs

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

    y_true  = np.squeeze(np.array(labels_total))
    y_pred  = np.array(probs_total).argmax(axis=1)
    y_score = np.array(probs_total)[:,1]

    report(y_true, y_pred,y_score)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))


    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Save model and tokenizer
model.save_pretrained("./models/roberta_augmentation_label_spreading_after_pretraining_241123_model.pt")
tokenizer.save_pretrained("./models/roberta_augmentation_label_spreading_after_pretraining_241123_tokenizer")


