#!/usr/bin/env python
# coding: utf-8

# In[22]:


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, AutoTokenizer, BartTokenizer
import torch.optim
import accelerate
import os
import json
from tqdm import tqdm
from time import time
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu_score
import nltk
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider


# In[23]:


data_root_path = os.path.join(os.getcwd(), 'v3/v2/en-hi')
model_path = os.path.join(os.path.join(os.getcwd(), 'Models'), 'Machine_Translation')


# In[24]:


class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        #self.data = self.load_data(data_path, sample_size)
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        #self.sample_size = sample_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['English']
        target_text = item['Hindi']
        
        inputs = self.tokenizer.encode_plus(input_text, max_length=self.max_length,
                                            padding='max_length', truncation=True, return_tensors='pt'
                                           )
        
        targets = self.tokenizer.encode(target_text, max_length=self.max_length,
                                        padding='max_length', truncation=True, return_tensors='pt'
                                        )
        
        return {'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets.squeeze()
               }
    """def load_data(self, data_path, sample_size):
        '''with open(data_path, 'r',encoding="utf8") as file:
            data = json.load(file)'''
        data = []
        with open(os.path.join(data_root_path, 'train.en'), 'r', encoding="utf8") as en_file:
            en_lines = en_file.readlines()
        with open(os.path.join(data_root_path, 'train.hi'), 'r', encoding="utf8") as hn_file:
            hn_lines = hn_file.readlines()
            
        for en_line, hn_line in zip(en_lines, hn_lines):
            data.append(dict(English=en_line.replace('\n', ''), Hindi=hn_line.replace('\n', '')))
            
        return sample(data, sample_size)"""


# In[25]:


model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)


# In[26]:


def load_data(data_path, sample_size):
    '''with open(data_path, 'r',encoding="utf8") as file:
        data = json.load(file)'''
    data = []
    with open(os.path.join(data_root_path, 'train.en'), 'r', encoding="utf8") as en_file:
        en_lines = en_file.readlines()
    with open(os.path.join(data_root_path, 'train.hi'), 'r', encoding="utf8") as hn_file:
        hn_lines = hn_file.readlines()

    for en_line, hn_line in zip(en_lines, hn_lines):
        data.append(dict(English=en_line.replace('\n', ''), Hindi=hn_line.replace('\n', '')))

    return sample(data, sample_size)


# In[6]:


data = load_data(data_root_path, 5600)


# In[7]:


train_data, test_data = train_test_split(data, test_size=0.1785, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.1304, shuffle=False)


# In[8]:


train_dataset = CustomDataset(tokenizer, train_data)
dev_dataset = CustomDataset(tokenizer, val_data)
test_dataset = CustomDataset(tokenizer, test_data)


# In[9]:


train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=25, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)


# In[10]:


def Fine_Tune(train_loader, val_loader, num_epochs, num_layers_to_finetune):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    
    for param in model.parameters():
        param.requires_grad = False
    
    for i in range(1, num_layers_to_finetune + 1):
        for param in model.model.encoder.layers[-i].parameters():
            param.requires_grad = True
        for param in model.model.decoder.layers[-i].parameters():
            param.requires_grad = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-6)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("Training phase")
        #for idx, batch in enumerate(train_loader):
        for batch in tqdm(train_loader):
            #time0 = time()
            #print(f"batch: {idx+1} starts")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            #time1 = time()
            #print(f"batch: {idx+1} ends\ntime taken: {time1-time0} seconds")
            #print(f"time taken: {time1-time0} seconds")
        
        model.eval()
        print("Validation phase")
        with torch.no_grad():
            total_val_loss = 0.0
            for val_batch in tqdm(val_loader):
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)
                val_output = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += val_output.loss.item()
            average_val_loss = total_val_loss / len(val_loader)
            
        print(f'Epoch: {epoch+1}/{num_epochs}, Validation Loss: {average_val_loss}')
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path, from_pt=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))


# In[11]:


def Generate_translation_show(val_loader):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for val_batch in val_loader:
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            labels = val_batch['labels'].to(device)
            generated_ids = model.generate(input_ids, attention_mask=attention_mask,
                                           max_length=60, num_beams=2, repetition_penalty=2.0,
                                           length_penalty=2.0, early_stopping=True
                                          )
            generated_title = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            actual_title = tokenizer.decode(labels[0], skip_special_tokens=True)
            
            print(f'Input Text: {input_text}\nGenerated Translation: {generated_title}\nActual Translation: {actual_title}')
            print('\n'+'='*50+'\n')


# In[12]:


def Generate_translation(val_loader):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    generated_translations = []
    actual_translations = []
    with torch.no_grad():
        for val_batch in val_loader:
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            labels = val_batch['labels'].to(device)
            generated_ids = model.generate(input_ids, attention_mask=attention_mask,
                                           max_length=60, num_beams=2, repetition_penalty=2.0,
                                           length_penalty=2.0, early_stopping=True
                                          )
            generated_translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            actual_translation = tokenizer.decode(labels[0], skip_special_tokens=True)
            generated_translations.append(generated_translation)
            actual_translations.append(actual_translation)
    return generated_translations, actual_translations


# In[13]:


def calculate_bleu_scores(actual, generated):
    smoothie = SmoothingFunction().method4
    actual_tokenized = [[nltk.word_tokenize(act) for act in group] for group in actual]
    generated_tokenized = [nltk.word_tokenize(gen) for gen in generated]
    score = corpus_bleu(actual_tokenized, generated_tokenized, smoothing_function=smoothie)
    return score


# In[14]:


def calculate_rouge_scores(actual, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, actual)
    return scores


# In[15]:


def calculate_cider_scores(actual, generated):
    act_dict = {idx: [line] for idx, line in enumerate(actual)}
    gen_dict = {idx: [line] for idx, line in enumerate(generated)}
    cider = Cider()
    (score, scores) = cider.compute_score(act_dict, gen_dict)
    return score


# In[12]:


Fine_Tune(train_loader, val_loader, 5, 2)


# In[27]:


generated_translations, actual_translations = Generate_translation(test_loader)


# In[17]:


bleu_score = calculate_bleu_scores(actual_translations, generated_translations)


# In[18]:


rouge_score = calculate_rouge_scores(actual_translations, generated_translations)


# In[19]:


cider_score = calculate_cider_scores(actual_translations, generated_translations)


# In[20]:


bleu_score, rouge_score, cider_score


# In[21]:


Generate_translation_show(test_loader)


# In[ ]:




