from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import sys 

class Tokenizer(object):
    def __init__(self, checkpoint, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = max_length

        self.tokenized_paper_texts = {}
        self.tokenized_dataset_texts = {}

  

    def preprocess_textual_data_papers(self, paper_texts):

        for paper_id, text in tqdm(enumerate(paper_texts), desc='Tokenizing Paper'):
            tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            if 'token_type_ids' in tokens:
                del tokens['token_type_ids']
            self.tokenized_paper_texts[paper_id] = tokens

        return self.tokenized_paper_texts
    
    def preprocess_textual_data_datasets(self, dataset_texts):

        for dataset_id, text in tqdm(enumerate(dataset_texts), desc='Tokenizing Dataset'):
            tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            if 'token_type_ids' in tokens:
                del tokens['token_type_ids']
            self.tokenized_dataset_texts[dataset_id] = tokens

        return self.tokenized_dataset_texts