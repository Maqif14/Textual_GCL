from random import shuffle,randint,choice,sample
import numpy as np
import random
import torch

def next_batch_pairwise(data, batch_size):
    paper_batch, pos_batch, neg_batch = [], [], []
    input_ids_p_batch, attention_mask_p_batch = [], []
    input_ids_pos_d_batch, attention_mask_pos_d_batch = [], []
    input_ids_neg_d_batch, attention_mask_neg_d_batch = [], []

    papers = list(data.training_set_u.items())
    random.shuffle(papers) 

    for paper, interactions in papers:
        pos_dataset = random.choice(list(interactions.keys()))
        pos_interaction = interactions[pos_dataset]

        input_ids_p = pos_interaction['input_ids_p']
        attention_mask_p = pos_interaction['attention_mask_p']

        input_ids_p = string_to_tensor(input_ids_p)
        attention_mask_p = string_to_tensor(attention_mask_p)

        # Get positive dataset info
        input_ids_d_pos = data.training_set_i[pos_dataset][paper]['input_ids_d']
        attention_mask_d_pos = data.training_set_i[pos_dataset][paper]['attention_mask_d']

        input_ids_d_pos = string_to_tensor(input_ids_d_pos)
        attention_mask_d_pos = string_to_tensor(attention_mask_d_pos)

        neg_dataset = random.choice([i for i in data.dataset if i != pos_dataset])

        paper_num = list(data.training_set_i[neg_dataset].keys())[0]
    

        input_ids_d_neg = data.training_set_i[neg_dataset][paper_num]['input_ids_d']
        attention_mask_d_neg = data.training_set_i[neg_dataset][paper_num]['attention_mask_d']

        input_ids_d_neg = string_to_tensor(input_ids_d_neg)
        attention_mask_d_neg = string_to_tensor(attention_mask_d_neg)

        paper_batch.append(int(paper))
        pos_batch.append(int(pos_dataset))
        neg_batch.append(int(neg_dataset))

        input_ids_p_batch.append(torch.tensor(input_ids_p))  
        attention_mask_p_batch.append(torch.tensor(attention_mask_p))

        input_ids_pos_d_batch.append(torch.tensor(input_ids_d_pos))
        attention_mask_pos_d_batch.append(torch.tensor(attention_mask_d_pos))

        input_ids_neg_d_batch.append(torch.tensor(input_ids_d_neg))
        attention_mask_neg_d_batch.append(torch.tensor(attention_mask_d_neg))

        # Once we reach the batch size, yield the batch
        if len(paper_batch) == batch_size:
            yield torch.tensor(paper_batch), torch.tensor(pos_batch), torch.tensor(neg_batch), \
                  torch.stack(input_ids_p_batch), torch.stack(attention_mask_p_batch), \
                  torch.stack(input_ids_pos_d_batch), torch.stack(attention_mask_pos_d_batch), \
                  torch.stack(input_ids_neg_d_batch), torch.stack(attention_mask_neg_d_batch)

            # Reset the batch lists for the next batch
            paper_batch, pos_batch, neg_batch = [], [], []
            input_ids_p_batch, attention_mask_p_batch = [], []
            input_ids_pos_d_batch, attention_mask_pos_d_batch = [], []
            input_ids_neg_d_batch, attention_mask_neg_d_batch = [], []


def string_to_tensor(tensor_string):
    tensor_string = tensor_string.strip().replace("input_ids_p: tensor([", "").replace("])", "").replace("attention_mask_p: tensor([", "").replace("attention_mask_d: tensor([", "").replace("input_ids_d: tensor([", "")
    
    tensor_values = [int(val.strip()) for val in tensor_string.split(',') if val.strip().isdigit()]
    
    return torch.tensor(tensor_values)


