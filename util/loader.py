import numpy as np
import csv
from re import split


class Loader(object):
    def __init__(self):
        pass

    def load_dataset(file):
        data = []
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                paper_id = row['paper_id']
                dataset_id = row['dataset_id']
                weight = row['weight']
                input_ids_p = row['input_ids_p']
                attention_mask_p = row['attention_mask_p']
                input_ids_d = row['input_ids_d']
                attention_mask_d = row['attention_mask_d']
                data.append([paper_id, dataset_id, float(weight), input_ids_p, attention_mask_p, input_ids_d, attention_mask_d])

        return (data)