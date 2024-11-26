from util.loader import Loader
from Model import Model
import torch
import pickle

file_training = './datafinder/train.csv'
file_testing = './datafinder/test.csv'

print('=' * 80)
print('Preprocess Data.....')

training_data = Loader.load_dataset(file_training)
test_data = Loader.load_dataset(file_testing)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model(training_data, test_data, device)
print('Training Model...')
model.train(device)
print('Testing.....')
rec_list = model.test(device, load_saved=True)
print(rec_list)
print('Evaluating...')
model.evaluate(rec_list)
