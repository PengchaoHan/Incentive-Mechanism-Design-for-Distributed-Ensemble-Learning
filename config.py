import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=1)  # 1 - use GPU; 0 - do not use GPU
parser.add_argument('-seeds', type=int, default=1)  # e.g., 1,2,3
parser.add_argument('-n', type=int, default=100)


args = parser.parse_args()

use_gpu = bool(args.gpu)
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda:0') if use_gpu else torch.device('cpu')


dataset = 'MNIST'
model_name = 'ModelCNNMnist'  # 'ModelCNNMnist', 'LeNet5', 'LeNet5Half'


optimizer = 'Adam'
step_size = 0.001  # learning rate of clients, Adam optimizer
batch_size_train = 32
batch_size_eval = 512
max_iter = 1000  #1000  # Maximum number of iterations to run
seed = args.seeds
num_iter_one_output = 200
num_of_base_learners = args.n  # 10,20,30,40,50, 60,70,80,90,100
dataset_file_path = os.path.join(os.path.dirname(__file__), 'dataset_data_files')
results_file_path = os.path.join(os.path.dirname(__file__), 'results/')
