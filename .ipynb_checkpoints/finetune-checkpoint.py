import pandas as po
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from trainer import train
from tester import test
from preprocess import load_and_preprocess
from finetuning_model import finetuning_model

train_df, dev_df = load_and_preprocess()

num_epochs = 5
embedding_dim = 768
num_output_classes = 2

model = finetuning_model(embedding_dim, num_output_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(train_df, num_epochs, model, loss_function, optimizer, load = True, load_path = 'saved_models/model_1')

predictions_df = test(dev_df, model)
