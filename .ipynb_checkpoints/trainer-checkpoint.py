from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


def train(train_df, num_epochs, model, loss_function, optimizer, load_path = None, load=False):
    if load and load_path is not None:
        model.load_state_dict(torch.load(load_path))
        print('loaded model')
        
    else:
        for epoch in range(num_epochs):
            print('Training Epoch - {}'.format(epoch))
            for i in tqdm(range(len(train_df))):
                model.zero_grad()

                output = model(train_df.iloc[i]['Question1_embedding'], train_df.iloc[i]['Question2_embedding'])
                target = torch.tensor([train_df.iloc[i]['Similarity']], dtype = torch.long)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

            print('loss = {}'.format(loss))
            
        save_path = 'saved_models/model_1'    
        print('Saving model at - {}'.format(save_path))
        torch.save(model.state_dict(), save_path)
