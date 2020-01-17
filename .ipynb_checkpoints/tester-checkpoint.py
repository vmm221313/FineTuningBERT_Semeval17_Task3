import pandas as po
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim


def test(test_df, model):
    model.eval()
    predictions_df = po.DataFrame(columns = ['Predicted Similarity', 'Actual Similarity'])
    for i in tqdm(range(len(test_df))):
        with torch.no_grad():
            
            output = model(test_df.iloc[i]['Question1_embedding'], test_df.iloc[i]['Question2_embedding']) 
            prediction = nn.Softmax(dim=1)(output)[0].numpy().argmax()

            row = {'Predicted Similarity': prediction,
                   'Actual Similarity': test_df.iloc[i]['Similarity']
                  }

            predictions_df = predictions_df.append(row, ignore_index=True)
    
    predictions_df = predictions_df.astype(int)
    print('average_precision score = {}'.format(average_precision_score(predictions_df['Actual Similarity'], predictions_df['Predicted Similarity'])))
        
    return predictions_df
