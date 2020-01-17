import torch
import torch.nn as nn
import torch.optim as optim


class finetuning_model(nn.Module):
    def __init__(self, embedding_dim, num_output_classes):
        super(finetuning_model, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_output_classes = num_output_classes

        self.linear = nn.Linear(embedding_dim*2, num_output_classes)

    def forward(self, sentence1, sentence2):
        
        sentence1 = torch.tensor(sentence1)
        sentence2 = torch.tensor(sentence2)
        
        diff = sentence1 - sentence2
        mult_elementwise = sentence1*sentence2
        
        input_ = torch.cat([diff, mult_elementwise], dim = 1)
        lin_out = self.linear(input_)
    
        return lin_out
