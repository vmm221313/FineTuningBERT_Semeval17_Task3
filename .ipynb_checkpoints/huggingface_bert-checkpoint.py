import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def build_vocab(df):
    print('Building Vocabulary...')
    


def get_cls_embeeding_of_sent(sent, bert_model, tokenizer):
    sent_ids = torch.tensor([tokenizer.encode(sent)])
    sent_cls_embedding = bert_model(sent_ids)[0][0][0]
    
    return sent_cls_embedding


def add_bert_embeddings_to_df(df):
    bert_model = BertModel.from_pretrained('bert-base-uncased/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    df["Question1_embedding"] = df["Question1"].apply(lambda row: get_cls_embeeding_of_sent(row, bert_model, tokenizer))
    df["Question2_embedding"] = df["Question2"].apply(lambda row: get_cls_embeeding_of_sent(row, bert_model, tokenizer))
    
    return df
