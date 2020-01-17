import os
import torch
import pandas as po
from tqdm import tqdm


def add_bert_embeddings_to_df(df):
    from bert_serving.client import BertClient
    bc = BertClient()

    df["Question1_embedding"] = df["Question1"].apply(lambda row: bc.encode([row]))
    df["Question2_embedding"] = df["Question2"].apply(lambda row: bc.encode([row]))
    
    return df
