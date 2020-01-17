import os
import json
import nltk
import pandas as po
from tqdm import tqdm

from bert_as_a_service import add_bert_embeddings_to_df


### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        token = token.lower()
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")", "@",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    tokens1 = ' '.join(tokens1)
    return tokens1


def load_and_preprocess():
    if os.path.exists('data/cleaned_data_with_embeddings'):
        df = po.read_pickle('data/cleaned_data_with_embeddings')
    
    else:
        with open('data/semeval-2016_2017-task3-subtaskB-english.json', 'r') as f:
            data = json.load(f)

        df = po.DataFrame(columns = ['Question1', 'Question2', 'Similarity'])
        for key in tqdm(data):
            for i in range(len(data[key])):
                for j in range(len(data[key][i]['Threads'])):
                    if len(data[key][i]['Threads'][j]['RelQuestion']['RelQBody']) != 0:
                        row = {'Question1': data[key][i]['OrgQBody'],
                               'Question2': data[key][i]['Threads'][j]['RelQuestion']['RelQBody'],
                               'Similarity': data[key][i]['Threads'][j]['RelQuestion']['RELQ_RELEVANCE2ORGQ']
                              }
                        df = df.append(row, ignore_index=True)

        stopwords = list(set(nltk.corpus.stopwords.words("english")))

        ### tokenize & remove funny characters
        df["Question1"] = df["Question1"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
        df["Question2"] = df["Question2"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
        
        empty_questions = []
        for i in range(len(df)):
            if len(df.iloc[i]['Question1']) == 0 or len(df.iloc[i]['Question2']) == 0:
                empty_questions.append(i)
        
        df = df.drop(empty_questions, axis = 0).reset_index(drop = True)        
        
        df = add_bert_embeddings_to_df(df)
        
        df.to_pickle('data/cleaned_data_with_embeddings')

    df['Similarity'] = df['Similarity'].replace('PerfectMatch', 1)
    df['Similarity'] = df['Similarity'].replace('Relevant', 1)
    df['Similarity'] = df['Similarity'].replace('Irrelevant', 0)
            
    train = df[:int(0.8*len(df))].reset_index(drop = True)
    test = df[int(0.8*len(df)):].reset_index(drop = True)
    
    print('Loaded train set of length - {}'.format(len(train)))
    print('Loaded test set of length - {}'.format(len(test)))
    
    return train, test
