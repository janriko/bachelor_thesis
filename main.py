import random

import transformers
from typing import Dict, List, TextIO
import json
from transformers import BertTokenizer, BertModel, BertConfig, PretrainedConfig
import torch

import file_loader.FileLoader
import tokenizing
from models import *
import preprocessing
import pickle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from models.preprocessing.PreprocessedDataset import PreprocessedDataset
from models.raw_state.JerichoDataset import JerichoDataset
from models.tokenized.TokenizedDataset import TokenizedDataset

if __name__ == "__main__":
    # either get preprocessed dataset from cache (give file name as parameter)
    # or recompile dataset from file (give filename with jericho dataset)
    # preprocessed_data_set: PreprocessedDataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph("JerichoWorld-main/data/small_training_set.json")

    # create tokenized ids with dataset (give dataset)
    # or load from cache (don't give dataset as parameter)
    token_ids: TokenizedDataset = file_loader.get_tokenized_text_and_graph_ids()  # preprocessed_data_set)

    train_dataloader = DataLoader(
        token_ids, shuffle=False, batch_size=10
    )

    # print(token_ids.tokenized_states[3].text_token_ids)
    # print(token_ids.tokenized_states[3].graph_token_ids)

    text_ids = list(map(list, token_ids.tokenized_states))

    print(text_ids)

    # for batch in train_dataloader:
    #     break
    # print({k: v.shape for k, v in batch.text_token_ids})





    # Dataloader to iterate over Dataset
    ###DataLoader(token_ids)
    # bert-base-uncased with the paramteres
    ###text_encoder_configuration = BertConfig(
    ###   num_hidden_layers=6,
    ###    num_attention_heads=6,
    ###    feed=3072,
    ###)
    ###model = BertModel.from_pretrained("bert-base-uncased", text_encoder_configuration)

    ###data_loader = DataLoader(token_ids, batch_size=10).__iter__()
    ###epochs = 10 # ai durchläufe
    ###for epoch in range(epochs):
    ###    for batch in data_loader:
    ###        # move batch to gpu
    ###        model(batch)


    # data visualizing
    ###pass
    # for data in preprocessed_data_set.preprocessed_states[1:2]:
    #     print("")
    #     print("-------------------")
    #     print("")
    #     print("text: " + data.concatenatedText)
    #     print("graph: " + data.graphWithTriples)

# print(pythonCustomObj.action)
# print(pythonCustomObj.state.location.name)
# print(pythonCustomObj.state.location.name)


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
#
# last_hidden_states = outputs.last_hidden_state
