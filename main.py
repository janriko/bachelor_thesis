import random

import transformers
from typing import Dict, List
import json
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from models import *
import preprocessing
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

if __name__ == "__main__":
    json_string = open("JerichoWorld-main/data/small_training_set.json", "r")
    python_obj: dict = json.load(json_string)
    custom_jericho_dataset: JerichoDataset = preprocessing.map_json_to_python_obj(python_obj)

    preprocessed_data_set = preprocessing.map_all_state_transitions_to_preprocessed_state(custom_jericho_dataset)

    for data in preprocessed_data_set:
        print("text: " + data.concatenatedText)
        print("graph: " + data.graphWithTriples)
        print("")
        print("-------------------")
        print("")


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
