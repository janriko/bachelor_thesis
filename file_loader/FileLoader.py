from io import StringIO
from typing import Generic, TypeVar, List

import json
import pickle

import preprocessing
from models.preprocessing.PreprocessedState import PreprocessedState
from models.preprocessing.PreprocessedDataset import PreprocessedDataset
from models.raw_state.JerichoDataset import JerichoDataset
from models.tokenized.TokenizedDataset import TokenizedDataset

# preprocessed_dataset_text_file = 'cached_data/PreprocessedDataset/preprocessed_states/concatenatedText.txt'
# preprocessed_dataset_graph_file = 'cached_data/PreprocessedDataset/preprocessed_states/graphWithTriples.txt'
from tokenizing import tokenize_text_and_graph

preprocessed_dataset_file = 'cached_data/preprocessed_dataset'
tokenized_dataset_file = 'cached_data/tokenized_dataset'


# gets tokenized text [O_t + V_t] and tokenized graph [G_t]
# If a filename is given -> re preprocess it from file and save to cache file
#                   else -> use cached data
# O_t:
#       [OBS] West of House You are standing in an open field west of a white house...
# V_t:
#       [ACT] go north [ACT] open mailbox ...
# G_t:
#       [GRAPH] you, in, West of House [TRIPLE] West of house, has, mailbox [TRIPLE] mailbox, is, openable ...
def get_preprocessed_and_tokenized_text_and_graph(file_name: str = None) -> PreprocessedDataset:
    if file_name is None:
        # load from cached file
        preprocessed_dataset: PreprocessedDataset = pickle.load(open(preprocessed_dataset_file, 'rb'))
    else:
        json_string = open(file_name, "r")

        # pre-processing
        python_obj: dict = json.load(json_string)
        custom_jericho_dataset: JerichoDataset = preprocessing.map_json_to_python_obj(python_obj)

        # create pre processed dataset:
        preprocessed_dataset: PreprocessedDataset = preprocessing.map_all_state_transitions_to_preprocessed_state(dataset=custom_jericho_dataset)

        # save dataset to cache file
        pickle.dump(preprocessed_dataset, open(preprocessed_dataset_file, 'wb'))

    return preprocessed_dataset


def get_tokenized_text_and_graph_ids(preprocessed_data_set: PreprocessedDataset = None) -> TokenizedDataset:
    if preprocessed_data_set is None:
        # load from cached file
        tokenized_dataset: TokenizedDataset = pickle.load(open(tokenized_dataset_file, 'rb'))
    else:
        tokenized_dataset = tokenize_text_and_graph(preprocessed_data_set)
        # save dataset to cache file
        pickle.dump(tokenized_dataset, open(tokenized_dataset_file, 'wb'))

    return tokenized_dataset
