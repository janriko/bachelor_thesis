from typing import List, Union
import ast

from datasets.arrow_dataset import Dataset


class GraphTokenizer:
    tokens = ast.literal_eval(open("create_vocab/graph_encoder_vocabulary_training.txt", "r").read())

    def __init__(self):
        pass

    def token_to_id(self, token: str) -> id:
        pass

    def convert_triple_to_ids(self, triple: str) -> List[id]:
        two_ids: List[id] = []
        first_token = triple.split("]")
        return two_ids

    def encode(self, text: Dataset) -> Dataset:
        text_list = text["graph"]
        input_ids: List[List[id]] = [[]]
        attention_mask: List[id] = []

        for state_index in range(len(text_list)):
            attention_mask_counter = 0
            while len(text_list[state_index]) > 0:
                attention_mask_counter += 1
                index_of_next_triple = text_list[state_index].find('[')
                # if state doesn't contain any more triples, take last triple
                if index_of_next_triple == -1:
                    index_of_next_triple = len(text)
                # add tokenized triple to list
                input_ids[state_index].append(text[0:index_of_next_triple])
                # remove just tokenized triple
                text_list[state_index] = text[index_of_next_triple:]
            attention_mask.append(attention_mask_counter)
        new_dataset = text.add_column("input_ids", input_ids)
        new_dataset = new_dataset.add_column("attention_mask", attention_mask)
        new_dataset = new_dataset.add_column("token_type_ids", attention_mask)
        return new_dataset

    def decode(self, words: List[int]):
        pass
