from typing import List, Dict

from datasets import Dataset, DatasetDict
from datasets.features import features

import file_loader
from data_models import *
from data_models.preprocessing.PreprocessedDataset import PreprocessedDataset
from data_models.raw_state.JerichoDataset import JerichoDataset
from data_models.raw_state.JerichoState import JerichoState
from data_models.raw_state.JerichoStateTransition import JerichoStateTransition
from data_models.raw_state.JerichoTransitionList import JerichoTransitionList

def map_json_to_python_obj(pythonObj: dict) -> JerichoDataset:
    all_state_transitions: JerichoDataset = JerichoDataset(worldList=JerichoTransitionList(list()))
    for liste_index in range(len(pythonObj)):
        all_state_transitions.append(JerichoTransitionList(transitionList=list()))
        for state_trans_index in range(len(pythonObj[liste_index])):
            all_state_transitions[liste_index].append(
                JerichoStateTransition(**(pythonObj[liste_index][state_trans_index])))
    return all_state_transitions


def map_all_state_transitions_to_preprocessed_state(dataset: JerichoDataset) -> Dataset:
    # pre_processed_list: PreprocessedDataset = PreprocessedDataset(preprocessed_state=list())
    pre_processed_dict = {
        "labels": [],
        "text": [],
        "graph": []
    }
    for listIndex in range(dataset.__len__()):
        for stateTransIndex in range((dataset[listIndex]).__len__()):
            # pre_processed_list.append(
            #     PreprocessedState(
            pre_processed_dict["labels"].append(listIndex*dataset.__len__()+stateTransIndex)
            pre_processed_dict["text"].append(concatenate_state_to_text_encoder_string(dataset[listIndex][stateTransIndex].state))
            pre_processed_dict["graph"].append(concatenate_state_to_graph_encoder_string(dataset[listIndex][stateTransIndex].state.graph))
            #     )
            # )
    return Dataset.from_dict(pre_processed_dict)


def concatenate_state_to_text_encoder_string(state: JerichoState) -> PreprocessedState:
    tags: Dict[str] = {
        "observableText": "[OBS]",
        "validActs": "[ACT]"
    }
    return_str = tags["observableText"] + state.obs.replace("\n", "")

    if state.valid_acts.__len__() > 0:
        for valid_act in state.valid_acts.values():
            return_str = return_str + tags["validActs"] + valid_act
    else:
        return_str = ""

    return return_str


def concatenate_graph_triple(tag: str, graph_entry: List[str]):
    return tag + graph_entry[0] + "." + graph_entry[1] + "." + graph_entry[2]


def concatenate_state_to_graph_encoder_string(graph: List[List[str]]):
    tags: Dict[str] = {
        "startingTag": "[GRAPH]",
        "inbetweenTag": "[TRIPLE]"
    }

    if graph.__len__() > 0:
        return_str = concatenate_graph_triple(tags["startingTag"], graph[0])

        for graph_entry in graph[1:]:
            return_str = return_str + concatenate_graph_triple(tags["inbetweenTag"], graph_entry)
    else:
        return_str = ""

    return return_str


def preprocessing() -> (DatasetDict, DatasetDict):
    """
    either get preprocessed dataset from cache (give file name as parameter)
    or recompile dataset from file (give filename with jericho dataset)
    """
    preprocessed_training_data_set: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=True, file_name="JerichoWorld-main/data/small_training_set.json")
    preprocessed_testing_data_set: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=False, file_name="JerichoWorld-main/data/small_test_set.json")

    dataset_dict = DatasetDict({"train": preprocessed_training_data_set, "test": preprocessed_testing_data_set})

    """
    create tokenized ids with dataset (give dataset)
    or load from cache (don't give dataset as parameter)
    """
    tokenized_datasets: DatasetDict = file_loader.get_tokenized_text_and_graph_ids(dataset_dict)

    # print(token_ids.get("train")["t_input_ids"])
    # print(token_ids.get("train")["t_attention_mask"])

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    return train_dataset, eval_dataset
