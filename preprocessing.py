from typing import List, Dict
from models import *
from models.preprocessing.PreprocessedDataset import PreprocessedDataset
from models.raw_state.JerichoDataset import JerichoDataset
from models.raw_state.JerichoState import JerichoState
from models.raw_state.JerichoStateTransition import JerichoStateTransition
from models.raw_state.JerichoTransitionList import JerichoTransitionList


def map_json_to_python_obj(pythonObj: dict) -> JerichoDataset:
    all_state_transitions: JerichoDataset = JerichoDataset(worldList=JerichoTransitionList(list()))
    for liste_index in range(len(pythonObj)):
        all_state_transitions.append(JerichoTransitionList(transitionList=list()))
        for state_trans_index in range(len(pythonObj[liste_index])):
            all_state_transitions[liste_index].append(
                JerichoStateTransition(**(pythonObj[liste_index][state_trans_index])))
    return all_state_transitions


def map_all_state_transitions_to_preprocessed_state(dataset: JerichoDataset) -> PreprocessedDataset:
    pre_processed_list: PreprocessedDataset = PreprocessedDataset(preprocessed_state=list())
    for listIndex in range(dataset.__len__()):
        for stateTransIndex in range((dataset[listIndex]).__len__()):
            pre_processed_list.append(
                PreprocessedState(
                    concatenate_state_to_text_encoder_string(dataset[listIndex][stateTransIndex].state),
                    concatenate_state_to_graph_encoder_string(dataset[listIndex][stateTransIndex].state.graph)
                )
            )
    return pre_processed_list


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
