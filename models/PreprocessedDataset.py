from typing import List
from torch.utils.data import Dataset
from models.PreprocessedState import PreprocessedState


class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_state):
        self.preprocessed_state: List[PreprocessedState] = preprocessed_state

    def __len__(self):
        return len(self.preprocessed_state)

    def __getitem__(self, index):
        return self.preprocessed_state[index]

    def append(self, nee_item: PreprocessedState):
        return self.preprocessed_state.append(nee_item)
