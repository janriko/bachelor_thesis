from typing import List
from torch.utils.data import Dataset
from models.tokenized.TokenizedState import TokenizedState


class TokenizedDataset(Dataset):
    def __init__(self, tokenized_states):
        self.tokenized_states: List[TokenizedState] = tokenized_states

    def __len__(self):
        return len(self.tokenized_states)

    def __getitem__(self, index):
        return self.tokenized_states[index]

    def append(self, tokenized_state: TokenizedState):
        self.tokenized_states.append(tokenized_state)
