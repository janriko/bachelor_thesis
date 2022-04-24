from typing import List


class TokenizedState:
    def __init__(self, text_token_ids, graph_token_ids):
        self.text_token_ids: List[int] = text_token_ids
        self.graph_token_ids: List[int] = graph_token_ids
