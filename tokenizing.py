from typing import List, Dict

import dataset
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BatchEncoding
from transformers.file_utils import PaddingStrategy

from data_models.preprocessing.PreprocessedDataset import PreprocessedDataset
from data_models.tokenized.TokenizedDataset import TokenizedDataset

# from data_models.tokenized.TokenizedState import TokenizedState

special_text_tokens: Dict[str, str] = {
    "cls_token": "[OBS]",
    "sep_token": "[ACT]",
}

special_graph_tokens: Dict[str, str] = {
    "cls_token": "[GRAPH]",
    "sep_token": "[TRIPLE]",
}

text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
graph_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # TODO: import correct graph tokenizer
# print(preprocessed_data_set.preprocessed_states[0].concatenatedText)

# add custom tokens
text_tokenizer.add_special_tokens(special_tokens_dict=special_text_tokens)
graph_tokenizer.add_special_tokens(special_tokens_dict=special_graph_tokens)


def text_tokenize_function(state):
    return text_tokenizer(
        text=state["text"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=1024,
        # return_tensors="pt"
    )


def graph_tokenize_function(state):
    return graph_tokenizer(
        text=state["graph"],
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=True,
        max_length=1024,
       # return_tensors="pt"
    )


def tokenize_text_and_graph(preprocessed_dataset: DatasetDict) -> DatasetDict:
    tokenized_datasets = preprocessed_dataset.map(text_tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("input_ids", "t_input_ids")
    tokenized_datasets = tokenized_datasets.rename_column("token_type_ids", "t_token_type_ids")
    tokenized_datasets = tokenized_datasets.rename_column("attention_mask", "t_attention_mask")

    tokenized_datasets = tokenized_datasets.map(graph_tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("input_ids", "g_input_ids")
    tokenized_datasets = tokenized_datasets.rename_column("token_type_ids", "g_token_type_ids")
    tokenized_datasets = tokenized_datasets.rename_column("attention_mask", "g_attention_mask")

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.remove_columns(["graph"])

    tokenized_datasets.set_format("torch")

    return tokenized_datasets
