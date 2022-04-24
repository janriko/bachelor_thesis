from typing import List, Dict

from transformers import BertTokenizer, BatchEncoding

from models.preprocessing.PreprocessedDataset import PreprocessedDataset
from models.tokenized.TokenizedDataset import TokenizedDataset
from models.tokenized.TokenizedState import TokenizedState

special_text_tokens: Dict[str, str] = {
    "cls_token": "[OBS]",
    "sep_token": "[ACT]",
}

special_graph_tokens: Dict[str, str] = {
    "cls_token": "[GRAPH]",
    "sep_token": "[TRIPLE]",
}


def tokenize_text_and_graph(preprocessed_data_set: PreprocessedDataset) -> TokenizedDataset:
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    graph_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # TODO: import correct graph tokenizer
    print(preprocessed_data_set.preprocessed_states[0].concatenatedText)

    # add custom tokens
    text_tokenizer.add_special_tokens(special_tokens_dict=special_text_tokens)
    graph_tokenizer.add_special_tokens(special_tokens_dict=special_graph_tokens)

    tokenized_dataset = TokenizedDataset([])

    for index in range(len(preprocessed_data_set.preprocessed_states)):
        # text
        text_model_inputs = text_tokenizer(
            text=preprocessed_data_set.preprocessed_states[index].concatenatedText,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        text_input_ids: List[int] = text_model_inputs.input_ids[0]
        text_token_type_ids: List[int] = text_model_inputs.token_type_ids[0]
        text_attention_mask: List[int] = text_model_inputs.attention_mask[0]

        # if index == 3:
        #     print(text_input_ids)
        #     print(text_token_type_ids)
        #     print(text_attention_mask)

        # graph
        graph_tokenizer_returner = graph_tokenizer(
            text=preprocessed_data_set.preprocessed_states[index].graphWithTriples,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        graph_input_ids: List[int] = graph_tokenizer_returner.input_ids[0]
        graph_token_type_ids: List[int] = graph_tokenizer_returner.token_type_ids[0]
        graph_attention_mask: List[int] = graph_tokenizer_returner.attention_mask[0]

        tokenized_dataset.append(TokenizedState(
            text_token_ids=text_input_ids,
            graph_token_ids=graph_input_ids
        ))

    # print("seperated text tokens")
    # print(text_tokenizer.convert_ids_to_tokens(tokenized_dataset[0].text_token_ids[0]))
    # print("text token id's")
    # print(tokenized_dataset[0].text_token_ids[0])

    # print("seperated graph tokens")
    # print(graph_tokenizer.convert_ids_to_tokens(tokenized_dataset[1].graph_token_ids[0]))
    # print("graph token id's")
    # print(tokenized_dataset[1].graph_token_ids[0])

    return tokenized_dataset
