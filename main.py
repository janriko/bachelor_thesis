import random

import transformers
from typing import Dict, List, TextIO
import json

from datasets import DatasetDict
from torch import LongTensor, tensor
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel, BertConfig, PretrainedConfig, get_scheduler
import torch

import config
import file_loader.FileLoader
import datasets

from Aggregator.Aggregator import Aggregator
from Decoder.ActionDecoder import ActionDecoder
from Decoder.GraphDecoder import GraphDecoder
from bert_models import TextModel, GraphModel
from data_models import *
import preprocessing
import pickle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from transformers import logging
from data_models.preprocessing.PreprocessedDataset import PreprocessedDataset
from data_models.raw_state.JerichoDataset import JerichoDataset
from data_models.tokenized.TokenizedDataset import TokenizedDataset

if __name__ == "__main__":
    logging.set_verbosity_error()

    train_dataset, eval_dataset = preprocessing.preprocessing()

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=16)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=16)

    num_epochs = config.num_epochs
    num_training_steps = config.num_training_steps(train_dataloader)
    text_encoder_model, text_encoder_optimizer, device, lr_schedular, progress_bar = TextModel.get_text_encoder_data(num_training_steps)

    aggregator = Aggregator()
    action_decoder = ActionDecoder()
    graph_decoder = GraphDecoder()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # separate batches
            text_batch: Dict = {key: value for key, value in batch.items() if (key[0] == "t") or (key[0] == "l")}
            graph_batch: Dict = {key: value for key, value in batch.items() if (key[0] == "g") or (key[0] == "l")}

            # rename batch keys
            text_batch["input_ids"] = text_batch.pop("t_input_ids")
            text_batch["token_type_ids"] = text_batch.pop("t_token_type_ids")
            text_batch["attention_mask"] = text_batch.pop("t_attention_mask")

            graph_batch["input_ids"] = graph_batch.pop("g_input_ids")
            graph_batch["token_type_ids"] = graph_batch.pop("g_token_type_ids")
            graph_batch["attention_mask"] = graph_batch.pop("g_attention_mask")

            text_outputs = TextModel.text_encoder_train_loop(
                batch_dataset=text_batch,
                text_encoder_model=text_encoder_model,
                text_encoder_optimizer=text_encoder_optimizer,
                device=device,
                lr_scheduler=lr_schedular,
                progress_bar=progress_bar
            )
            graph_outputs = GraphModel.graph_encoder_train_loop(
                batch_dataset=graph_batch,
                text_encoder_model=text_encoder_model,
                text_encoder_optimizer=text_encoder_optimizer,
                device=device,
                lr_scheduler=lr_schedular,
                progress_bar=progress_bar
            )

            # aggregator_outputs = hidden-S_t
            aggregator_outputs = aggregator.aggregate(text_outputs, graph_outputs)

            next_step = action_decoder.decode(text_outputs, aggregator_outputs)
            graph_decoder.decode(graph_outputs, aggregator_outputs)

            break
        break
