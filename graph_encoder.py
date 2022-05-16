from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig

import file_loader
import g_config
import preprocessing
from file_loader import FileLoader
from tokenizer.GraphTokenizer import GraphTokenizer

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_attention_heads=6,
    num_hidden_layers=6,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    position_embedding_type="absolute",
    use_cache=True,
    classifier_dropout=None,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)


if __name__ == "__main__":

    train_dataset: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=True, file_name="JerichoWorld-main/data/small_training_set.json")
    eval_dataset: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=False, file_name="JerichoWorld-main/data/small_test_set.json")

    train_dataset = FileLoader.remove_unnecessary_columns(train_dataset)
    eval_dataset = FileLoader.remove_unnecessary_columns(eval_dataset)

    # dataset_dict = DatasetDict({"train": train_dataset, "test": eval_dataset})

    encoder = BertModel(config=config)
    tokenizer = GraphTokenizer()

    tokenizer.encode(train_dataset)


    tokenized_datasets: DatasetDict = FileLoader.get_tokenized_text_and_graph_ids()
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=16)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=16)

    num_epochs = g_config.num_epochs
    num_training_steps = g_config.num_training_steps(train_dataloader)

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


