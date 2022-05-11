import torch
from numpy.distutils.cpuinfo import cpu
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from transformers import PretrainedConfig, BertModel, get_scheduler
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

config = PretrainedConfig(
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


def get_graph_encoder_data(num_training_steps) -> (BertModel, AdamW, cpu, LambdaLR, tqdm):
    pass


def graph_encoder_train_loop(batch_dataset, text_encoder_model, text_encoder_optimizer, device, lr_scheduler, progress_bar) -> BaseModelOutputWithPoolingAndCrossAttentions:
    pass
