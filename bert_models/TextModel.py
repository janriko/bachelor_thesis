import torch
from numpy.distutils.cpuinfo import cpu
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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


def get_text_encoder_data(num_training_steps) -> (BertModel, AdamW, cpu, LambdaLR, tqdm):
    text_encoder_model: BertModel = BertModel.from_pretrained("bert-base-uncased", config=config)
    optimizer: AdamW = AdamW(text_encoder_model.parameters(), lr=3e-4)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device: cpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    progress_bar = tqdm(range(num_training_steps))

    text_encoder_model.to(device)
    text_encoder_model.train()

    return text_encoder_model, optimizer, device, lr_scheduler, progress_bar


def text_encoder_train_loop(batch_dataset, text_encoder_model, text_encoder_optimizer, device, lr_scheduler, progress_bar) -> BaseModelOutputWithPoolingAndCrossAttentions:
    batch_dataset = {k: v.to(device) for k, v in batch_dataset.items()}
    outputs = text_encoder_model(**batch_dataset)
    loss = outputs.loss
    loss.backward()

    text_encoder_optimizer.step()
    lr_scheduler.step()
    text_encoder_optimizer.zero_grad()
    progress_bar.update(1)

    return outputs
