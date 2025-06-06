import jax.numpy as jnp
from src.tokenizer import BPETokenizer
from src.model.model import forward, softmax


def CrossEntropyLoss(cfg, params, data):
    tokenizer = BPETokenizer(vocab_size=cfg["vocab_size"])
    tokenized = [tokenizer.encode(d) for d in data]

    max_len = max(len(t) for t in tokenized)
    pad_id = 0
    tokens = jnp.array(
        [
            jnp.pad(
                jnp.array(token),
                pad_width=(0, max_len - len(token)),
                constant_values=pad_id,
            )
            for token in tokenized
        ]
    )

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:, jnp.newaxis]
    logits = forward(cfg, params, inputs)

    probs = softmax(logits)
    corrected_probs = jnp.take_along_axis(probs, targets, axis=-1).squeeze(-1)
    neg_log_prob = -jnp.log(corrected_probs + 1e-9)

    mask = inputs != pad_id
    masked_probs = neg_log_prob * mask
    loss = jnp.sum(masked_probs) / jnp.sum(mask)

    return loss
