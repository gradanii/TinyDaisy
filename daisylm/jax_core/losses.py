import jax.numpy as jnp
from jax_core.model.model import forward, softmax


def CrossEntropyLoss(cfg, params, batch):
    inputs, targets = batch
    logits = forward(cfg, params, inputs)

    probs = softmax(logits)
    corrected_probs = jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)
    neg_log_prob = -jnp.log(corrected_probs + 1e-9)

    mask = inputs != 0
    masked_probs = neg_log_prob * mask
    loss = jnp.sum(masked_probs) / jnp.sum(mask)

    return loss
