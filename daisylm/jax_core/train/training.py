import jax
import optax
import jax.numpy as jnp
from jax_core.tokenizer import BPETokenizer
from jax.random import PRNGKey
from jax_core.model.params import init_params
from jax_core.losses import CrossEntropyLoss
from daisylm.config import DAISY_CONFIG

cfg = DAISY_CONFIG
params = init_params(PRNGKey(42), cfg)

schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=2000,
    decay_steps=100000,
    end_value=1e-5,
)

optimizer = optax.adamw(schedule_fn)
opt_state = optimizer.init(params)

def preprocess_data(batch):
    tokenizer = BPETokenizer(vocab_size=cfg["vocab_size"])
    tokenized = [tokenizer.encode(d) for d in batch]

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
    return inputs, targets

@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(CrossEntropyLoss)(cfg, params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
