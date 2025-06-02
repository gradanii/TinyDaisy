import numpy as np
from .expert import forward, softmax


def CrossEntropyLoss(cfg, strings):
    tokens = [cfg.tokenizer.encode(s) for s in strings]

    max_len = max([len(t) for t in tokens])

    padded_tokens = []
    for token in tokens:
        pad_token = np.pad(token, (0, max_len - len(token)), constant_values=0)
        padded_tokens.append(pad_token)

    tok_array = np.stack(padded_tokens)
    targets = tok_array[:, 1:, np.newaxis]
    inputs = tok_array[:, :-1]

    logits = forward(cfg, inputs)

    probs = softmax(logits, axis=-1)
    correct_probs = np.take_along_axis(probs, targets, axis=-1)
    correct_probs = np.squeeze(correct_probs, axis=-1)

    neg_log_prob = -1 * np.log(correct_probs)

    mask = inputs != 0
    masked_loss = neg_log_prob * mask
    loss = np.sum(masked_loss) / np.sum(mask)

    return loss
