import collections


class tokenizer:
    def __init__(self, vocab=None, vocab_size=257):
        self.inv_vocab = vocab or {i: bytes([i]) for i in range(256)}
        self.byte_vocab = {v: k for k, v in self.inv_vocab.items()}
        self.new_token = max(self.inv_vocab.keys(), default=-1) + 1
        self.vocab_size = vocab_size

    def encoder(self, string):
        tokens = list(string.encode("utf-8"))

        while len(self.inv_vocab) < self.vocab_size:
            pairs = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
            frequency = collections.Counter(pairs)

            if frequency.most_common(1)[0][1] == 1:
                break

            mfp = frequency.most_common(1)[0][0]
            word = b"".join(self.inv_vocab[i] for i in mfp)

            ntokens = []
            self.byte_vocab[word] = self.new_token
            self.inv_vocab[self.new_token] = word

            i = 0
            while i < len(tokens):
                pair = tuple(tokens[i : i + 2])
                if pair == mfp:
                    ntokens.append(self.byte_vocab[word])
                    i += 2
                else:
                    ntokens.append(tokens[i])
                    i += 1

            self.new_token += 1

            tokens = ntokens

        return tokens

    def decoder(self, tokens):
        for token in tokens:
            if token not in self.inv_vocab:
                print("❌ Missing token in inv_vocab:", token)

        string = b"".join(self.inv_vocab[token] for token in tokens)

        return string.decode("utf-8")
