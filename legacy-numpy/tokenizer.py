import collections
import json


class BPETokenizer:
    def __init__(self, vocab=None, vocab_size=256):
        self.inv_vocab = vocab or {i: bytes([i]) for i in range(256)}
        self.byte_vocab = {v: k for k, v in self.inv_vocab.items()}
        self.new_token = max(self.inv_vocab.keys(), default=-1) + 1
        self.vocab_size = vocab_size
        self.frozen = True
        self.merges = []

    def encode(self, string):
        tokens = list(string.encode("utf-8"))

        if getattr(self, "frozen", False):
            for mfp in self.merges:
                i = 0
                ntokens = []
                while i < len(tokens):
                    pair = tuple(tokens[i : i + 2])
                    if pair == mfp:
                        merged = b"".join(self.inv_vocab[j] for j in pair)
                        ntokens.append(self.byte_vocab[merged])
                        i += 2
                    else:
                        ntokens.append(tokens[i])
                        i += 1
                tokens = ntokens
            return tokens

        while len(self.inv_vocab) < self.vocab_size:
            pairs = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
            frequency = collections.Counter(pairs)

            if frequency.most_common(1)[0][1] == 1:
                break

            mfp = frequency.most_common(1)[0][0]
            self.merges.append(mfp)

            word = b"".join(self.inv_vocab[i] for i in mfp)
            self.byte_vocab[word] = self.new_token
            self.inv_vocab[self.new_token] = word

            ntokens = []
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

    def decode(self, tokens):
        for token in tokens:
            if token not in self.inv_vocab:
                print("❌ Missing token in inv_vocab:", token)

        string = b"".join(self.inv_vocab[token] for token in tokens)

        return string.decode("utf-8")

    def save_vocab(self, path="vocab.json"):
        json_vocab = {
            str(k): v.decode("utf-8", errors="replace")
            for k, v in self.inv_vocab.items()
        }
        with open(path, "w") as f:
            json.dump(json_vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path="vocab.json"):
        with open(path, "r") as f:
            json_vocab = json.load(f)
        self.inv_vocab = {int(k): v.encode("utf-8") for k, v in json_vocab.items()}
        self.byte_vocab = {v: k for k, v in self.inv_vocab.items()}
        self.new_token = max(self.inv_vocab.keys(), default=-1) + 1

    def save_merges(self, path="merges.txt"):
        with open(path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                a_str = self.inv_vocab[a].decode("utf-8", errors="replace")
                b_str = self.inv_vocab[b].decode("utf-8", errors="replace")
                f.write(f"{a_str} {b_str}\n")

    def load_merges(self, path="merges.txt"):
        with open(path, "r", encoding="utf-8") as f:
            self.merges = []
            for line in f:
                a_str, b_str = line.strip().split()
                a_bytes = a_str.encode("utf-8")
                b_bytes = b_str.encode("utf-8")
                a = self.byte_vocab.get(a_bytes)
                b = self.byte_vocab.get(b_bytes)
                if a is not None and b is not None:
                    self.merges.append((a, b))
