import sys
import os
import unittest

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        tokenizer = BPETokenizer(vocab_size=300)
        text = "hello there, banana!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        self.assertEqual(decoded, text)

    def test_new_tokens_exist(self):
        tokenizer = BPETokenizer(vocab_size=270)
        tokenizer.encode("banana banana banana")

        new_tokens = list(tokenizer.inv_vocab.keys())[256:]
        self.assertGreater(len(new_tokens), 0)

    def test_utf8_emoji_support(self):
        tokenizer = BPETokenizer(vocab_size=400)
        text = "🚀🔥😎"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        self.assertEqual(decoded, text)


if __name__ == "__main__":
    unittest.main()
