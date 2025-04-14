import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=1024)

print(tokenizer.encode("banana banana banana banana"))
