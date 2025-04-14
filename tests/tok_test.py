import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer import tokenizer

tokenizer = tokenizer()
text = "hello world"
tokens = tokenizer.encoder(text)
og_string = tokenizer.decoder(tokens)

print("Original: ", text)
print("Tokens: ", tokens)
print("Reconstructed: ", og_string)
