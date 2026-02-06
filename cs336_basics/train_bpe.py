from cs336_basics.tokenizer.tokenizer import train_bpe
from pathlib import Path

input_path = Path(__file__).parent.parent / "data" / "test.txt"
special_tokens = ["<|endoftext|>"]
vocab_size = 599
vocab, merges = train_bpe(
    input_path.absolute(), vocab_size, special_tokens, True, desired_num_chunks=16
)

# print(vocab)
print(merges)
