from cs336_basics.tokenizer.utils import (
    pre_tokenize,
    gpt2_bytes_to_unicode,
)
from cs336_basics.tokenizer.bpe import train_bpe
import json
from tqdm import tqdm
import os
import numpy as np


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: tuple[tuple[int, int], int],
        special_tokens: list[bytes] = None,
    ):
        self.vocab = vocab
        self.re_vocab = {v: k for k, v in vocab.items()}
        if isinstance(merges[0][0], bytes):
            new_merges = (
                (
                    (self.re_vocab[merge[0]], self.re_vocab[merge[1]]),
                    self.re_vocab[merge[0] + merge[1]],
                )
                for merge in merges
            )
            merges = new_merges

        self.merges = {merge[0]: merge[1] for merge in merges}
        self.special_tokens = special_tokens if special_tokens is not None else []

    def encode(self, text: str) -> list[int]:
        res = []

        for tokens in pre_tokenize(text, self.special_tokens, True):
            if (
                len(tokens) == 1
                and isinstance(tokens[0], str)
                and tokens[0] in self.special_tokens
            ):
                res.append(self.re_vocab[tokens[0].encode("utf-8")])
                continue

            ids = [self.re_vocab[bytes([b])] for b in tokens]
            while len(ids) >= 2:
                best_pair = None
                best_rank = float("inf")

                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self.merges:

                        rank = self.merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break

                write_id, read_id = 0, 0
                while read_id < len(ids):
                    if (
                        read_id < len(ids) - 1
                        and (ids[read_id], ids[read_id + 1]) == best_pair
                    ):
                        ids[write_id] = self.merges[best_pair]
                        write_id += 1
                        read_id += 2
                    else:
                        ids[write_id] = ids[read_id]
                        write_id += 1
                        read_id += 1

                del ids[write_id:]

            res += ids

        return res

    def decode(self, ids: list[int]) -> str:
        byte_array = b"".join(self.vocab[id] for id in ids)

        return byte_array.decode("utf-8", errors="replace")

    def encode_iterable(self, texts):
        for text in texts:
            ids = self.encode(text)
            yield from ids

    @classmethod
    def get_tokenizer_from_vocab_merges_path(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes(
                [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
            )
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        pass
        vocab = {}
        merges = []

        with open(
            vocab_filepath,
            "r",
            encoding="utf-8",
            errors="surrogateescape",
        ) as f:
            vocab = json.load(f)

        my_vocab = {
            int(v): k.encode("utf-8", errors="surrogateescape")
            for k, v in vocab.items()
        }

        with open(merges_filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                a, b = map(str, line.split())
                merges.append(((vocab[a], vocab[b]), vocab[a + b]))

        return cls(my_vocab, merges, special_tokens)


def encode_file_to_bin(tokenizer, text_path, out_bin_path, dtype=np.uint16):
    total_bytes = os.path.getsize(text_path)

    with open(text_path, encoding="utf-8") as f_in, open(out_bin_path, "wb") as f_out:
        p_bar = tqdm(
            total=total_bytes, desc="Encoding to binary", unit="B", unit_scale=True
        )

        for line in f_in:
            token_ids = tokenizer.encode(line)
            arr = np.array(token_ids, dtype=dtype)
            arr.tofile(f_out)

            p_bar.update(len(line.encode("utf-8")))


def load_tokenizer_from_dir(dir_path: str) -> Tokenizer:
    vocab_path = os.path.join(dir_path, "vocab.json")
    merges_path = os.path.join(dir_path, "merges.txt")
    special_tokens_path = os.path.join(dir_path, "special_tokens.txt")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)
    return tokenizer


if __name__ == "__main__":
    merges_path = "/home/chioca/assignment1-basics/tests/fixtures/gpt2_merges.txt"
    vocab_path = "/home/chioca/assignment1-basics/tests/fixtures/gpt2_vocab.json"
    tokenizer = Tokenizer.get_tokenizer_from_vocab_merges_path(
        vocab_path, merges_path, ["<|endoftext|>"]
    )
    encode_file_to_bin(
        tokenizer, "/home/chioca/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "gpt_out_eval.bin"
    )
    ids = tokenizer.encode("I like to sing songs")
    print(tokenizer.decode(ids))
