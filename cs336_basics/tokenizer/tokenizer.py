from cs336_basics.tokenizer.utils import stream_pretokenize, pre_tokenize
import json


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: tuple[tuple[int, int], int],
        special_tokens: list[bytes] = None,
    ):
        self.vocab = vocab
        # self.merges = merges
        self.merges = {merge[0]: merge[1] for merge in merges}
        self.special_tokens = special_tokens
        self.re_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        res = []
        for tokens in pre_tokenize(text, self.special_tokens, False):
            ids = [self.re_vocab[bytes([b])] for b in tokens]

            i = 0
            while i < len(ids) - 1:
                cur_pair = (ids[i], ids[i + 1])
                replace = self.merges.get(cur_pair)
                if replace is None:
                    i += 1
                    continue

                ids[i] = replace
                ids.pop(i + 1)

            res += ids

        return res

    def decode(self, ids: list[int]) -> str:
        res = ""
        for id in ids:
            res += self.vocab[id].decode("utf-8")

        return res

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

        vocab = {
            int(k): v.encode("utf-8", errors="surrogateescape")
            for k, v in vocab.items()
        }

        with open(merges_filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                a, b, c = map(int, line.split())
                merges.append(((a, b), c))

        return cls(vocab, merges, special_tokens)


if __name__ == "__main__":
    merges_path = "/home/chioca/assignment1-basics/merges.txt"
    vocab_path = "/home/chioca/assignment1-basics/vocab.json"
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
    )
    ids = tokenizer.encode("I like to sing songs")
    print(tokenizer.decode(ids))
