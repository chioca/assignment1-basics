from cs336_basics.tokenizer.utils import pre_tokenize
import json


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
            if len(tokens) == 1 and tokens[0] in self.special_tokens:
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

                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                        new_ids.append(self.merges[best_pair])
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids

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
