import regex as re
from pathlib import Path
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import multiprocessing as mp
import os
from os import cpu_count
import mmap

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PATTERN = re.compile(PAT)
# test = "some text that i'll pre-tokenize"
# out = re.findall(PAT, test)
# print(out)


def cnt_pair(word_cnt: dict[tuple[bytes], int]) -> Counter:
    pair_cnt = Counter()

    for word, cnt in word_cnt.items():
        for pair in zip(word[:-1], word[1:]):
            pair_cnt[pair] += cnt

    return pair_cnt


def string_to_bytes(s: str, return_int: bool = False) -> list[int] | list[bytes]:
    byte_array = s.encode("utf-8")
    return (
        list(map(int, byte_array)) if return_int else [bytes([b]) for b in byte_array]
    )


def split_by_special_tokens(
    text: str, special_tokens: list[str], include_special: bool = False
) -> list[str]:
    if not special_tokens:
        return [text]

    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if include_special:
        special_chunks = re.split(f"({pattern})", text)
    else:
        # Split without capturing the special tokens
        special_chunks = re.split(pattern, text)

    return special_chunks


def pre_tokenize(
    string: str, special_tokens: list[str] = None, including_special: bool = False
):

    chunks = split_by_special_tokens(
        string, special_tokens, include_special=including_special
    )
    special_tokens = [] if special_tokens is None else special_tokens
    for chunk in chunks:
        if including_special and chunk in special_tokens:
            yield (chunk,)

        else:
            for match in PATTERN.finditer(chunk):
                yield tuple(match.group(0).encode("utf-8"))


_worker_mm = None
_worker_special_tokens = None


def init_worker(input_path, special_tokens):
    """
    每个 worker 初始化 mmap
    """

    global _worker_mm
    global _worker_special_tokens

    f = open(input_path, "rb")

    _worker_mm = mmap.mmap(
        f.fileno(),
        0,
        access=mmap.ACCESS_READ,
    )

    _worker_special_tokens = special_tokens


def process_chunk(args):
    """
    处理单个 chunk
    """

    start, end = args

    global _worker_mm
    global _worker_special_tokens

    chunk = _worker_mm[start:end]

    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        text = chunk.decode("utf-8", errors="ignore")

    local_counter = defaultdict(int)

    for tok in pre_tokenize(
        text,
        _worker_special_tokens,
        including_special=False,
    ):
        local_counter[tok] += 1

    return Counter(local_counter)


def get_counter(*args):
    boundaries, num_workers, input_path, special_tokens = args

    chunks = list(zip(boundaries[:-1], boundaries[1:]))

    final_counter = Counter()
    ctx = mp.get_context("forkserver")
    with ctx.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(input_path, special_tokens),
    ) as pool:

        for counter in pool.imap_unordered(process_chunk, chunks):
            final_counter.update(counter)

    return final_counter


def stream_pretokenize(
    input_path: str,
    special_tokens: list[str],
    chunk_size: int = 8 * 1024 * 1024,
    num_workers: int = None,
    work=None,
) -> Counter:

    if num_workers is None:
        num_workers = cpu_count()

    file_size = os.path.getsize(input_path)

    print(f"file size: {file_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"workers: {num_workers}")

    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(
            f,
            max(1, file_size // chunk_size),
            b"<|endoftext|>",
        )

    return work(boundaries, num_workers, input_path, special_tokens)


class HeapEntry:
    __slots__ = ("count", "pair")
    vocab = None

    def __init__(
        self,
        count: int,
        pair: tuple[int, int],
    ):
        self.count = count
        self.pair = pair

    def __lt__(self, other: "HeapEntry") -> bool:
        if self.count != other.count:
            return self.count > other.count
        first_bytes, second_bytes = self.vocab[self.pair[0]], self.vocab[self.pair[1]]
        if first_bytes != self.vocab[other.pair[0]]:
            return first_bytes > self.vocab[other.pair[0]]
        return second_bytes > self.vocab[other.pair[1]]


import time
from functools import wraps


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 开始计时
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.perf_counter()  # 结束计时

        duration = end_time - start_time
        print(f"函数 {func.__name__} 运行耗时: {duration:.6f} 秒")
        return result

    return wrapper


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
