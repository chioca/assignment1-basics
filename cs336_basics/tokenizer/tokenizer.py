from collections import Counter, defaultdict
from multiprocessing import Process, Queue
import regex as re
import os
from cs336_basics.tokenizer.utils import print_color, find_chunk_boundaries, timeit
from cs336_basics.tokenizer.merge_fn import (
    build_pair_heap,
    pop_most_frequent_pair,
    merge_pairs_with_heap_index,
)
from tqdm import trange

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
"""
缩写撇号处理：'(?:[sdmt]|ll|ve|re)
匹配内容：常见的英语缩写后缀，如 's, 'd, 'm, 't, 'll, 've, 're。
工程意义：将 I'll 切分为 I 和 'll。如果不单独处理,BPE 可能会把 ll 和后面的单词连在一起，导致词表冗余

单词/字母块： ?\p{L}+
匹配内容：可选的空格 + 一个或多个 Unicode 字母（\p{L}）。
工程意义：这是最常用的分支。它会捕获像 hello 或 world 这样的单词。前面的空格被包含在 Token 内，是 GPT 分词器的一大特点（即“空格前置”逻辑）

数字块： ?\p{N}+
匹配内容：可选的空格 + 一个或多个 Unicode 数字（\p{N}）。

工程意义：确保数字（如 123)被视作独立的单元,防止数字与字母混合在一起。

标点与符号： ?[^\s\p{L}\p{N}]+
匹配内容：可选的空格 + 一个或多个非空白、非字母、非数字的字符。
工程意义：匹配标点符号、特殊符号（如 !!!, @@@）。这保证了标点符号不会和单词粘连。

空白符处理：\s+(?!\S)|\s+
匹配内容：
\s+(?!\S)：匹配结尾处的空白符。
\s+：匹配多余的空格。
工程意义：确保连续的空格能被正确捕捉，而不会被丢弃，这对于代码缩进或特定格式的文本至关重要。
"""


def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    """初始化词汇表

    Args:
        special_tokens (list[str] | None, optional): 特殊token. Defaults to None.

    Returns:
        dict[int, bytes]: 词汇表
    """
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    counter_index = 256
    if isinstance(special_tokens, list):
        for token in special_tokens:
            vocab[counter_index] = token.encode("utf-8")
            counter_index += 1

    return vocab


def split_by_special_tokens(
    text: str, special_tokens: list[str], including_special: bool = False
) -> list[str]:
    if special_tokens is None:
        return [text]

    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if including_special:
        special_chunks = re.split(f"({pattern})", text)
    else:
        special_chunks = re.split(pattern, text)

    return special_chunks


def pre_tokenize(
    text: str,
    special_tokens: list[str],
    including_special: bool = False,
) -> Counter:

    word_counts = Counter()
    chunks = split_by_special_tokens(text, special_tokens, including_special)

    for chunk in chunks:
        # special token：作为原子处理
        if chunk in special_tokens:
            if including_special:
                word_counts[tuple(chunk.encode("utf-8"))] += 1
            continue

        # 普通文本：正则切分
        for match in re.finditer(PAT, chunk):
            word = match.group(0)
            word_encoded = tuple(word.encode("utf-8"))
            word_counts[word_encoded] += 1

    return word_counts


def pair_counts(word_counter: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    for ids, count in word_counter.items():
        for pair in zip(ids, ids[1:]):
            pairs[pair] += count

    return pairs


def get_most_frequent_pair(pair_counter: dict[tuple[int, int], int]) -> tuple[int, int]:
    max_freq = max(pair_counter.values())
    res = max(pair_counter.items(), key=lambda x: (x[1], x[0]))[0]

    return res


def add_pair_to_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    index1, index2 = pair
    vocab[len(vocab)] = vocab[index1] + vocab[index2]
    return len(vocab) - 1


def merge_pair_ids(
    word_counter: dict[tuple[int], int],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int], int], dict[tuple[int, int], int]]:
    new_word_counter: dict[tuple[int], int] = defaultdict(int)
    new_pair_counter: dict[tuple[int, int], int] = defaultdict(int)

    for ids, frequency in word_counter.items():
        i = 0
        new_word = []
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(ids[i])
                i += 1
        for _pair in zip(new_word, new_word[1:]):
            new_pair_counter[_pair] += frequency
        new_word_counter[tuple(new_word)] += frequency

    return new_word_counter, new_pair_counter


def update_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    new_id = len(vocab)
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return new_id


def pre_tokenize_string_worker(*args):
    input_path, special_tokens, queue, start, end, include_special = args
    with open(input_path, "rb") as f:
        f.seek(start)
        # chunk = f.read(end - start).decode(encoding="utf-8", errors="ignore")
        raw = f.read(end - start)
    chunk = raw.decode("utf-8")
    word_counter = pre_tokenize(chunk, special_tokens, include_special)
    queue.put(word_counter)


@timeit
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
    **kwargs,
):

    with open(input_path, "rb") as f:
        num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 1. Pre-tokenization
    # 1.1 Find chunk boundaries
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=kwargs.get("desired_num_chunks", 5),
            split_special_token=b"\n",
        )

    if verbose:
        print_color(
            f"Identified {len(chunk_boundaries) - 1} chunks for pre-tokenization."
        )

    # 1.2 Count word frequencies across chunks using multiprocessing
    queue = Queue()
    processes: list[Process] = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string_worker,
            args=(input_path, special_tokens, queue, start, end, False),
        )
        processes.append(p)
        p.start()

    word_counter = Counter()
    for _ in range(len(processes)):
        try:
            partial_counter = queue.get()
            word_counter.update(partial_counter)
        except:
            continue
    for p in processes:
        p.join()

    pairs_counter = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word in word_counter:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_to_words[pair].add(word)
            pairs_counter[pair] += word_counter[word]

    pair_heap = build_pair_heap(pairs_counter, vocab)
    for i in trange(num_merges):
        most_frequent_pair = pop_most_frequent_pair(pair_heap, pairs_counter)
        new_id = update_vocab(vocab, most_frequent_pair)

        word_counter, pairs_counter, pair_heap, pair_to_words = (
            merge_pairs_with_heap_index(
                word_counter,
                pairs_counter,
                most_frequent_pair,
                new_id,
                vocab,
                pair_heap,
                pair_to_words,
            )
        )

        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

    # if kwargs.get("save_path"):
    #     save_vocab_and_merges(vocab, merges, kwargs["save_path"])
    #     with open(
    #         os.path.join(kwargs["save_path"], "special_tokens.txt"),
    #         "w",
    #         encoding="utf-8",
    #     ) as f:
    #         if special_tokens:
    #             for token in special_tokens:
    #                 f.write(f"{token}\n")

    return vocab, merges


string = """ 
low low low low low <|endoftext|>
lower lower widest widest widest <|endoftext|>
newest newest newest newest newest newest 
"""
special_tokens: list[str] = ["<|endoftext|>"]


# def train_bpe(
#     string: str = string,
#     vocab_size: int = 263,
#     special_tokens: list[str] = special_tokens,
#     save_path: str | None = None,
# ):
#     # 初始化词汇表
#     vocab = init_vocab(special_tokens)
#     word_counter = pre_tokenize(string, special_tokens)
#     pair_freqs = pair_counts(word_counter)
#     num_train = vocab_size - len(vocab)
#     merges: list[tuple[int, int, int]] = []
#     for i in range(num_train):
#         pair = get_most_frequent_pair(pair_freqs)
#         new_id = add_pair_to_vocab(vocab, pair)
#         word_counter, pair_freqs = merge_pair_ids(word_counter, pair, new_id)
#         merges.append((pair, new_id))
#         print(f"the {i+1} epoch is {pair[0]} + {pair[1]} -> {new_id}")
#     return (vocab, word_counter, merges)


# vocab, word_counter, merges = train_bpe(
#     string=string,
#     vocab_size=256 + 1 + 6,
#     special_tokens=special_tokens,
# )

# print(vocab, word_counter, merges)
