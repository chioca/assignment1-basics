from collections import Counter, defaultdict

GPT2_TOKENIZER_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


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


def init_vacab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
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


def pair_counts(word_counter: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    for ids, count in word_counter.items():
        for pair in zip(ids, ids[1:]):
            pairs[pair] += count

    return pairs


def get_most_frequent_pair(pair_counter: dict[tuple[int, int], int]) -> tuple[int, int]:
    max_freq = max(pair_counter.values())
    candidates = [pair for pair, freq in pair_counter.items() if freq == max_freq]
    res = max(candidates)

    return res


def add_pair_to_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    index1, index2 = pair
    vocab[len(vocab)] = vocab[index1] + vocab[index2]
    return len(vocab) - 1


def merge_pair_ids(
    word_counter: dict[tuple[bytes] | tuple[int], int],
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

        new_word_counter[tuple(new_word)] += 1
