
import regex
from collections import Counter, defaultdict
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_initial_corpus_counts(text: str) -> Counter:
    words = regex.findall(PAT, text)
    counts = Counter()
    for word in words:
        counts[tuple(word.encode('utf-8'))] += 1
    return counts


def get_status(corpus_counts: Counter):
    pair_counts = defaultdict(int)
    for word_tuple, frequency in corpus_counts.items():
        for i in range(len(word_tuple)-1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] += frequency


def train_bpe(input_path: str, vocab_size: int, specil_tokens: list[str]):
    """bpe构建词表"""
    with open(input_path, 'rb') as f:
        text = f.read().decode("utf-8", errors="replace")
    words = regex.findall(PAT, text)
    corpus_counts = Counter(tuple(w.encode("utf-8")) for w in words)
    vocab = {i: bytes([i]) for i in range(256)}

    for special_token in specil_tokens:
        new_idx = len(vocab)
        vocab[new_idx] = special_token.encode('utf-8')

    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        pairs_status = defaultdict(int)
        for word_tuple, frequency in corpus_counts.items():
            for pair in zip(word_tuple, word_tuple[1:]):
                pairs_status[pair] += frequency

        if not pairs_status:
            break
        best_pair = max(pairs_status.items(), key=lambda x: (x[1], x[0]))[0]

        p1_bytes, p2_bytes = vocab[best_pair[0]], vocab[best_pair[1]]
        merges.append((p1_bytes, p2_bytes))

        new_id = len(vocab)
        vocab[new_id] = p1_bytes + p2_bytes

        corpus_counts = _merge_corpus(corpus_counts, best_pair, new_id)
    return vocab, merges


def _merge_corpus(corpus_counts, pair, new_id):
    """更新词频字典中的每个元组"""
    new_counts = {}
    for word_tuple, freq in corpus_counts.items():
        new_tuple = _apply_merge(word_tuple, pair, new_id)
        new_counts[new_tuple] = freq
    return new_counts


def _apply_merge(word_tuple, pair, new_id):
    """在单个元组内部执行合并"""
    new_word = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i+1] == pair[1]:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    return tuple(new_word)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # 建立特殊标记的字符串到 ID 的映射
        self.special_tokens_map = {
            token: i + 256
            for i, token in enumerate(special_tokens)
        }

        # 建立合并规则的快速查找表：(id1, id2) -> new_id
        self.byte_toid = {byte: id for id, byte in self.vocab.items()}
        self.id_merges: list[tuple[int, int, int]] = []
        for p1_bytes, p2_bytes in merges:
            id1 = self.byte_toid[p1_bytes]
            id2 = self.byte_toid[p2_bytes]
            new_id = self.byte_toid[p1_bytes + p2_bytes]
            self.id_merges.append((id1, id2, new_id))

    def encode(self, text: str) -> list[int]:
        # TODO: 第一步依然是正则分词
        if self.special_tokens:
            special_pattern = "(" + "|".join(regex.escape(t)
                                             for t in self.special_tokens) + ")"
            parts = regex.split(special_pattern, text)
        else:
            parts = [text]
        res = []
        for part in parts:
            if part in self.special_tokens_map:
                # 如果是特殊标记，直接添加 ID
                res.append(self.special_tokens_map[part])
            elif part:
                # 如果是普通文本，应用 PAT 正则和 BPE 合并
                words = regex.findall(PAT, part)
                for word in words:
                    word_ids = list(word.encode("utf-8"))
                    word_ids = self._apply_merges(word_ids)
                    res.extend(word_ids)
        return res

    def _apply_merges(self, word_ids: list[int]) -> list[int]:
        for (id1, id2, new_id) in self.id_merges:
            word_ids = self._replace_pair(word_ids, (id1, id2), new_id)
        return word_ids

    def _replace_pair(self, word_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """在 ID 序列中，将所有的 (p1, p2) 替换为 new_id"""
        new_ids = []
        i = 0
        while i < len(word_ids):
            if i < len(word_ids) - 1 and word_ids[i] == pair[0] and word_ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(word_ids[i])
                i += 1
        return new_ids

    def decode(self, ids: list[int]) -> str:
        # 实现 ID 到字符串的还原
        bytes_list = []
        for id in ids:
            bytes_list.append(self.vocab[id])
        all_bytes = b"".join(bytes_list)
        return all_bytes.decode(encoding='utf-8', errors='replace')


if __name__ == '__main__':
    # 测试脚本
    special = ["<|endoftext|>"]
    # 模拟训练返回的数据
    mock_vocab = {i: bytes([i]) for i in range(256)}
    mock_vocab[256] = b"<|endoftext|>"
    # 假设有一个合并规则：'h'(104) + 'i'(105) -> 257
    mock_merges = [(b'h', b'i')]
    mock_vocab[257] = b"hi"

    tokenizer = Tokenizer(mock_vocab, mock_merges, special)

    # 测试 1: 普通文本合并
    print(f"Encode 'hi': {tokenizer.encode('hi')}")  # 应输出 [257]

    # 测试 2: 特殊标记保护
    # 应输出 [257, 256]
    print(f"Encode with special: {tokenizer.encode('hi<|endoftext|>')}")
