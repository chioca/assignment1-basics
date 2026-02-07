from collections import Counter, defaultdict
import heapq


class HeapItem:
    def __init__(
        self, neg_freq: int, pair_bytes: tuple[bytes, bytes], pair: tuple[int, int]
    ):
        self.neg_freq = neg_freq
        self.pair_bytes = pair_bytes
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        return self.pair_bytes > other.pair_bytes  # reverse order for max-heap behavior


def build_pair_heap(pairs_freqs: Counter, vocab: dict[int, bytes]):
    heap = []
    for (a, b), f in pairs_freqs.items():
        if f > 0:
            item = HeapItem(-f, (vocab[a], vocab[b]), (a, b))
            heapq.heappush(heap, item)
    return heap


def pop_most_frequent_pair(heap, pairs_counter: Counter) -> tuple[int, int]:
    while heap:
        item = heap[0]  # Peek at the top item
        neg_f = item.neg_freq
        pair = item.pair
        cur_f = pairs_counter.get(pair, 0)
        if (
            cur_f <= 0 or -neg_f != cur_f
        ):  # frequency changed, which means the pair we store in heap is stale
            heapq.heappop(heap)
            continue
        return pair

    raise ValueError("No positive-frequency pairs remain")


def get_new_word(
    word: tuple[int, ...], target_pair: tuple[int, int], new_id: int
) -> tuple[int, ...]:
    new_word: list[int] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == target_pair:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def merge_pairs_with_heap_index(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    target_pair: tuple[int, int],
    new_id: int,
    vocab: dict[int, bytes],
    pair_heap,
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> tuple[
    dict[tuple[int, ...], int],
    Counter,
    list,
    dict[tuple[int, int], set[tuple[int, ...]]],
]:
    new_word_counter: dict[tuple[int, ...], int] = dict(word_counter)
    updated_pair_counter: Counter = pair_counter.copy()
    changed_pairs: set[tuple[int, int]] = set()

    affected_words = list(pair_to_words.get(target_pair, set()))

    for word in affected_words:
        freq = word_counter.get(word, 0)
        if freq <= 0 or len(word) < 2:
            continue
        # 移除受影响的word频率
        del new_word_counter[word]

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            updated_pair_counter[pair] -= freq
            changed_pairs.add(pair)

            s = pair_to_words.get(pair)
            if s is not None:
                s.discard(word)
                if not s:
                    del pair_to_words[pair]

        new_word = get_new_word(word, target_pair, new_id)
        new_word_counter[new_word] = new_word_counter.get(new_word, 0) + freq
        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                updated_pair_counter[pair] += freq
                changed_pairs.add(pair)
                pair_to_words.setdefault(pair, set()).add(new_word)

    if pair_heap is not None:
        for pair in changed_pairs:
            freq = updated_pair_counter.get(pair, 0)
            if freq > 0:
                heapq.heappush(
                    pair_heap, HeapItem(-freq, (vocab[pair[0]], vocab[pair[1]]), pair)
                )

    return dict(new_word_counter), updated_pair_counter, pair_heap, pair_to_words
