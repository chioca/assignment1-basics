from cs336_basics.tokenizer.utils import (
    stream_pretokenize,
    HeapEntry,
    time_it,
    get_counter,
)
from tqdm import tqdm
from collections import Counter, defaultdict
import heapq
import json


def save_to_file(
    vocab: dict[int, bytes],
    merges: tuple[tuple[int, int], int],
    vocab_path: str | None = None,
    merges_path: str | None = None,
):

    vocab_path = "vocab.json" if vocab_path is None else vocab_path
    merges_path = "merges.txt" if merges_path is None else merges_path

    vocab = {k: v.decode("utf-8", errors="surrogateescape") for k, v in vocab.items()}
    merges = [
        str(merge[0][0]) + " " + str(merge[0][1]) + " " + str(merge[1]) + "\n"
        for merge in merges
    ]

    with open(vocab_path, "w+") as f:
        json.dump(vocab, f)

    with open(merges_path, "w+", encoding="utf-8", errors="surrogateescape") as f:
        f.writelines(merges)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    is_for_test: bool = True,
    is_save: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {b: bytes([b]) for b in range(256)}
    vocab_len = len(vocab)

    if special_tokens is not None:
        for special_token in special_tokens:
            if special_token.encode("utf-8") not in vocab.values():
                vocab[vocab_len] = special_token.encode("utf-8")
                vocab_len += 1

    word_cnt = stream_pretokenize(
        input_path, special_tokens, num_workers=8, work=get_counter
    )
    unique_words = list(list(w) for w in word_cnt.keys())
    word_counts = list(word_cnt.values())

    pair_cnt = Counter()
    pair_to_words = defaultdict(set)

    for id, word in enumerate(unique_words):
        for pair in zip(word[:-1], word[1:]):
            cnt = word_counts[id]
            pair_cnt[pair] += cnt
            pair_to_words[pair].add(id)

    HeapEntry.vocab = vocab
    heap = [HeapEntry(cnt, pair) for pair, cnt in pair_cnt.items()]
    heapq.heapify(heap)

    merges = []
    for _ in tqdm(range(len(vocab), vocab_size)):
        if not pair_cnt:
            break

        while heap:
            entry = heap[0]
            current_count = pair_cnt.get(entry.pair, 0)
            if entry.count == current_count and current_count > 0:
                max_pair = entry.pair
                break
            heapq.heappop(heap)

        if max_pair is None or pair_cnt.get(max_pair, 0) <= 0:
            break

        new_idx = len(vocab)

        vocab[new_idx] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((max_pair, new_idx))

        affected_indices = pair_to_words[max_pair]
        for idx in list(affected_indices):
            word = unique_words[idx]
            count = word_counts[idx]

            for i in range(len(word) - 1):
                old_p = (word[i], word[i + 1])
                pair_cnt[old_p] -= count
                pair_to_words[old_p].discard(idx)

                new_count = pair_cnt[old_p]
                if new_count > 0:
                    heapq.heappush(heap, HeapEntry(new_count, old_p))

            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                    new_word.append(new_idx)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            for i in range(len(new_word) - 1):
                new_p = (new_word[i], new_word[i + 1])
                pair_cnt[new_p] += count
                pair_to_words[new_p].add(idx)
                heapq.heappush(heap, HeapEntry(pair_cnt[new_p], new_p))

            unique_words[idx] = new_word

        if max_pair in pair_cnt and pair_cnt[max_pair] <= 0:
            del pair_cnt[max_pair]
        pair_to_words.pop(max_pair, None)
    ret_merges = []
    for merge in merges:
        ret_merges.append((vocab[merge[0][0]], vocab[merge[0][1]]))
    if is_for_test:
        return vocab, ret_merges
    if is_save:
        save_to_file(vocab, merges)
    return vocab, merges


if __name__ == "__main__":
    input_path = "/home/chioca/assignment1-basics/data/owt_train.txt"
    vocab, merges = train_bpe(
        input_path, 10_000, ["<|endofeof|>"], is_for_test=False, is_save=True
    )
    print(merges)
