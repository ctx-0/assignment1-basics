import regex as re
from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# TODO: optimize
def pre_tokenize(corpus: str) -> list[str]:
    return re.findall(PAT, corpus)


def get_pre_token_freq(pre_tokens):
    d = defaultdict(int)
    for t in pre_tokens:
        d[tuple(t.encode("utf-8"))] += 1
    return d


def get_byte_pair_freq(d):
    byte_pair_freq = defaultdict(int)
    for tok_bytes, freq in d.items():
        for i in range(len(tok_bytes) - 1):
            byte_pair_freq[(tok_bytes[i], tok_bytes[i + 1])] += freq
    return byte_pair_freq


def merge(d, best_pair, vocab_idx):
    new_d = defaultdict(int)

    for tok_bytes, freq in d.items():
        new_bytes = []
        i = 0
        while i < len(tok_bytes):
            if i != len(tok_bytes) - 1 and (tok_bytes[i], tok_bytes[i + 1]) == best_pair:
                new_bytes.append(vocab_idx)
                i += 2
            else:
                new_bytes.append(tok_bytes[i])
                i += 1

        new_d[tuple(new_bytes)] += freq
    return new_d


def train(corpus, special_tokens, loops):

    pre_tokens = pre_tokenize(corpus)

    vocab = {i: bytes([i]) for i in range(256)}  # 0..255
    vocab_idx = 256
    merges = []

    d = get_pre_token_freq(pre_tokens)
    byte_pair_freq = get_byte_pair_freq(d)
    if not byte_pair_freq: # edge case
        return vocab, vocab_idx, merges
    best_pair = max(byte_pair_freq, key=byte_pair_freq.get)

    for i in range(loops):
        print(f"{i}th merge:", best_pair, vocab[best_pair[0]], vocab[best_pair[1]])

        merges.append(best_pair)
        vocab[vocab_idx] = vocab[best_pair[0]] + vocab[best_pair[1]]

        d = merge(d, best_pair, vocab_idx)
        vocab_idx += 1

        byte_pair_freq = get_byte_pair_freq(d)
        if not byte_pair_freq:
            break
        best_pair = max(byte_pair_freq, key=byte_pair_freq.get)

    return (vocab, vocab_idx, merges)


if __name__ == "__main__":
    test_corpus = """low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest"""

    special_tokens = ["<|endoftext|>"]

    # vocab, _, merges = train(test_corpus, 60)
    vocab, vocab_idx, merges = train(
        corpus=test_corpus,
        special_tokens=special_tokens,
        loops=60,
    )

    print(f"{merges=}")

    for idx in range(251, 256):
        print(idx, vocab[idx])
    print()
    for idx in range(256, vocab_idx):
        print(idx, vocab[idx])
