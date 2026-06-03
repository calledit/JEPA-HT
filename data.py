import torch
from torch.utils.data import IterableDataset

from config import Config


_FINEWEB_VAL_DOCS = 500
_FINEWEB_SHUFFLE_BUFFER = 10_000


class GPT2Tokenizer:
    """Wraps tiktoken GPT-2 BPE encoder."""
    vocab_size = 50257

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self.eot_token: int = self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids) -> str:
        return self._enc.decode([int(i) for i in ids])


class SequenceDataset(IterableDataset):
    """Yields fixed-length token ID tensors from a HuggingFace streaming dataset."""

    def __init__(self, hf_dataset, tokenizer: GPT2Tokenizer, sequence_length: int, window_size: int):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.docs_consumed = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.hf_dataset
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        buf: list[int] = []
        for doc in dataset:
            buf.extend(self.tokenizer.encode(doc["text"]))
            buf.extend([self.tokenizer.eot_token] * self.window_size)
            self.docs_consumed += 1
            while len(buf) >= self.sequence_length:
                yield torch.tensor(buf[: self.sequence_length], dtype=torch.long)
                del buf[: self.sequence_length]


def _build_fineweb_dataset(cfg: Config, skip_docs: int = 0):
    from datasets import load_dataset

    print("Loading FineWeb-Edu (streaming)...")
    tokenizer = GPT2Tokenizer()

    def _stream():
        return load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )

    print(f"  Building val set from first {_FINEWEB_VAL_DOCS} docs...")
    val_tokens: list[int] = []
    for doc in _stream().take(_FINEWEB_VAL_DOCS):
        val_tokens.extend(tokenizer.encode(doc["text"]))
        val_tokens.append(tokenizer.eot_token)
    val_data = torch.tensor(val_tokens, dtype=torch.long)

    train_stream = _stream().skip(_FINEWEB_VAL_DOCS + skip_docs).shuffle(
        buffer_size=_FINEWEB_SHUFFLE_BUFFER
    )
    if skip_docs:
        print(f"  Skipping {skip_docs:,} previously consumed documents")
    print(f"  Val: {len(val_tokens):,} tokens | Train: streaming")

    train_dataset = SequenceDataset(train_stream, tokenizer, cfg.sequence_length, cfg.window_size)
    return train_dataset, val_data, tokenizer


def build_dataset(cfg: Config, skip_docs: int = 0):
    if cfg.dataset == "fineweb_edu":
        return _build_fineweb_dataset(cfg, skip_docs)
    raise ValueError(f"Unknown dataset: {cfg.dataset!r}")
