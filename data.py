import torch
from torch.utils.data import IterableDataset

from config import Config


_FINEWEB_VAL_DOCS = 500
_FINEWEB_SHUFFLE_BUFFER = 10_000


class ByteTokenizer:
    """Encodes text as raw UTF-8 bytes. Vocabulary is 0–255."""
    vocab_size = 256
    eot_token: int = 0  # null byte as document separator

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes([int(i) for i in ids if i != self.eot_token]).decode("utf-8", errors="replace")


class SequenceDataset(IterableDataset):
    """Yields fixed-length byte ID tensors from a HuggingFace streaming dataset."""

    def __init__(self, hf_dataset, tokenizer: ByteTokenizer, sequence_length: int, window_size: int):
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
    tokenizer = ByteTokenizer()

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
    print(f"  Val: {len(val_tokens):,} bytes | Train: streaming")

    train_dataset = SequenceDataset(train_stream, tokenizer, cfg.sequence_length, cfg.level0_window_size)
    return train_dataset, val_data, tokenizer


def build_dataset(cfg: Config, skip_docs: int = 0):
    if cfg.dataset == "fineweb_edu":
        return _build_fineweb_dataset(cfg, skip_docs)
    raise ValueError(f"Unknown dataset: {cfg.dataset!r}")
