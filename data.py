import os
import torch
from torch.utils.data import IterableDataset

from config import Config

_HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")


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
    """Yields fixed-length byte ID tensors from a HuggingFace streaming dataset.

    Each yielded tensor starts at the beginning of a document chunk — never at
    padding zeros. Docs shorter than sequence_length are zero-padded at the end.
    Docs longer than sequence_length are split into non-overlapping chunks.
    """

    def __init__(self, hf_dataset, tokenizer: ByteTokenizer, sequence_length: int, skip_docs: int = 0):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.docs_consumed = skip_docs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.hf_dataset
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        for doc in dataset:
            raw = self.tokenizer.encode(doc["text"])
            if not raw:
                continue
            self.docs_consumed += 1
            for start in range(0, len(raw), self.sequence_length):
                chunk = raw[start : start + self.sequence_length]
                if len(chunk) < self.sequence_length:
                    chunk = chunk + [self.tokenizer.eot_token] * (self.sequence_length - len(chunk))
                yield torch.tensor(chunk, dtype=torch.long)


def _build_fineweb_dataset(cfg: Config, skip_docs: int = 0):
    from datasets import load_dataset

    print("Loading FineWeb-Edu (streaming)...")
    tokenizer = ByteTokenizer()

    def _stream():
        return load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            cache_dir=_HF_CACHE_DIR,
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

    train_dataset = SequenceDataset(train_stream, tokenizer, cfg.sequence_length, skip_docs=skip_docs)
    return train_dataset, val_data, tokenizer


def build_dataset(cfg: Config, skip_docs: int = 0):
    if cfg.dataset == "fineweb_edu":
        return _build_fineweb_dataset(cfg, skip_docs)
    raise ValueError(f"Unknown dataset: {cfg.dataset!r}")
