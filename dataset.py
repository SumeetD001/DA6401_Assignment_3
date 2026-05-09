"""
dataset.py — Multi30k Dataset (De→En) with spaCy tokenisation
DA6401 Assignment 3: "Attention Is All You Need"
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ──────────────────────────────────────────────────────────────────────
# Special token indices (must be consistent everywhere)
# ──────────────────────────────────────────────────────────────────────
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]


# ──────────────────────────────────────────────────────────────────────
# Simple Vocabulary helper
# ──────────────────────────────────────────────────────────────────────

class Vocab:
    """Minimal vocabulary: token ↔ index mapping."""

    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi   # str  → int
        self.itos = itos   # int  → str

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def lookup_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return "<unk>"


def build_vocab_from_counter(counter: Counter, min_freq: int = 2) -> Vocab:
    """Build a Vocab from a token frequency Counter."""
    itos: List[str] = list(SPECIALS)
    for token, freq in sorted(counter.items(), key=lambda x: -x[1]):
        if freq >= min_freq and token not in SPECIALS:
            itos.append(token)
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return Vocab(stoi, itos)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class Multi30kDataset(Dataset):
    """
    Multi30k De→En dataset.

    Loads from HuggingFace (`bentrevett/multi30k`), tokenises with spaCy,
    builds shared vocabularies on the training split, and converts
    sentences to integer index sequences.

    Args:
        split        : 'train', 'validation', or 'test'.
        min_freq     : Minimum token frequency for inclusion in vocab.
        src_vocab    : Pre-built source Vocab (pass for val/test).
        tgt_vocab    : Pre-built target Vocab (pass for val/test).
        max_src_len  : Truncate source sentences longer than this.
        max_tgt_len  : Truncate target sentences longer than this.
    """

    def __init__(
        self,
        split: str = "train",
        min_freq: int = 2,
        src_vocab: Optional[Vocab] = None,
        tgt_vocab: Optional[Vocab] = None,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
    ) -> None:
        self.split       = split
        self.min_freq    = min_freq
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # ── Load raw data ──────────────────────────────────────────
        from datasets import load_dataset
        raw = load_dataset("bentrevett/multi30k", split=split)
        self.raw_de: List[str] = raw["de"]
        self.raw_en: List[str] = raw["en"]

        # ── Load spaCy tokenisers ──────────────────────────────────
        import spacy
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download",
                            "de_core_news_sm"], check=True)
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download",
                            "en_core_web_sm"], check=True)
            self.spacy_en = spacy.load("en_core_web_sm")

        # ── Build or reuse vocabularies ────────────────────────────
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # ── Tokenise & encode ──────────────────────────────────────
        self.src_data, self.tgt_data = self.process_data()

    # ──────────────────────────────────────────────────────────────
    # Tokenisation helpers
    # ──────────────────────────────────────────────────────────────

    def tokenise_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenise_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ──────────────────────────────────────────────────────────────
    # Vocabulary building  (call only on the train split)
    # ──────────────────────────────────────────────────────────────

    def build_vocab(self) -> Tuple[Vocab, Vocab]:
        """
        Builds source (de) and target (en) vocabularies.
        Includes <unk>, <pad>, <sos>, <eos>.
        """
        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()

        for de_sent, en_sent in zip(self.raw_de, self.raw_en):
            src_counter.update(self.tokenise_de(de_sent))
            tgt_counter.update(self.tokenise_en(en_sent))

        src_vocab = build_vocab_from_counter(src_counter, self.min_freq)
        tgt_vocab = build_vocab_from_counter(tgt_counter, self.min_freq)

        return src_vocab, tgt_vocab

    # ──────────────────────────────────────────────────────────────
    # Data processing
    # ──────────────────────────────────────────────────────────────

    def process_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Tokenise every sentence and convert to integer index lists.
        Prepends <sos> and appends <eos>.
        """
        src_data: List[List[int]] = []
        tgt_data: List[List[int]] = []

        for de_sent, en_sent in zip(self.raw_de, self.raw_en):
            src_tokens = self.tokenise_de(de_sent)[: self.max_src_len]
            tgt_tokens = self.tokenise_en(en_sent)[: self.max_tgt_len]

            src_ids = [SOS_IDX] + self.src_vocab.lookup_indices(src_tokens) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + self.tgt_vocab.lookup_indices(tgt_tokens) + [EOS_IDX]

            src_data.append(src_ids)
            tgt_data.append(tgt_ids)

        return src_data, tgt_data

    # ──────────────────────────────────────────────────────────────
    # PyTorch Dataset interface
    # ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src, tgt


# ──────────────────────────────────────────────────────────────────────
# Collate function (used by DataLoader)
# ──────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Pad sequences in a batch to equal length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


# ──────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = 128,
    min_freq: int   = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocab, Vocab]:
    """
    Build train / val / test DataLoaders sharing a common vocabulary.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    train_ds = Multi30kDataset(split="train",      min_freq=min_freq)
    val_ds   = Multi30kDataset(split="validation", min_freq=min_freq,
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)
    test_ds  = Multi30kDataset(split="test",       min_freq=min_freq,
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers)

    return (train_loader, val_loader, test_loader,
            train_ds.src_vocab, train_ds.tgt_vocab)