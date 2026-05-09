import math
import copy
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

GDRIVE_FILE_ID  = "1tGSS4ytD7XcwiR0PEFBtIeQHDcj54fJe" 
CHECKPOINT_PATH = "checkpoint_best.pt"         


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0) 
    return torch.matmul(attn_w, V), attn_w


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    tgt_len  = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal   = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def _split(self, x):
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        b, _, s, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split(self.W_q(query))
        K = self._split(self.W_k(key))
        V = self._split(self.W_v(value))
        out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        return self.W_o(self._merge(out))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
      

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    UNK_IDX = 0
    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.1,
        _src_vocab=None,
        _tgt_vocab=None,
    ) -> None:
        super().__init__()

        _state_dict = None
        if src_vocab_size is None or tgt_vocab_size is None:
            ckpt           = self._download_and_load_checkpoint()
            cfg            = ckpt["model_config"]
            src_vocab_size = cfg["src_vocab_size"]
            tgt_vocab_size = cfg["tgt_vocab_size"]
            d_model        = cfg.get("d_model",   d_model)
            N              = cfg.get("N",          N)
            num_heads      = cfg.get("num_heads",  num_heads)
            d_ff           = cfg.get("d_ff",       d_ff)
            dropout        = cfg.get("dropout",    dropout)
            _src_vocab     = ckpt.get("src_vocab", _src_vocab)
            _tgt_vocab     = ckpt.get("tgt_vocab", _tgt_vocab)
            _state_dict    = ckpt["model_state_dict"]

        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding  = PositionalEncoding(d_model, dropout)

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.fc_out        = nn.Linear(d_model, tgt_vocab_size)
        self.fc_out.weight = self.tgt_embedding.weight

        self._init_weights() # Xavier

        if _state_dict is not None:
            self.load_state_dict(_state_dict)

        self.src_vocab  = _src_vocab
        self.tgt_vocab  = _tgt_vocab
        self._spacy_de  = None
      

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _download_and_load_checkpoint() -> dict:
        if not os.path.exists(CHECKPOINT_PATH):
            try:
                import gdown
            except ImportError:
                import subprocess, sys
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "gdown", "--quiet"],
                    check=True,
                )
                import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            print(f"Downloading checkpoint → {CHECKPOINT_PATH}")
            gdown.download(url, CHECKPOINT_PATH, quiet=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    def _get_spacy_de(self):
        if self._spacy_de is None:
            import spacy
            try:
                self._spacy_de = spacy.load("de_core_news_sm")
            except OSError:
                self._spacy_de = spacy.blank("de")
        return self._spacy_de

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder stack + projection.  → logits: [B, tgt_len, tgt_vocab]"""
        x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, german_sentence: str, max_len: int = 100) -> str:
        self.eval()
        device = next(self.parameters()).device
        spacy_de = self._get_spacy_de()
        tokens   = [tok.text.lower() for tok in spacy_de.tokenizer(german_sentence)]

        src_ids  = [self.SOS_IDX] + self.src_vocab.lookup_indices(tokens) + [self.EOS_IDX]
        src      = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=self.PAD_IDX).to(device)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys     = torch.tensor([[self.SOS_IDX]], dtype=torch.long, device=device)

            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=self.PAD_IDX).to(device)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == self.EOS_IDX:
                    break

        out_tokens = []
        for idx in ys.squeeze(0).tolist():
            if idx in (self.SOS_IDX, self.PAD_IDX):
                continue
            if idx == self.EOS_IDX:
                break
            out_tokens.append(self.tgt_vocab.lookup_token(idx))

        return " ".join(out_tokens)
