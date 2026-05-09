"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import EOS_IDX, SOS_IDX, PAD_IDX


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    The <pad> token always gets 0 probability; its loss contribution
    is zeroed out so we don't penalise padding positions.

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        # Uniform smoothing mass is spread over all non-pad tokens
        # We use KLDivLoss with a hand-crafted target distribution
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value (mean over non-pad tokens).
        """
        # Build the smoothed target distribution
        # shape: [N, vocab_size]
        smooth_dist = torch.full(
            (target.size(0), self.vocab_size),
            fill_value=self.smoothing / (self.vocab_size - 2),  # -2: exclude pad & true token
            device=logits.device,
        )

        # Assign confidence to the correct class
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out the <pad> column in the target distribution
        smooth_dist[:, self.pad_idx] = 0.0

        # Mask rows where the *target* itself is <pad>
        pad_mask = (target == self.pad_idx)
        smooth_dist[pad_mask] = 0.0

        # KL-div expects log-probabilities for the input
        log_probs = torch.log_softmax(logits, dim=-1)

        loss = self.criterion(log_probs, smooth_dist)

        # Normalise by the number of non-pad tokens
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss / n_tokens


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss per token over the epoch (float).
    """
    model.train(is_train)
    total_loss  = 0.0
    total_steps = 0

    with torch.set_grad_enabled(is_train):
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Teacher forcing: decoder input drops the last token,
            # target labels drop the first (<sos>) token.
            tgt_input  = tgt[:, :-1]   # [batch, tgt_len - 1]
            tgt_labels = tgt[:, 1:]    # [batch, tgt_len - 1]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len - 1, vocab_size]

            # Reshape for loss
            batch_size, tgt_len, vocab_size = logits.shape
            logits_flat  = logits.reshape(-1, vocab_size)     # [N, vocab]
            labels_flat  = tgt_labels.reshape(-1)             # [N]

            loss = loss_fn(logits_flat, labels_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (helps with stability)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss  += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    mode_str = "TRAIN" if is_train else "VALID"
    print(f"Epoch {epoch_num:03d} [{mode_str}]  loss={avg_loss:.4f}")
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    # Encode source once
    memory = model.encode(src, src_mask)   # [1, src_len, d_model]

    # Start with <sos>
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
        logits   = model.decode(memory, src_mask, ys, tgt_mask)
        # logits: [1, cur_len, vocab]

        # Greedy: pick the highest-probability token at the last position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        ys = torch.cat([ys, next_token], dim=1)                      # [1, cur_len+1]

        if next_token.item() == end_symbol:
            break

    return ys   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocab object  (supports .lookup_token(idx)).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    model.eval()
    hypotheses: list = []   # list of token-string lists
    references:  list = []  # list of [token-string list]  (list-of-lists)

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)     # [1, src_len]
                src_mask = make_src_mask(src_i, pad_idx=PAD_IDX).to(device)

                pred_ids = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                ).squeeze(0).tolist()   # list of ints

                # Remove special tokens from prediction
                def ids_to_tokens(ids):
                    tokens = []
                    for idx in ids:
                        if idx in (SOS_IDX, PAD_IDX):
                            continue
                        if idx == EOS_IDX:
                            break
                        tokens.append(tgt_vocab.lookup_token(idx))
                    return tokens

                hyp = ids_to_tokens(pred_ids)

                # Build reference from ground-truth tgt (drop <sos>/<eos>/<pad>)
                ref = ids_to_tokens(tgt[i].tolist())

                hypotheses.append(hyp)
                references.append([ref])   # NLTK expects list-of-lists

    # Compute corpus BLEU (×100 to get percentage)
    smoother   = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses,
                             smoothing_function=smoother) * 100.0
    print(f"Test BLEU: {bleu_score:.2f}")
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saved dict keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    # Collect all constructor kwargs needed to recreate the model
    model_config = {
        "src_vocab_size": model.src_embedding.num_embeddings,
        "tgt_vocab_size": model.tgt_embedding.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
    }

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config":         model_config,
        },
        path,
    )
    print(f"Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded ← {path}  (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Full training experiment:
        1. Init W&B
        2. Build Multi30k dataset / vocabs
        3. Create DataLoaders
        4. Instantiate Transformer
        5. Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. NoamScheduler(warmup_steps=4000)
        7. LabelSmoothingLoss(smoothing=0.1)
        8. Training loop with val and checkpointing
        9. Final BLEU on test set
    """
    import wandb
    from dataset import get_dataloaders
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ────────────────────────────────────────────
    config = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        batch_size   = 128,
        num_epochs   = 15,
        warmup_steps = 4000,
        smoothing    = 0.1,
        min_freq     = 2,
    )

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Data ───────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
    )
    print(f"Src vocab size: {len(src_vocab)}  |  Tgt vocab size: {len(tgt_vocab)}")

    # ── Model ──────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,          # Noam scheduler scales the actual LR
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # ── LR Scheduler ───────────────────────────────────────────────
    scheduler = NoamScheduler(
        optimizer,
        d_model      = cfg.d_model,
        warmup_steps = cfg.warmup_steps,
    )

    # ── Loss ───────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = PAD_IDX,
        smoothing  = cfg.smoothing,
    )

    # ── Training loop ─────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         optimizer.param_groups[0]["lr"],
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch,
                            path="checkpoint_best.pt")

        # Always save latest
        save_checkpoint(model, optimizer, scheduler, epoch,
                        path="checkpoint_latest.pt")

    # ── Final BLEU ─────────────────────────────────────────────────
    # Reload best checkpoint for evaluation
    load_checkpoint("checkpoint_best.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    print(f"\nFinal Test BLEU: {bleu:.2f}")

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()