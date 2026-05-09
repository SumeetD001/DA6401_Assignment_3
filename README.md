# DA6401 Assignment 3 — Attention Is All You Need
 
PyTorch implementation of the Transformer architecture from  
[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) for German→English translation on the Multi30k dataset.
 
---
 
## Project Structure
 
```
assignment3/
├── model.py          # Transformer architecture (MHA, PE, Encoder, Decoder)
├── dataset.py        # Multi30k loading, spaCy tokenisation, Vocab
├── lr_scheduler.py   # Noam learning-rate scheduler
├── train.py          # Training loop, greedy decode, BLEU eval, checkpointing
├── requirements.txt
└── README.md
```
 
---
 
## Setup
 
```bash
# 1. Install dependencies
pip install -r requirements.txt
 
# 2. Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
 
# 3. NLTK data (for BLEU)
python -c "import nltk; nltk.download('punkt')"
```
 
---
 
## Training
 
```bash
python train.py
```
 
Trains for 15 epochs with Noam scheduler, saves `checkpoint_best.pt` and `checkpoint_latest.pt`.
 
---
 
## Inference
 
```python
from model import Transformer
model = Transformer()   # downloads weights from Google Drive automatically
print(model.infer("Ein Mann geht spazieren."))
# → "a man is walking ."
```
 
---
