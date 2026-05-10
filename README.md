# DA6401 Assignment 3 — Attention Is All You Need
 
## Repository Link

https://github.com/SumeetD001/DA6401_Assignment_3

---

## W&B Report

https://wandb.ai/sumeet01-iitmaana/da6401-a3-1/reports/Task-2-WandB-Report-for-Assignment-3--VmlldzoxNjgyOTUzNA?accessToken=iupdp6sbm9y5kxqbww7obw4ikpk2l8nd0tc4kiwuyzli3pquaipbhv6qjed4ioq9
 
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
