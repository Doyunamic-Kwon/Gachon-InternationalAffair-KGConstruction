"""
Step 5: Deep Learning (Bi-LSTM + Attention)
────────────────────────────────────────────
Updated: corpus_clean.jsonl 1730건 기반
Train/Val/Test Split:
  - Train: Gold 231 + Silver 948 = 1,179건 (68.2%)
  - Val:   Gold 26 (1.5%) - early stopping
  - Test:  Template 525 (30.3%) - OOD evaluation
"""

import os, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

plt.rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

BATCH_SIZE = 32
EPOCHS = 30
HIDDEN_DIM = 100
EMBEDDING_DIM = 100
DROPOUT = 0.3
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Device: {DEVICE}")

# ═══════════════════════════════════════════════════
# 1. DATA LOADING & SPLIT
# ═══════════════════════════════════════════════════

def load_corpus_with_split():
    """
    Load corpus_clean.jsonl and split into train/val/test
    Train/Val: Gold (vocab consistency)
    Test: Template (OOD evaluation)
    """
    with open("data/re_fixed_v6/corpus_clean.jsonl") as f:
        corpus = [json.loads(line) for line in f]
    
    # Source별 분리 (첫 257개가 gold로 가정)
    gold_rows = corpus[:257]
    silver_rows = corpus[257:]
    
    # Gold: 90% train, 10% val
    np.random.seed(42)
    gold_indices = np.random.permutation(len(gold_rows))
    train_idx = gold_indices[:int(len(gold_rows) * 0.9)]  # 231
    val_idx = gold_indices[int(len(gold_rows) * 0.9):]    # 26
    
    train_gold = [gold_rows[i] for i in train_idx]
    val_gold = [gold_rows[i] for i in val_idx]
    
    # Silver: all train
    train_silver = silver_rows
    
    # Separate template (문맥이 template인 것 = context_src == "template")
    test_template = [r for r in corpus if r.get("context_src") == "template"]
    
    # Fallback: 컨텍스트로 템플릿 식별 불가면, 마지막 525개 사용
    if len(test_template) < 500:
        test_template = corpus[-525:]
    
    train_data = train_gold + train_silver
    val_data = val_gold
    test_data = test_template[:525]
    
    print(f"✓ Train: {len(train_data)}건 (Gold {len(train_gold)} + Silver {len(train_silver)})")
    print(f"✓ Val:   {len(val_data)}건 (Gold)")
    print(f"✓ Test:  {len(test_data)}건 (Template)")
    
    return train_data, val_data, test_data

# ═══════════════════════════════════════════════════
# 2. TOKENIZER & VOCAB
# ═══════════════════════════════════════════════════

class SimpleTokenizer:
    """Character-level + special token aware"""
    
    def __init__(self, vocab_size=1000):
        self.vocab = {}
        self.token2idx = {}
        self.idx2token = {}
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.build_vocab(vocab_size)
    
    def build_vocab(self, max_size):
        """Build vocab from frequent tokens"""
        self.token2idx = {"<PAD>": 0, "<UNK>": 1, "[E1]": 2, "[/E1]": 3, "[E2]": 4, "[/E2]": 5}
        self.idx2token = {v: k for k, v in self.token2idx.items()}
    
    def tokenize(self, text):
        """Split by whitespace + marker aware"""
        text = str(text)
        tokens = re.split(r'(\[/?E[12]\])', text)
        return [t for t in tokens if t.strip()]
    
    def encode(self, text, max_len=100):
        """Convert text to token indices"""
        tokens = self.tokenize(text)
        indices = []
        for token in tokens[:max_len]:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                # Character-level fallback
                for char in token[:10]:
                    indices.append(ord(char) % 256 + 10)
        
        # Pad
        while len(indices) < max_len:
            indices.append(self.PAD_IDX)
        return indices[:max_len]

# ═══════════════════════════════════════════════════
# 3. DATASET & DATALOADER
# ═══════════════════════════════════════════════════

class REDataset(Dataset):
    def __init__(self, data, tokenizer, relation2idx, max_len=100):
        self.data = data
        self.tokenizer = tokenizer
        self.relation2idx = relation2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        marked_text = row.get("marked_text", row.get("sentence", ""))
        relation = row.get("relation", "UNKNOWN")
        
        # Normalize relation label
        rel_key = relation.strip().upper()
        if rel_key not in self.relation2idx:
            rel_key = "NO_RELATION"
        
        tokens = self.tokenizer.encode(marked_text, self.max_len)
        relation_idx = self.relation2idx[rel_key]
        
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "relation": torch.tensor(relation_idx, dtype=torch.long),
            "text": marked_text,
        }

# ═══════════════════════════════════════════════════
# 4. MODEL: Bi-LSTM + Attention
# ═══════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, lstm_out, mask=None):
        """
        lstm_out: (batch, seq_len, hidden_dim*2)
        Returns: (batch, hidden_dim*2)
        """
        scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        weighted = (lstm_out * weights).sum(dim=1)  # (batch, hidden_dim*2)
        return weighted

class BiLSTMRE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True,
            bidirectional=True, dropout=dropout if hidden_dim > 1 else 0
        )
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len)
        Returns: (batch, output_dim)
        """
        mask = (token_ids != 0).bool()  # PAD mask
        
        embedded = self.embedding(token_ids)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch, seq_len, hidden*2)
        
        context = self.attention(lstm_out, mask)  # (batch, hidden*2)
        
        output = self.fc(context)  # (batch, output_dim)
        return output

# ═══════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════

def compute_class_weights(train_data):
    """Compute class weights to handle imbalance"""
    rels = [r.get("relation", "UNKNOWN").upper() for r in train_data]
    counts = Counter(rels)
    total = len(rels)
    weights = {rel: total / (count * len(counts)) for rel, count in counts.items()}
    return weights

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        relations = batch["relation"].to(device)
        
        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, relations)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(relations.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1

def evaluate(model, dataloader, criterion, device, idx2rel):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            relations = batch["relation"].to(device)
            
            logits = model(tokens)
            loss = criterion(logits, relations)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(relations.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, all_preds, all_labels

# ═══════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════

def run_deep_learning():
    print("--- 🚀 Step 5. Deep Learning (Bi-LSTM + Attention) ---\n")
    
    # Load data
    train_data, val_data, test_data = load_corpus_with_split()
    
    # Build relation index
    all_rels = set()
    for data in [train_data, val_data, test_data]:
        for row in data:
            rel = row.get("relation", "UNKNOWN").upper().strip()
            all_rels.add(rel)
    
    relation2idx = {rel: idx for idx, rel in enumerate(sorted(all_rels))}
    idx2rel = {idx: rel for rel, idx in relation2idx.items()}
    num_relations = len(relation2idx)
    
    print(f"✓ Relations: {num_relations}개")
    for rel, idx in sorted(relation2idx.items())[:5]:
        print(f"  {idx}: {rel}")
    
    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=5000)
    
    # Datasets
    train_ds = REDataset(train_data, tokenizer, relation2idx)
    val_ds = REDataset(val_data, tokenizer, relation2idx)
    test_ds = REDataset(test_data, tokenizer, relation2idx)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = BiLSTMRE(
        vocab_size=5256,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=num_relations,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Class weights
    class_weights = compute_class_weights(train_data)
    weight_tensor = torch.tensor(
        [class_weights.get(rel, 1.0) for rel in sorted(all_rels)],
        dtype=torch.float
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n▶ Training...")
    best_val_f1 = 0
    patience_counter = 0
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    
    for epoch in range(EPOCHS):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE, idx2rel)
        
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch+1:2d} | "
              f"TrLoss {train_loss:.4f} TrF1 {train_f1:.4f} | "
              f"VaLoss {val_loss:.4f} VaF1 {val_f1:.4f}", end="")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "step5_bilstm_best.pt")
            print(" ✓ (best)")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load("step5_bilstm_best.pt"))
    
    # Test evaluation
    print(f"\n▶ Test Evaluation...")
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE, idx2rel
    )
    
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=sorted(all_rels), zero_division=0))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["train_loss"], label="Train", marker="o")
    axes[0].plot(history["val_loss"], label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training Curve (Loss)")
    axes[0].grid()
    
    # F1
    axes[1].plot(history["train_f1"], label="Train", marker="o")
    axes[1].plot(history["val_f1"], label="Val", marker="s")
    axes[1].axhline(test_f1, color="r", linestyle="--", label=f"Test F1={test_f1:.4f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].set_title("Training Curve (F1)")
    axes[1].grid()
    
    plt.tight_layout()
    plt.savefig("docs/step5_training_curve.png", dpi=300)
    print(f"\n✅ docs/step5_training_curve.png 저장")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Test, F1={test_f1:.4f})")
    plt.xticks(range(num_relations), sorted(all_rels), rotation=45, ha="right")
    plt.yticks(range(num_relations), sorted(all_rels))
    plt.tight_layout()
    plt.savefig("docs/step5_confusion_matrix.png", dpi=300)
    print(f"✅ docs/step5_confusion_matrix.png 저장")
    
    print(f"\n요약:")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Test F1:     {test_f1:.4f}")
    print(f"  Model:       {model.__class__.__name__}")
    
    return test_f1

if __name__ == "__main__":
    run_deep_learning()
