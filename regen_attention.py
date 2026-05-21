"""
Standalone script: regenerate docs/attention_heatmap.png
with latest model (English-only text, save to docs/).
Uses same BiLSTMAttention architecture as step4_deep_learning_re.py.
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("docs", exist_ok=True)
plt.rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

from step1_data_loader import load_gold_standard, load_silver_standard

print("Loading data...")
gold_df   = load_gold_standard()
silver_df = load_silver_standard()
silver_valid = silver_df.dropna(subset=['relation'])
silver_valid = silver_valid[silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')]

texts, labels = [], []
for _, row in gold_df.iterrows():
    texts.append(row['marked_text']); labels.append(row['final_relation'])
for _, row in silver_valid.iterrows():
    rel = row.get('relation', '')
    if rel and rel != 'NA':
        texts.append(row.get('marked_text', '')); labels.append(rel)

X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42)

word2idx = {'<PAD>': 0, '<UNK>': 1}
for t in X_tr:
    for w in str(t).split():
        if w not in word2idx:
            word2idx[w] = len(word2idx)

unique_labels = sorted(set(labels))
label2idx = {l: i for i, l in enumerate(unique_labels)}
idx2label = {i: l for l, i in label2idx.items()}
MAX_LEN = 50

class REDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts; self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        ws  = str(self.texts[i]).split()
        seq = [word2idx.get(w, 1) for w in ws[:MAX_LEN]] + [0]*(MAX_LEN - min(len(ws), MAX_LEN))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label2idx[self.labels[i]], dtype=torch.long)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn_w = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_v = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_w = torch.tanh(self.attn_w(lstm_out))
        attn_w = self.attn_v(attn_w).squeeze(2)
        attn_w = torch.softmax(attn_w, dim=1)
        context = torch.bmm(attn_w.unsqueeze(1), lstm_out).squeeze(1)
        return self.fc(context), attn_w

device = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

tr_loader = DataLoader(REDataset(X_tr, y_tr), batch_size=32, shuffle=True)
te_loader = DataLoader(REDataset(X_te, y_te), batch_size=32)

model = BiLSTMAttention(len(word2idx), 128, 64, len(unique_labels)).to(device)
opt  = optim.Adam(model.parameters(), lr=5e-3)
crit = nn.CrossEntropyLoss()

EPOCHS = 10
print("Training...")
for ep in range(EPOCHS):
    model.train(); total = 0
    for seq, lbl in tr_loader:
        seq, lbl = seq.to(device), lbl.to(device)
        opt.zero_grad()
        out, _ = model(seq)
        loss = crit(out, lbl); loss.backward(); opt.step()
        total += loss.item()
    print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {total/len(tr_loader):.4f}")

model.eval()
preds, trues = [], []
with torch.no_grad():
    for seq, lbl in te_loader:
        out, _ = model(seq.to(device))
        preds.extend(torch.argmax(out,1).cpu().numpy())
        trues.extend(lbl.numpy())
macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
print(f"Macro F1: {macro_f1:.4f}")

# Sample text for attention heatmap — pick first non-NO_RELATION test sample
sample_idx = next((i for i, l in enumerate(y_te) if l != 'NO_RELATION'), 0)
sample_text  = X_te[sample_idx]
sample_label = y_te[sample_idx]
words = str(sample_text).split()[:MAX_LEN]
seq   = [word2idx.get(w, 1) for w in words] + [0]*(MAX_LEN - len(words))
with torch.no_grad():
    out, attn = model(torch.tensor([seq], dtype=torch.long).to(device))
pred_label   = idx2label[torch.argmax(out,1).item()]
weights      = attn.cpu().numpy()[0][:len(words)]
short_words  = [w[:8] for w in words]

plt.figure(figsize=(14, 4))
sns.heatmap([weights], xticklabels=short_words, yticklabels=['Attention'],
            cmap='Reds', annot=True, fmt=".2f", cbar=False)
plt.title(f"Attention Heatmap  |  True: {sample_label}  |  Pred: {pred_label}",
          fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig("docs/attention_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ docs/attention_heatmap.png saved  (F1={macro_f1:.4f})")
