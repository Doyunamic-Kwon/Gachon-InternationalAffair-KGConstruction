import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from step1_data_loader import load_gold_standard, load_silver_standard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

plt.rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

class REDataset(Dataset):
    def __init__(self, texts, labels, word2idx, label2idx, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        words = str(self.texts[idx]).split()
        seq = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words[:self.max_len]]
        # Padding
        if len(seq) < self.max_len:
            seq += [self.word2idx['<PAD>']] * (self.max_len - len(seq))
            
        label = self.label2idx[self.labels[idx]]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        # Attention Layer
        self.attn_w = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_v = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x) # (B, L, E)
        lstm_out, _ = self.lstm(embedded) # (B, L, 2*H)
        
        # Attention Mechanism
        attn_weights = torch.tanh(self.attn_w(lstm_out)) # (B, L, 2*H)
        attn_weights = self.attn_v(attn_weights).squeeze(2) # (B, L)
        attn_weights = torch.softmax(attn_weights, dim=1) # (B, L)
        
        # Context Vector
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1) # (B, 2*H)
        
        out = self.fc(context) # (B, num_classes)
        return out, attn_weights

def run_deep_learning_pipeline():
    print("--- 🚀 Step 4. Deep Learning RE 파이프라인 (Bi-LSTM + Attention) ---")
    
    gold_df = load_gold_standard()
    silver_df = load_silver_standard()
    
    texts = []
    labels = []
    
    for _, row in gold_df.iterrows():
        texts.append(row['marked_text'])
        labels.append(row['final_relation'])
        
    for _, row in silver_df.iterrows():
        rel = row.get('relation', '')
        if rel and rel != "NA":
            texts.append(row.get('marked_text', ''))
            labels.append(rel)
            
    print(f"총 학습/평가 데이터: {len(labels)}건")

    # 먼저 분리 → 단어 사전은 train에서만 구축 (test 어휘 누출 방지)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 단어 사전 구축 (train만 사용)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for text in X_train:
        for word in str(text).split():
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    # 라벨 사전 구축
    unique_labels = sorted(list(set(labels)))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    train_dataset = REDataset(X_train, y_train, word2idx, label2idx)
    test_dataset = REDataset(X_test, y_test, word2idx, label2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # MacOS 용 MPS 지원 (선택사항)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    model = BiLSTMAttention(vocab_size=len(word2idx), embed_dim=128, hidden_dim=64, num_classes=len(label2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print(f"\n모델 학습 시작 (Device: {device})...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            out, _ = model(seq)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        
    # 평가
    print("\n모델 평가 중...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seq, label in test_loader:
            seq, label = seq.to(device), label.to(device)
            out, _ = model(seq)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"▶ Deep Learning (Bi-LSTM+Attention) Macro F1-Score: {macro_f1:.4f}")
    
    # Attention Heatmap 시각화
    print("\nAttention Heatmap 시각화 생성 중...")
    # 테스트 셋에서 관계가 있는 첫 번째 문장 선택
    sample_idx = 0
    for i, l in enumerate(y_test):
        if l != "NO_RELATION":
            sample_idx = i
            break
            
    sample_text = X_test[sample_idx]
    sample_label = y_test[sample_idx]
    
    words = str(sample_text).split()[:50]
    seq = [word2idx.get(w, word2idx['<UNK>']) for w in words]
    seq += [0] * (50 - len(seq))
    
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    out, attn_weights = model(seq_tensor)
    pred_idx = torch.argmax(out, dim=1).item()
    pred_label = idx2label[pred_idx]
    
    attn_weights = attn_weights.cpu().detach().numpy()[0][:len(words)]
    
    plt.figure(figsize=(14, 4))
    sns.heatmap([attn_weights], xticklabels=words, yticklabels=['Attention Weight'], cmap='Reds', annot=True, fmt=".2f", cbar=False)
    plt.title(f"Attention Heatmap  |  True: {sample_label}  |  Pred: {pred_label}", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300)
    print("✅ attention_heatmap.png 저장 완료")

if __name__ == "__main__":
    run_deep_learning_pipeline()
